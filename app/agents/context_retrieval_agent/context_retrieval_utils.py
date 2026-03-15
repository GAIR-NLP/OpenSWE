import os
from typing import Dict, List, Any
from loguru import logger
import inspect
import re
from typing import Any
from app.data_structures import MessageThread
from app.model import common
from app.utils import parse_function_invocation
import json
from app.post_process import ExtractStatus, is_valid_json
import itertools
import requests


class RepoBrowseManager:
    def __init__(self, project_path: str):
        self.project_path = os.path.abspath(project_path)  # Ensure absolute path
        self.index: Dict = {}
        self._build_index()

    def _build_index(self):
        """Build the index by parsing the repository structure."""
        self._update_index(self.project_path)

    def _update_index(self, current_path: str):
        """Recursively update the index with files and directories."""
        for root, dirs, files in os.walk(current_path):
            relative_root = os.path.relpath(root, self.project_path)
            current_level = self.index
            if relative_root != ".":  # Handle nested directories
                for part in relative_root.split(os.sep):
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
            for file in files:
                current_level[file] = None  # Mark files as leaf nodes

    def browse_folder(self, path: str, depth: int) -> tuple[str, str, bool]:
        """Browse a folder in the repository from the given path and depth.

        Args:
            path: The folder path to browse, relative to the project root
            depth: How many levels deep to browse the folder structure

        Returns:
            A formatted string showing the folder structure

        Raises:
            ValueError: If the path is outside the project directory
        """
        if not path or path == "/":
            abs_path = self.project_path
        else:
            # Check if the path is an absolute path
            if os.path.isabs(path):
                abs_path = path  # If absolute, use it directly
            else:
                # If relative path, join with project root and convert to absolute
                abs_path = os.path.abspath(os.path.join(self.project_path, path))

        if not abs_path.startswith(self.project_path):
            return "Path does not exist", "Path does not exist", False

        relative_path = os.path.relpath(abs_path, self.project_path)
        if relative_path == ".":
            current_level = self.index
        else:
            current_level = self.index
            for part in relative_path.split(os.sep):
                if part not in current_level:
                    return "Path not found", "Path not found", False  # Path not found
                current_level = current_level[part]

        structure_result = self._get_structure(current_level, depth)
        structure = self._format_structure(structure_result)
        result = f"You are browsing the path: {abs_path}. The browsing Depth is {depth}.\nStructure of this directory:\n\n{self._format_structure(structure_result)}"

        return result, "folder structure collected", True

    def search_files_by_keyword(self, keyword: str) -> tuple[str, str, bool]:
        """Search for files in the repository whose names contain the given keyword.

        Args:
            keyword: The keyword to search for in file names

        Returns:
            tuple: (formatted result string, summary message, success flag)
        """
        matching_files = []
        self._search_index(self.index, keyword, "", matching_files)

        if not matching_files:
            return (
                f"No files found containing the keyword '{keyword}'.",
                "No matching files found",
                True,
            )

        max_files = 50
        if len(matching_files) > max_files:
            result = f"Found {len(matching_files)} files containing the keyword '{keyword}'. Showing the first {max_files}:\n\n"
            matching_files = matching_files[:max_files]
        else:
            result = f"Found {len(matching_files)} files containing the keyword '{keyword}':\n\n"

        formatted_files = "\n".join(
            [f"- {os.path.normpath(file)}" for file in matching_files]
        )
        result += formatted_files
        return result, "File search completed successfully", True

    def _search_index(
        self,
        current_level: Dict,
        keyword: str,
        current_path: str,
        matching_files: List[str],
    ):
        """Recursively search the index for files containing the keyword in their names."""
        for key, value in current_level.items():
            new_path = os.path.join(current_path, key)
            if value is None:  # It's a file
                if keyword.lower() in key.lower():
                    matching_files.append(new_path)
            else:  # It's a directory
                self._search_index(value, keyword, new_path, matching_files)

    def _get_structure(self, structure: Dict, depth: int) -> Dict:
        """Get the structure of the repository from the given path and depth."""
        if depth == 0:
            return {}
        result = {}
        for key, value in structure.items():
            if value is None:  # It's a file
                result[key] = None
            else:  # It's a directory
                result[key] = self._get_structure(value, depth - 1)
        return result

    def _format_structure(self, structure: Dict, indent: int = 0) -> str:
        """Format the structure into a string with proper indentation."""
        result = ""
        for key, value in structure.items():
            if value is None:  # It's a file
                result += "    " * indent + key + "\n\n"
            else:  # It's a directory
                result += "    " * indent + key + "/\n\n"
                result += self._format_structure(value, indent + 1)
        return result

    def browse_file(self, file_path: str) -> str:
        """
        Browse and return up to the first MAX_LINES lines of a file, wrapped in markers.

        Args:
            file_path: Path to the file relative to the project root.

        Returns:
            A string in the form:

            === FILE START: <relative path> ===
            [up to MAX_LINES lines of content]
            --- CONTENT TRUNCATED ---
            === FILE END: <relative path> ===

        Raises:
            ValueError: if the file is outside of project_path
            FileNotFoundError: if the file does not exist
        """
        abs_path = os.path.abspath(file_path)
        if not abs_path.startswith(self.project_path):
            raise ValueError(f"Path '{file_path}' is outside of project directory.")
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"File not found: '{file_path}'")

        MAX_LINES = 1000
        START_MARKER = f"=== FILE START: {file_path} ==="
        END_MARKER = f"=== FILE END:   {file_path} ==="
        TRUNC_MARKER = "--- CONTENT TRUNCATED ---"

        with open(abs_path, "r", encoding="utf-8") as f:
            # read up to MAX_LINES
            lines = list(itertools.islice(f, MAX_LINES))
            content = "".join(lines)
            # check if there’s more
            more = f.readline()
            if more:
                content += "\n" + TRUNC_MARKER

        return "\n".join([START_MARKER, content, END_MARKER])

    def get_webpage_content(self, url: str, timeout: int = 60) -> str:
        """Fetch and return the content of a web page using Jina Reader API.

        Args:
            url: The URL of the web page to fetch
            timeout: Maximum time in seconds to wait for the response (default: 10)

        Returns:
            The content of the web page as a string

        Raises:
            ValueError: If the URL is invalid or the request fails
            TimeoutError: If the request times out
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL - must start with http:// or https://")

        jina_reader_url = f"https://r.jina.ai/{url}"

        try:
            response = requests.get(jina_reader_url, timeout=timeout)
            response.raise_for_status()

            # Validate content type
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                raise ValueError(f"Unsupported content type: {content_type}")

            return response.text

        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to fetch web content: {str(e)}")

    def browse_file_for_environment_info(
        self, file_path: str, custom_query: str = ""
    ) -> tuple[str, str, bool]:
        """Browse a file and extract environment setup information.

        Args:
            repo_browse_manager: Instance for managing repo browsing.
            file_path: The path to the file to browse, relative to the project root.

        Returns:
            A string containing extracted environment setup info.
        """
        try:
            logger.info("entering browse")
            # Step 1: Browse the file content
            file_content = self.browse_file(file_path)
            logger.info(f"{file_content}")
            file_content = (
                f"[File Content: {file_path}]\n{file_content}\n[/File Content]"
            )

            # Step 2: Use LLM to extract environment information
            extracted_info = browse_file_run_with_retries(file_content, custom_query)

            # Step 3: Return extracted information
            return extracted_info, "Get File Info", True

        except ValueError as e:
            logger.info(f"Invalid file path: {str(e)}")
            return "Invalid file path:", "Invalid file path:", False

            # raise
        except FileNotFoundError as e:
            logger.info(f"File not found: {str(e)}")
            return "File not found", "File not found", False

            # raise
        except Exception as e:
            logger.info(f"Unexpected error browsing file: {str(e)}")
            return (
                "Unexpected error browsing file",
                "Unexpected error browsing file",
                False,
            )

            # raise RuntimeError(f"Failed to browse file: {str(e)}") from e


PROXY_PROMPT = """
You are an agent whose job is to:

1. **Extract API calls** from a context-retrieval analysis text.  
2. **Decide whether to terminate** the context-retrieval process.  

---

### Input
The text you receive is **an analysis of the context retrieval process**.  

The text will consist of two parts:
1. **Do we need to collect more context?**  
   - Identify if additional files, folders, or webpages should be browsed for environment setup details.
   - Extract API calls from this section (leave empty if none are needed).

2. **Should we terminate the context retrieval process?**  
   - If all necessary information has been collected, set `"terminate": true`.  
     - You should extract detailed collected information form analyssis of the context retrieval agent. This information will be used by other agent.
   - Otherwise, set `"terminate": false` and provide all collected details.

API List:

- browse_folder(path: str, depth: str): Browse and return the folder structure for a given path in the repository.  The depth is a string representing a number of folder levels to include in the output such as ``1''. 
- browse_file_for_environment_info(file_path: str, custom_query: str): Call an agent to browse a file such as README or CONTRIBUTING.md and extract environment setup and running tests information. Use the `custom_query` parameter to tell the agent any extra details it should pay special attention to (for example, 'pom.xml dependency versions').
- search_files_by_keyword(keyword: str): Search for files in the repository whose names contain the given keyword.

### **IMPORTANT RULES**:
- **Extract all relevant API calls from the text**:
  - If files like `requirements.txt`, `setup.cfg`, `setup.py` are mentioned, call `browse_file_for_environment_info()` on them.
  - If a directory needs exploration, use `browse_folder()`, ensuring `depth` defaults to `"1"` if unspecified.
- If the API call includes a path, the default format should use Linux-style with forward slashes (/).
- Ensure all API calls are valid Python expressions.
- browse_file_for_environment_info("path.to.file") should be written as browse_file_for_environment_info("path/to/file")
- the browse_folder API call MUST include the depth parameter, defaulting to "1" if not provided.
- You MUST ignore the argument placeholders in API calls. For example:
    Invalid Example: browse_folder(path="src", depth=1) 
    Valid Example: browse_folder("src",1)
- Provide your answer in JSON structure like this:
{
    "API_calls": ["api_call_1(args)", "api_call_2(args)", ...],
    "collected_information": <Content of collected information>.
    "terminate": true/false
}

"""


def proxy_apis_with_retries(
    text: str, retries=5
) -> tuple[str | None, list[MessageThread]]:
    msg_threads = []
    for idx in range(1, retries + 1):
        logger.debug(
            "Trying to select search APIs in json. Try {} of {}.", idx, retries
        )

        res_text, new_thread = run_proxy(text)
        msg_threads.append(new_thread)
        res_text = extract_json_from_response(res_text)
        res_text = res_text.lstrip("```json").rstrip("```")
        logger.debug(res_text)
        extract_status, data = is_valid_json(res_text)

        if extract_status != ExtractStatus.IS_VALID_JSON:
            logger.debug("Invalid json. Will retry.")
            continue

        valid, diagnosis = is_valid_response_proxy(data)
        if not valid:
            logger.debug(f"{diagnosis}. Will retry.")
            continue

        logger.debug("Extracted a valid json")
        return res_text, msg_threads
    return None, msg_threads


def run_proxy(text: str) -> tuple[str, MessageThread]:
    """
    Run the agent to extract issue to json format.
    """

    msg_thread = MessageThread()
    msg_thread.add_system(PROXY_PROMPT)
    msg_thread.add_user(f"<analysis>\n{text}</analysis>")
    res_text, *_ = common.SELECTED_MODEL.call(
        msg_thread.to_msg(), response_format="json_object"
    )

    msg_thread.add_model(res_text, [])  # no tools

    return res_text, msg_thread


def is_valid_response_proxy(data: Any) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Json is not a dict"

    if not data.get("terminate"):
        terminate = data.get("terminate")
        if terminate is None:
            return False, "'terminate' parameter is missing"

        if not isinstance(terminate, bool):
            return False, "'terminate' parameter must be a boolean (true/false)"

    else:
        if not data.get("collected_information"):
            summary = data.get("collected_information")
            if summary is None:
                return False, "'collected_information' parameter is missing"

            if not isinstance(summary, str):
                return False, "'collected_information' parameter must be a str"

        for api_call in data["API_calls"]:
            if not isinstance(api_call, str):
                return False, "Every API call must be a string"

            try:
                func_name, func_args = parse_function_invocation(api_call)
            except Exception:
                return False, "Every API call must be of form api_call(arg1, ..., argn)"
            function = getattr(RepoBrowseManager, func_name, None)
            if function is None:
                return False, f"the API call '{api_call}' calls a non-existent function"

            arg_spec = inspect.getfullargspec(function)
            arg_names = arg_spec.args[1:]  # first parameter is self

            if len(func_args) != len(arg_names):
                return False, f"the API call '{api_call}' has wrong number of arguments"

    return True, "OK"


def extract_json_from_response(res_text: str):
    """
    从文本响应中提取 JSON 代码块
    """
    json_extracted = None

    # Pattern 1: 识别 ```json 标记的代码块
    json_matches = re.findall(r"```json([\s\S]*?)```", res_text, re.IGNORECASE)
    if json_matches:
        json_extracted = json_matches[0].strip()

    # Pattern 2: 识别普通的 ``` 代码块
    if not json_extracted:
        json_code_blocks = re.findall(r"```([\s\S]*?)```", res_text, re.IGNORECASE)
        for content in json_code_blocks:
            clean_content = content.strip()
            # 尝试解析为 JSON，确保是 JSON 格式
            try:
                json.loads(clean_content)  # 测试是否有效 JSON
                json_extracted = clean_content
                break
            except json.JSONDecodeError:
                continue  # 跳过非 JSON 代码块

    return json_extracted if json_extracted else res_text  # 返回提取的 JSON 或原始文本


BROWSE_CONTENT_PROMPT = """
You are an autonomous file-browsing and analysis agent. Now the user gives you a file. Your overall mission is:
1. To review the given file content.
2. To extract any details necessary for setting up the project's environment and running its test suite.
3. To pay special attention to contents related to custom user queries.

Primary objectives:
- **Identify libraries, packages, and their exact versions.**
- List any environment variables or configuration files.
- Extract the exact commands or scripts used to run tests, including all relevant flags/options.
- **Pay special attention to commands for running individual or specific test files, not just commands for running all tests.**
- Note any prerequisites (e.g., required OS packages, language runtimes).

Formatting rules:
- Return your answer enclosed within `<analysis></analysis>` tags.
- Always wrap your structured key information in `[Key Information from <filename>] ... [/Key Information]` tags, making clear where the information was sourced (Do not use abstract path).
- Use bullet lists for clarity.
- Keep it concise and human-readable.
- Preserve original value formats (e.g., version strings, paths, flags).
- Keep the final answer concise. Do not include irrelevant information. If no relevant content is found, simply state "No relevant information found."

Example format:
<analysis>
[Key Information from README.md]
- setup command:
  - pip install -r requirements.txt
  - pip install -r requirements-dev.txt (**For development dependencies**)
  - pip install -r requirements-test.txt (**For test dependencies**)
- Libraries:
  - flask==2.0.3 (**Exact version**)
  - gunicorn==20.1.0 (**Exact version**)
  - pytest==7.1.2 (**Exact version**)
- Runtime Requirements:
  - Python >=3.8 (**Exact version**)
  - Node.js >=14.0 (**Exact version**)
  - Java >=8.0 (**Exact version**)
- Testing:
  - Test framework: pytest
  - **Test command (single test file): pytest tests/test_api.py**
- Key environment variables:
  - DEBUG=true
[/Key Information]
</analysis>
"""


def browse_file_run_with_retries(
    content: str, custom_query: str, retries: int = 3
) -> str:
    """Run file content analysis with retries and return the parsed <analysis> content."""
    parsed_result = ""
    for idx in range(1, retries + 1):
        logger.debug("Analyzing file content. Try {} of {}", idx, retries)

        res_text, _ = browse_file_run(content, custom_query)

        # Extract <analysis> content if valid
        parsed_result = parse_analysis_tags(res_text)
        if parsed_result:
            logger.success("Successfully extracted environment config")
            logger.info("*" * 6)
            logger.info(parsed_result)
            logger.info("*" * 6)
            return parsed_result
        else:
            content += "Please wrap result in clean xml identifier, do not use ```to wrap results. "
            logger.debug(res_text)
            logger.debug("Invalid response or missing <analysis> tags, retrying...")
    if parsed_result:
        return parsed_result
    else:
        return "Do not get the content of the file."


def browse_file_run(content: str, custom_query: str) -> tuple[str, MessageThread]:
    """Run the simplified content analysis agent."""
    msg_thread = MessageThread()
    msg_thread.add_system(BROWSE_CONTENT_PROMPT)
    msg_thread.add_user(f"File content:\n{content}\n")  # Truncate to prevent overflow
    msg_thread.add_user(f"Custom query from user:\n{custom_query}\n")
    res_text, *_ = common.SELECTED_MODEL.call(msg_thread.to_msg())
    msg_thread.add_model(res_text, [])
    return res_text, msg_thread


def parse_analysis_tags(data: str) -> str | None:
    """Extract and return the content within <analysis>...</analysis> tags."""
    pattern = r"<analysis>([\s\S]+?)</analysis>"
    match = re.search(pattern, data)
    if match:
        return match.group(1).strip()  # Return the content inside <analysis> tags
    return None


SYSTEM_PROMPT = """You are a context_retrieval_agent responsible for gathering **precise and necessary information** from the local repository to support environment setup and test execution. After gathering the information, you will **generate a concise report** summarizing the key findings related to the setup and test execution.

Sometimes, another agent (such as a test analysis agent) may explicitly request specific information to help fix issues like Dockerfile errors or evaluation script failures.

Your primary goal is to:

- **If a specific request is provided by a calling agent, focus your retrieval narrowly on that request, extracting only the explicitly required files or data.**
- **If no explicit request is given by another agent, or if the request is incomplete or unclear, perform a basic and limited exploration of the repository to collect general environment and test execution information. Avoid exhaustive or in-depth searches.**
- **Pay special attention to the following information when collecting and summarizing:**
  - **Exact versions** of dependencies, libraries, and programming languages (e.g., `flask==2.0.3`, `python3.9`, `node 18`)
  - **Commands** for setting up the environment and executing tests (e.g., `pip install -r requirements.txt`, `pytest tests/test_api.py`)
  - Any environment configuration details (e.g., `.env` files, specific OS package dependencies, etc.)
  - Specific test commands for individual or specific test files, not just generic test execution commands.

### Suggested Retrieval Areas

Only investigate the following areas **if explicitly requested** by the calling agent. Focus your retrieval on the minimal set of files or configurations needed to resolve the issue efficiently and accurately.

1. **Environment Setup Information**
   - **Exact dependencies and their versions**: This includes dependencies listed in files like `requirements.txt`, `package.json`, `pom.xml`, `build.gradle`, etc. Ensure that the exact version for each dependency is captured.
   - **Programming language versions**: Ensure to capture version information like Python (e.g., `python3.9`), Node.js (e.g., `node 18`), Java (e.g., `java 17`), and others as specified in relevant configuration files (`.nvmrc`, `.python-version`, etc.)
   - **Environment configuration files**: Collect details from `.env`, `.bashrc`, or `.zshrc` if applicable, focusing on version-dependent environment variables and paths.
   - **OS-specific requirements**: Note any OS-dependent configurations (e.g., specific Linux package dependencies in `apt` or `yum`).

2. **Test Execution Information**
   - **Precise test commands**: Focus on specific commands or instructions for running individual tests or specific test files, not just commands for running all tests. Look for test commands in documentation like `README.md`, `CONTRIBUTING.md`, `tests/README.md`, etc.
   - **CI/CD configurations**: Look into files like `.github/workflows/`, `.ci.yml`, `travis.yml`, or other pipeline configuration files that might include commands for running tests or specific test environments.
   - **Test execution in context**: Extract any specific instructions about running tests, such as flags for specific test cases, test suites, or environments. Also, pay attention to dependencies relevant to testing like test frameworks (e.g., `pytest`, `JUnit`, `Mocha`), env frameworks (e.g. poetry, uv), and their versions.

3. **Organize Results for other agents**
   - Present findings in a structured way so they can be used to generate the Dockerfile and evaluation script accurately. The **final report** should:
     - Highlight the **specific versions** of dependencies, libraries, and testing tools.
     - Include **commands** for setup and testing (e.g., `pip install`, `npm install`, `pytest`).
     - Note any environment variables or configuration details relevant to the environment setup and test execution.
     - Provide clear, concise, and actionable information, making it easier for other agents to proceed with resolving any setup or test execution issues.

### Important Notes:
- The repository has already been **cloned locally**; you are working within the local repository directory.  
- You are **not expected to search broadly**; retrieve only the files and information explicitly requested by the calling agent.  
- Avoid redundant or speculative searches—**be goal-driven and cost-efficient**.  
"""


USER_PROMPT = (
    "Your task is to gather sufficient context from the repository and external sources to understand how to set up the project's environment. To achieve this, you can use the following APIs to browse and extract relevant information:"
    "\n- browse_folder(path: str, depth: str): Browse and return the folder structure for a given path in the repository.  The depth is a string representing a number of folder levels to include in the output such as ``1''. "
    "\n- browse_file_for_environment_info(file_path: str, custom_query: str): Call an agent to browse a file such as README or CONTRIBUTING.md and extract environment setup and running tests information. Use the `custom_query` parameter to tell the agent any extra details it should pay special attention to (for example, 'what java version do we need?')."
    "\n- search_files_by_keyword(keyword: str): Search for files in the repository whose names contain the given keyword."
    "\n\nYou may invoke multiple APIs in one round as needed to gather the required information."
    "\n\nNow analyze the repository and use the necessary APIs to gather the information required to understand and set up the environment. Ensure each API call has concrete arguments as inputs."
)

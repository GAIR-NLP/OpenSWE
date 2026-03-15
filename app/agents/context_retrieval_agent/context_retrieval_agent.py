from app.data_structures import MessageThread, FunctionCallIntent
from app.agents.context_retrieval_agent import context_retrieval_utils
import inspect
import json
from app.agents.agent import Agent
from app.task import SweTask
import os
from loguru import logger
from os.path import join as pjoin
from pathlib import Path
from app.model import common
from app.log import (
    print_acr,
    print_banner,
    print_retrieval,
)
from app.utils import parse_function_invocation


class ContextRetrievalAgent(Agent):

    api_functions: list[str] = [
        "browse_folder",
        "search_files_by_keyword",
        "browse_file_for_environment_info",
    ]

    def __init__(
        self,
        task: SweTask,
        output_dir: str,
        repo_basic_info: str,
        max_context_retrieval_round: int = 10,
    ):
        super().__init__(agent_id="ContextRetrievalAgent")
        self.msg_thread = MessageThread()
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.run_count = 0
        self.repo_browse_manager = context_retrieval_utils.RepoBrowseManager(
            self.task.project_path
        )
        self.root_structure = self.browse_folder("/", 1)[0]
        self.root_structure_info = (
            f" Root directory structure of target repository: {self.root_structure}\n\n"
        )
        self.repo_basic_info = repo_basic_info
        self.reference_setup = None

        self.max_context_retrieval_round = max_context_retrieval_round
        self.init_msg_thread()

    def init_msg_thread(self) -> None:
        self.msg_thread = MessageThread()
        self.add_system_message(context_retrieval_utils.SYSTEM_PROMPT)
        self.add_user_message(self.repo_basic_info)
        self.add_user_message(self.root_structure_info)
        user_prompt = context_retrieval_utils.USER_PROMPT
        if "flash" in common.SELECTED_MODEL.name:
            user_prompt += "Now tell me your summary and APIs you plan to invoke:"
        self.add_user_message(user_prompt)

    def browse_folder(self, path: str, depth: int) -> str:
        """
        Browse and return the folder structure for a given path in the repository.

        Args:
            path: The folder path to browse, relative to the project root.
            depth: The number of folder levels to include in the output (depth>0).

        Returns:
            A string representation of the folder structure.
            Example output for path='src' and depth='2':
            src/
                main.py
                utils/
                    helper.py
                    constants.py
        """
        depth = int(depth)
        return self.repo_browse_manager.browse_folder(path, depth)[0]

    def search_files_by_keyword(self, keyword: str) -> str:
        """Search for files in the repository whose names contain the given keyword.

        Args:
            keyword: The keyword to search for in file names

        Returns:
            A formatted string showing the matching files (up to 10), or a message if too many files are found.
        """
        return self.repo_browse_manager.search_files_by_keyword(keyword)[0]

    def browse_file_for_environment_info(
        self, file_path: str, custom_query: str
    ) -> tuple[str, str, bool]:
        """
        Browse a file and extract environment setup information, with an optional custom query.

        Args:
            file_path: The path to the file to browse, relative to the project root.
            custom_query: A free‐form string describing what extra information the agent should look for
                          (e.g. 'pom.xml dependency versions', 'custom test profiles', etc.).

        Returns:
            extracted_info: Detailed info extracted from the file.
            summary:      A brief summary of the findings.
            success:      Whether extraction succeeded.
        """
        # Ensure file_path is correctly adjusted to be relative to project root

        try:
            if not file_path.startswith(self.task.project_path):
                file_path = pjoin(self.task.project_path, file_path)
            # Attempt to extract the environment setup information from the file

            extracted_info, summary, success = (
                self.repo_browse_manager.browse_file_for_environment_info(
                    file_path, custom_query
                )
            )
            return extracted_info, summary, success

        except Exception as e:
            # Log the error message for debugging purposes
            logger.error(f"Error while browsing file {file_path}: {e}")

            # Return an appropriate error message or an empty string
            return "", f"Error extracting env info: {e}", False

    def proxy_apis(self, text: str) -> tuple[str | None, str, list[MessageThread]]:
        """Proxy APIs to another agent."""
        tool_output, new_thread = context_retrieval_utils.proxy_apis_with_retries(
            text
        )  # FIXME: type of `text`
        if tool_output is None:
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
        else:
            summary = "The tool returned the selected search APIs in json format generated by another agent."
        return tool_output, summary, new_thread

    def run_task(self, print_callback=None, **_) -> tuple[str, str, bool]:
        self.run_count += 1
        context_retrieval_round = -1
        task_output = None
        summary = None
        success = None

        self.reset_tool_sequence()
        while True:
            context_retrieval_round += 1

            context_retrieval_output_dir = (
                self.get_latest_context_retrieval_output_dir()
            )
            os.makedirs(context_retrieval_output_dir, exist_ok=True)
            # f'{output_dir}/output_context_retrieval_{api_manager.context_retrieval_num}'
            conversation_file = pjoin(
                context_retrieval_output_dir,
                f"conversation_{context_retrieval_round}.json",
            )
            # save current state before starting a new round
            self.msg_thread.save_to_file(conversation_file)

            print_banner(
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} CONTEXT RETRIEVAL ROUND {context_retrieval_round}"
            )

            print_acr(
                # prompt,
                "context retrieval",
                f"context retrieval {context_retrieval_round}",
                print_callback=print_callback,
            )
            # get_action
            res_text, *_ = common.SELECTED_MODEL.call(self.msg_thread.to_msg())
            self.add_model_message(res_text, tools=[])
            print_retrieval(
                res_text,
                f"context retrieval {context_retrieval_round}",
                print_callback=print_callback,
            )

            # parse acrtion from response
            selected_apis, _, proxy_threads = self.proxy_apis(res_text)

            proxy_log = Path(
                context_retrieval_output_dir,
                f"agent_proxy_{context_retrieval_round}.json",
            )
            proxy_messages = [thread.to_msg() for thread in proxy_threads]
            proxy_log.write_text(json.dumps(proxy_messages, indent=4))

            if selected_apis is None:
                msg = "The repo browsing API calls seem invalid. Please check the arguments you give carefully and try again."
                self.add_user_message(msg)
                print_acr(
                    msg,
                    f"context retrieval {context_retrieval_round} ",
                    print_callback=print_callback,
                )
                continue

            selected_apis_json = json.loads(selected_apis)

            json_api_calls = selected_apis_json.get("API_calls", [])
            is_termination = selected_apis_json.get("terminate", None)
            summary_of_collected_information = selected_apis_json.get(
                "collected_information", None
            )
            if is_termination:
                msg_summary_of_collected_information = f"Collected information from context retrieval agent:\n{summary_of_collected_information}\n\n"

                # msg_summary_of_collected_information = self.root_structure_info+'\n\n' + msg_summary_of_collected_information
                task_output = msg_summary_of_collected_information
                summary = "Collect context information successfully."
                success = True

                break
            formatted = []
            if json_api_calls:
                formatted.append("API calls:")
                for call in json_api_calls:
                    formatted.extend([f"\n- `{call}`"])

            print_acr(
                "\n".join(formatted),
                "Agent-selected API calls",
                print_callback=print_callback,
            )

            # init observation
            # prepare response from tools
            collated_tool_response = ""

            for api_call in json_api_calls:
                try:
                    func_name, func_args = parse_function_invocation(api_call)
                    arg_spec = inspect.getfullargspec(
                        getattr(context_retrieval_utils.RepoBrowseManager, func_name)
                    )

                    arg_names = arg_spec.args[1:]  # first parameter is self

                    assert len(func_args) == len(
                        arg_names
                    ), f"Number of argument is wrong in API call: {api_call}"

                    kwargs = dict(zip(arg_names, func_args))
                    intent = FunctionCallIntent(func_name, kwargs, None)
                except Exception as call_api_e:
                    collated_tool_response += (
                        f"Exception when calling {api_call}: {call_api_e}\n\n"
                    )
                    continue
                # action -> obeservation
                tool_output, _, _ = self.dispatch_intent(intent)
                # merge observation
                collated_tool_response += f"Result of {api_call}:\n\n"
                collated_tool_response += f"{tool_output}\n\n"

            # observation -> thought
            self.add_user_message(collated_tool_response)
            print_acr(
                collated_tool_response,
                f"context retrieval {context_retrieval_round}",
                print_callback=print_callback,
            )
            # # thought
            # msg = "Let's analyze collected context first"
            # self.add_user_message(msg)
            # print_acr(
            #     msg, f"context retrieval {context_retrieval_round}", print_callback=print_callback
            # )
            # #thought
            # res_text, *_ = common.SELECTED_MODEL.call(self.msg_thread.to_msg())
            # self.add_model_message(res_text, tools=[])
            # print_retrieval(res_text, f"context retrieval {context_retrieval_round}", print_callback=print_callback)

            # thought -> action
            if context_retrieval_round < self.max_context_retrieval_round:
                msg = """\
Let's analyze the collected information and answer these questions:

1. If there is no specific requirement from the test_analysis_agent, or it has already been satisfied, then determine whether the current information about environment setup and test execution is sufficient. If not, identify what files, contexts, or details are missing, and propose the next appropriate actions based on the basic information from the repository. A good place to start would be inspecting the repository structure and reviewing key documents like `README.md`, `CONTRIBUTING.md`, and basic CI configurations. Avoid aggressive actions like performing deep repository browsing if not explicitly required.

2. If there are specific requirements from the test_analysis_agent, have these been fully satisfied by the collected information? If yes, you can stop here and provide a detailed summary of all collected information that meets those requirements. The summary must strictly preserve the original content and clearly indicate each section’s source using the format: [Section Title from filename]...[/Section Title from filename].

[Environment Setup from README.md]
To install test dependencies, run:
pip install -r requirements-test.txt
[/Environment Setup from README.md]

3. If the information is sufficient, stop and provide the detailed summary as described above.
"""

                # if isinstance(common.SELECTED_MODEL, ollama.OllamaModel):
                #     # llama models tend to always output search APIs and buggy locations.
                #     msg += "\n\nNOTE: If you have already identified the bug locations, do not make any search API calls."
                self.add_user_message(msg)
                print_acr(
                    msg,
                    f"context retrieval {context_retrieval_round}",
                    print_callback=print_callback,
                )
            else:
                task_output = ""
                summary = "Collect context information failure."
                success = False
                break
        self.dump_tool_sequence(self.get_latest_context_retrieval_output_dir())
        self.init_msg_thread()
        return task_output, summary, success

    def _read_file(self, path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception:
            return ""

    def get_latest_context_retrieval_output_dir(self) -> str:
        """
        Return the directory of the most recent Context retrieval outputs.
        """
        return os.path.join(
            self.output_dir, f"context_retrieval_agent_{self.run_count}"
        )

    def browse_readme(self) -> str:
        """
        This is only used in ablation study: w/o the context retrieval agent.
        """
        readme_list = ["README.md", "README.rst", "README.txt"]
        for readme_name in readme_list:
            file_path = pjoin(self.task.project_path, readme_name)
            try:
                readme_content = self.repo_browse_manager.browse_file(file_path)
                return f"The content of {readme_name} in the target repository:\n<README>\n{readme_content}\n</README>\n"
            except Exception:
                continue
        return ""

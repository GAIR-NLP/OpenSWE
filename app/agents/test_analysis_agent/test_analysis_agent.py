from pathlib import Path
import os
import re
from loguru import logger
from app.agents.agent import Agent
from app.data_structures import FunctionCallIntent, MessageThread
from app.task import SweTask
from app.agents.test_analysis_agent import test_analysis_utils
from app.agents.test_analysis_agent.docker_utils import (
    cleanup_container,
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    BuildImageError,
    build_container,
    EvaluationError,
    push_image,
)
import docker
import re
from app.log import log_exception, setup_logger, close_logger
from app.log import (
    print_acr,
    print_banner,
    print_retrieval,
)
import json
from os.path import join as pjoin
import traceback

MAX_LINE_NUM = 600
ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class BuildError(Exception):
    def __init__(self, reason: str, build_log: list[str]):
        self.msg = reason
        self.build_log = build_log


class TestAnalysisAgent(Agent):
    """
    Agent responsible for:
      1. Loading the latest test_output.txt
      2. Formatting it with line numbers and truncation
      3. Sending it to the test-log-analysis utility (agent_analyze_test_log)
    """

    api_functions = ["setup_docker_and_run_test"]

    def __init__(
        self,
        task: SweTask,
        output_dir: str,
        repo_basic_info: str,
        client: docker.DockerClient,
    ):
        super().__init__(agent_id=self.__class__.__name__)
        self.msg_thread = MessageThread()
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_count = 0
        self.run_test_num = 0
        self.setup_dockerfile_num = 0
        self.repo_basic_info = repo_basic_info
        self.task_id = task.task_id.lower()
        self.client = client
        self.test_analysis_dir = os.path.join(self.output_dir, "test_analysis_agent")
        # self.build_image_dir = os.path.join(self.output_dir, "build_image")
        # self.run_test_dir = os.path.join(self.output_dir, "run_test")
        self.eval_script_skeleton: str | None = None
        self.dockerfile: str | None = None
        self.eval_script: str | None = None
        self.timeout = 3600
        self.disable_context_retrieval = False
        self.disable_run_test = False
        # self.init_msg_thread()

    def init_msg_thread(self) -> None:
        """
        Reset the message thread and inject the base prompt before each run_task.
        """
        self.msg_thread = MessageThread()
        # Choose the appropriate system prompt
        # if getattr(agent_analyze_test_log, "SYSTEM_PROMPT_WIT_WEB_SEARCH", None) and getattr(self.task, 'enable_web_search', False):
        #     self.msg_thread.add_system(test_analysis_utils.SYSTEM_PROMPT_WIT_WEB_SEARCH)
        # else:
        if self.disable_context_retrieval:
            self.add_system_message(
                test_analysis_utils.SYSTEM_PROMPT_WITHOUT_CONTEXT_RETRIEVAL
            )
        elif self.disable_run_test:
            self.add_system_message(test_analysis_utils.SYSTEM_PROMPT_WITHOUT_RUN_TEST)
        else:
            self.add_system_message(test_analysis_utils.SYSTEM_PROMPT)
        # Inject repository basic information
        self.add_user_message(self.repo_basic_info)
        self.add_user_message(
            f"The current dockerfile used to setup environemnt:\n{self.dockerfile}"
        )
        self.add_user_message(
            f"The current eval script (omit test patch to decrease length) used to run tests:\n{self.eval_script_skeleton}"
        )

    def get_latest_test_analysis_output_dir(self):
        output_dir = f"{self.test_analysis_dir}_{self.analysis_count}"
        return output_dir

    def _truncate_log(self, content: str) -> str:
        n_lines = content.count("\n")
        if n_lines < MAX_LINE_NUM:
            return content
        lines = content.splitlines()

        head_size = MAX_LINE_NUM // 2
        tail_size = MAX_LINE_NUM - head_size

        head = lines[:head_size]
        tail = lines[-tail_size:]
        omission = f"[..., {len(lines) - head_size - tail_size} lines omitted ...]"

        return "\n".join(head + [omission] + tail)

    def _golden_to_name(self, golden: bool) -> str:
        return "testWithFix" if golden else "testOnly"

    def _read_latest_test_log(self, golden: bool) -> str:
        test_dir = self.get_latest_test_analysis_output_dir()
        name = self._golden_to_name(golden)
        path = os.path.join(test_dir, f"test_output_{name}.txt")
        try:
            logs = Path(path).read_text()
            return f"<{name}>\n{self._truncate_log(logs)}\n</{name}>"
        except FileNotFoundError:
            return ""

    def get_test_log_with_line_numbers(self) -> str:

        testOnly = self._read_latest_test_log(False)
        testWithPatch = self._read_latest_test_log(True)
        return f"Test log:\n{testOnly}\n{testWithPatch}\n\n"

    def run_task(
        self, print_callback=None, disable_context_retrieval=False, **_
    ) -> tuple[str, str, bool]:
        """
        2. Read and format the test log
        3. Add formatted log to the message thread
        4. Invoke agent_analyze_test_log.run_with_retries(...)
        5. Return (tool_output, summary, ok)
        """
        self.init_msg_thread()
        print_banner(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} Try to setup docker and run tests "
        )

        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir, exist_ok=True)
        intent = FunctionCallIntent("setup_docker_and_run_test", {}, None)
        if self.client is None:
            # Skip docker/test execution if client is not available (disable_all_docker)
            tool_output = "Docker operations disabled (no client provided)."
            success = False
            build_image_status = False
        else:
            tool_output, _, success = self.dispatch_intent(intent)
            build_image_status = "Image built successfully" in tool_output
        if self.client is None:
            print_acr(
                "Docker disabled; skipping build/run.",
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis",
                print_callback=print_callback,
            )
            error_in_building_dockerfile = f"Docker is disabled or unavailable. Generated Dockerfile/eval.sh only.\n{tool_output}\n\n"
            self.add_user_message(error_in_building_dockerfile)
        elif "Image built successfully!" not in tool_output:
            # fail to build image. go to dockefile refine.
            print_acr(
                "Build Image Failure!",
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis",
                print_callback=print_callback,
            )
            error_in_building_dockerfile = f"We can not run tests successfully, cause we encounter some errors when building dockerfile. As follows:\n{tool_output}\n\n"
            self.add_user_message(error_in_building_dockerfile)
        elif success:
            build_image_status = True
            print_acr(
                "Build Image Successfully!",
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis ",
                print_callback=print_callback,
            )
            test_log = self.get_test_log_with_line_numbers()
            self.add_user_message(test_log)
        else:
            # Docker built successfully but test validation failed
            # This is still valuable information that needs to be analyzed by LLM
            build_image_status = True
            print_acr(
                "Build Image Successfully but Test Validation Failed!",
                f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis ",
                print_callback=print_callback,
            )
            test_log = self.get_test_log_with_line_numbers()
            self.add_user_message(test_log)
            # Continue to LLM analysis instead of returning early

        # if we judge that we achieve the goal, terminate the process
        # if test log show that it fails, we go to plan for futrure directions
        # judge whether achieve the goal, if not planning for the work in the next stage.
        print_acr(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  Try to analyze the test log ",
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis round ",
            print_callback=print_callback,
        )

        analysis = test_analysis_utils.run_with_retries(
            self.msg_thread,
            disable_context_retrieval=disable_context_retrieval,
            print_callback=print_callback,
        )
        task_output = analysis
        analysis_file = Path(
            f"{self.get_latest_test_analysis_output_dir()}/analysis.json"
        )

        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception as e:
                to_save = {}
        else:
            # analysis 既不是 dict 也不是 str，按需处理
            to_save = {}

        to_save["build_image_status"] = build_image_status
        # If the environment is too hard for agent to setup, the agent will mark is_finish = True and exit with success = False
        # In that case, we consider the task as unsolvable and discard the current sample.
        to_save["solvable"] = True
        if success == False and to_save.get("is_finish") == True:
            # agent fail to judge
            to_save["is_finish"] = True
            to_save["solvable"] = False
            task_output = json.dumps(to_save)
            summary = "Agent's is_finish incorrect."
            success = False

        elif task_output is None:
            to_save["is_finish"] = False
            task_output = json.dumps(to_save)
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
            success = False

        else:
            summary = "The tool returned the selected search APIs in json format generated by another agent."
            success = True

        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)

        # need to save analysis into json file
        conversation_file = pjoin(test_log_output_dir, f"conversation.json")
        self.msg_thread.save_to_file(conversation_file)

        return task_output, summary, success

    def run_task_without_run_test(self, print_callback=None) -> tuple[str, str, bool]:
        """
        This function is just for ablation study
        """
        self.init_msg_thread()
        print_banner(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} Try to setup docker and run tests "
        )

        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir, exist_ok=True)

        # if we judge that we achieve the goal, terminate the process
        # if test log show that it fails, we go to plan for futrure directions
        # judge whether achieve the goal, if not planning for the work in the next stage.
        print_acr(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  Try to analyze the test log ",
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num}  test analysis round ",
            print_callback=print_callback,
        )

        success = False
        analysis = test_analysis_utils.run_with_retries(
            self.msg_thread, disable_run_test=True, print_callback=print_callback
        )
        task_output = analysis
        analysis_file = Path(
            f"{self.get_latest_test_analysis_output_dir()}/analysis.json"
        )

        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception as e:
                to_save = {}
        else:
            # analysis 既不是 dict 也不是 str，按需处理
            to_save = {}

        # to_save['build_image_status'] = build_image_status

        if task_output is None:
            task_output = "<empty>"
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
            success = False

        else:
            summary = "The tool returned the selected search APIs in json format generated by another agent."
            success = True

        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)

        # need to save analysis into json file
        conversation_file = pjoin(test_log_output_dir, f"conversation.json")
        self.msg_thread.save_to_file(conversation_file)

        return task_output, summary, success

    def build_docker_image(
        self,
        dockerfile,
        cur_build_image_dir,
        task_id,
        image_name,
        build_image_logger,
        client: docker.DockerClient,
    ):
        """Build Docker image with detailed logging and error handling."""

        build_image_logger.info(
            f"Building image {task_id}\n" f"Using dockerfile:\n{dockerfile}\n"
        )

        dockerfile_path = f"{self.output_dir}/Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        command_output = []
        capturing = False
        response = client.api.build(
            path=self.output_dir,
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform="linux/x86_64",
            # nocache=True,
        )

        buffer = ""

        for chunk in response:
            if "stream" in chunk:

                buffer += (
                    ansi_escape.sub("", chunk["stream"])
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    if line.startswith("Step "):
                        last_command = line
                        command_output = [line]
                        capturing = True
                    elif capturing:
                        command_output.append(line)

                    build_image_logger.info(line)

            elif "errorDetail" in chunk and capturing:

                if buffer.strip():
                    command_output.append(buffer.strip())
                    build_image_logger.info(buffer.strip())
                    buffer = ""

                error_msg = ansi_escape.sub("", chunk["errorDetail"]["message"])
                build_image_logger.error(f"Error: {error_msg}")
                command_output.append(f"Error: {error_msg}")

                raise BuildError(error_msg, build_log=command_output)

        if buffer.strip():
            build_image_logger.info(buffer.strip())

        build_image_logger.info("Image built successfully!")

    def image_tag(self):
        return f"openswe--{self.task_id}"

    def setup_docker_and_run_test(self) -> tuple[str, str, bool]:
        # building docker image first

        dockerfile = self.dockerfile
        assert dockerfile is not None

        eval_script = self.eval_script
        tool_output = ""
        summary = ""
        success = False
        self.setup_dockerfile_num += 1
        cur_build_image_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_build_image_dir, exist_ok=True)
        build_image_logger = setup_logger(
            self.task_id, Path(f"{cur_build_image_dir}/build_image.log")
        )
        # image_name = f"{self.task_id}:latest_{self.setup_dockerfile_num}"
        image_name = self.image_tag()

        dockerfile_path = f"{cur_build_image_dir}/Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        try:
            self.build_docker_image(
                dockerfile,
                cur_build_image_dir,
                self.task_id,
                image_name,
                build_image_logger,
                self.client,
            )
            tool_output += "Image built successfully!\n"
            summary += f"Docker image {image_name} built successfully.\n"
        except BuildError as e:

            build_log = list(e.build_log)
            if len(build_log) > MAX_LINE_NUM:
                half = MAX_LINE_NUM // 2
                skipped = len(build_log) - MAX_LINE_NUM
                build_log = (
                    build_log[:half]
                    + [f"...skipped {skipped} lines..."]
                    + build_log[-half:]
                )
            tool_output += "\n".join(build_log)
            build_image_logger.error(e)
            summary += f"Failed to build Docker image."
            success = False
            return tool_output, summary, success
        except Exception as e:

            build_image_logger.error(f"Unexpected error: {str(e)}")
            tool_output += f"{str(e)}\n"
            summary += f"Unexpected error when building images."
            success = False
            return tool_output, summary, success
        finally:
            close_logger(build_image_logger)

        success = True
        exit_codes = []
        for golden in (False, True):
            test_output, test_summary, test_success, outputs = self.run_test(golden)
            tool_output += test_output
            summary += test_summary
            success = success and test_success
            code = self._extract_exit_code_from_output(outputs)
            exit_codes.append(code)

        dual_valid, dual_msg = self._validate_swebench_dual_test(
            exit_codes[0], exit_codes[1], build_image_logger
        )
        self.add_user_message(
            f"We have built docker image, run eval script, and check exit code\n{dual_msg}"
        )

        success = success and dual_valid
        if success:
            # Push container to remote repository if DOCKER_REPOSITORY is set
            docker_repository = os.environ.get("DOCKER_REPOSITORY", "")
            print(
                "Test succeed, try pushing",
                f"repo={docker_repository}, success={success}",
            )
            if docker_repository and success:
                try:
                    # We use tag based image naming
                    remote_tag = f"{docker_repository}:{image_name}"
                    build_image_logger.info(
                        f"Pushing container {image_name} to {remote_tag}..."
                    )
                    push_image(self.client, image_name, remote_tag, build_image_logger)
                    tool_output += f"Container pushed to {remote_tag}.\n"
                    summary += "Container pushed to remote.\n"
                except Exception as e:
                    build_image_logger.error(f"Failed to push container: {e}")
                    tool_output += f"Warning: Failed to push container to remote: {e}\n"
                    # Don't fail the entire test run if push fails

        return tool_output, summary, success

    def _extract_exit_code_from_output(self, output: str) -> int:
        """
        从测试输出中提取OPENSWE_EXIT_CODE

        Args:
            output: 测试输出

        Returns:
            exit_code: 退出码，如果无法提取则返回-1
        """
        match = re.search(r"OPENSWE_EXIT_CODE=(\d+)", output)
        if match:
            return int(match.group(1))
        return -1

    def _validate_swebench_dual_test(
        self, testonly_code: int, testwithfix_code: int, run_test_logger
    ) -> tuple[bool, str]:
        """
        Validate SWEBench dual test conditions

        Args:
            exit_code_without_patch: Exit code without patch applied
            exit_code_with_patch: Exit code with patch applied
            run_test_logger: Logger instance

        Returns:
            (success, message): Whether validation passed and detailed message
        """
        code_msg = f"Exit code with/wo goldpatch: {testonly_code}/{testwithfix_code}"
        run_test_logger.info(code_msg)

        # Check if exit code extraction failed
        if testonly_code == -1:
            msg = "TestOnly failed: Unable to extract OPENSWE_EXIT_CODE from output"
            return False, msg

        if testwithfix_code == -1:
            msg = "TestWithFix failed: Unable to extract OPENSWE_EXIT_CODE from output"
            return False, msg

        # Validate dual conditions
        without_patch_failed = testonly_code != 0
        with_patch_succeeded = testwithfix_code == 0

        if without_patch_failed and with_patch_succeeded:
            return True, "Exitcode validation PASSED!"
        else:
            reasons = []
            reasons.append(code_msg)
            if not without_patch_failed:
                reasons.append(f"TestOnly eval should fail but passed")
            if not with_patch_succeeded:
                reasons.append(f"TestWithFix eval should pass but failed")
            msg = f"SWEBench dual validation FAILED: {'; '.join(reasons)}"
            return False, msg

    def run_test(self, goldpatch: bool = True) -> tuple[str, str, bool, str]:
        tool_output = ""
        summary = ""
        success = False
        assert self.eval_script_skeleton is not None

        patch = (
            self.task.patch + "\n" + self.task.test_patch
            if goldpatch
            else self.task.test_patch
        )
        testname = self._golden_to_name(goldpatch)
        eval_script = test_analysis_utils.replace_heredoc_content(
            self.eval_script_skeleton, patch
        )
        self.run_test_num += 1
        self.reset_tool_sequence()
        cur_test_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_test_dir, exist_ok=True)
        run_test_logger = setup_logger(
            self.task_id, Path(f"{cur_test_dir}/run_test.log"), mode="a"
        )
        # test_image_name = f"{self.task_id}:latest_{self.setup_dockerfile_num}"
        test_image_name = self.image_tag()
        # test_container_name =  f"{self.task_id}:test_{self.run_test_num}"
        test_container_name = f"{self.task_id}-test{self.run_test_num}-{testname}"
        instance_id = self.task_id
        container = None
        test_output_path = f"{cur_test_dir}/test_output_{testname}.txt"
        test_output = ""
        try:
            container = build_container(
                self.client,
                test_image_name,
                test_container_name,
                instance_id,
                run_test_logger,
            )

            container.start()
            run_test_logger.info(f"Container for {instance_id} started: {container.id}")
            tool_output += f"Container {container.id} started.\n"
            summary += "Container started.\n"

            # Copy model prediction as patch file to container
            # patch_file = Path(f"{cur_test_dir}/patch.diff")
            # patch_file.write_bytes((patch or "").encode('utf-8'))
            # run_test_logger.info(
            #     f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
            # )
            # copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

            # # Attempt to apply patch to container
            # val = container.exec_run(
            #     "git apply -p1 -v /tmp/patch.diff",
            #     workdir="/testbed",
            #     user="root",
            # )
            # exit_code = val.exit_code
            # output = val.output.decode("utf-8", errors="replace")

            # if exit_code != 0:
            #     run_test_logger.info("Failed to apply patch to container, trying again...")
            #     run_test_logger.error(f"git apply returned exit_code={exit_code}. Output:\n{output}")
            #     # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
            #     val = container.exec_run(
            #         "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
            #         workdir="/testbed",
            #         user="root",
            #     )
            #     if val.exit_code != 0:
            #         run_test_logger.info(f"Apply patch fail:\n{val.output.decode('utf-8')}")
            #         raise EvaluationError(
            #             instance_id,
            #             f"Apply patch fail:\n{val.output.decode('utf-8')}. Check if you apply patch in incorrect directories.",
            #             run_test_logger,
            #         )
            #     else:
            #         run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            # else:
            #     run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            # tool_output += "Patch applied successfully.\n"
            # summary += "Patch applied.\n"
            #         # Get git diff before running eval script
            # git_diff_output_before = (
            #     container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            # )
            # run_test_logger.info(f"Git diff before:\n{git_diff_output_before}")

            eval_file = Path(f"{self.get_latest_test_analysis_output_dir()}/eval.sh")
            eval_file.write_bytes(eval_script.encode("utf-8"))
            run_test_logger.info(
                f"Eval script for {instance_id} written to {eval_file}, now applying to container..."
            )
            copy_to_container(container, eval_file, Path("/eval.sh"))

            # Run eval script, write output to logs
            result = exec_run_with_timeout(
                container, "/bin/bash /eval.sh", timeout=self.timeout
            )
            test_output = result.decode("utf-8")

            with open(test_output_path, "w") as f:
                f.write(test_output)
            run_test_logger.info(
                f"Test output for {instance_id} ({testname}) written to {test_output_path}"
            )

            # Get git diff after running eval script
            # git_diff_output_after = (
            #     container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            # )

            # # Check if git diff changed after running eval script
            # run_test_logger.info(f"Git diff after:\n{git_diff_output_after}")
            # if git_diff_output_after != git_diff_output_before:
            #     run_test_logger.info(f"Git diff changed after running eval script")
            #     tool_output += "Note: Git diff changed after test execution.\n"
            #     summary += "Git diff changed.\n"

        except EvaluationError as e:
            error_msg = (
                f"EvaluationError {instance_id}: {e}\n"
                f"{traceback.format_exc()}\n"
                f"Check ({getattr(run_test_logger,"log_file")}) for more information."
            )
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Evaluation error occurred.\n"
            success = False

        except Exception as e:
            error_msg = (
                f"Error in evaluating model for {instance_id}: {e}\n"
                f"{traceback.format_exc()}\n"
                f"Check ({getattr(run_test_logger,"log_file")}) for more information."
            )
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Unexpected error occurred.\n"
            success = False
        else:
            if not os.path.exists(test_output_path):
                tool_output += "Do not generate test_output.txt. Please check the correctness of dockerfile and eval script.\n"
                summary += "Fail to obtain test results."
                success = False
            else:
                tool_output += f"Find test_output.txt! Waiting for analysis. "
                summary += "Obtain test results successfully."
                success = True

        finally:
            # Remove instance container + image, close logger
            cleanup_container(self.client, container, run_test_logger)

            close_logger(run_test_logger)
        self.dump_tool_sequence(self.get_latest_test_analysis_output_dir())
        return tool_output, summary, success, test_output

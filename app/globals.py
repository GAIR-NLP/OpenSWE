"""
Values of global configuration variables.
"""

# Overall output directory for results
output_dir: str = ""
setup_dir: str = ""
results_path: str = ""

# upper bound of the number of conversation rounds for the agent
conv_round_limit: int = 15

context_retrieval_round_limit: int = 15

organize_output_only: bool = False

# whether to perform sbfl
enable_sbfl: bool = False

# whether to perform layered search
enable_layered: bool = True

# whether to perform our own validation
enable_validation: bool = False

# whether to do angelic debugging
enable_angelic: bool = False

# whether to do perfect angelic debugging
enable_perfect_angelic: bool = False


# A special mode to only save SBFL result and exit
only_save_sbfl_result: bool = False

# timeout for test cmd execution, currently set to 5 min
test_exec_timeout: int = 300


# Used with disable_patch_generation - constrains or extends the amount of context retrieval rounds
context_generation_limit: int = -1

get_version: bool = False

enable_web_search: bool = False

agent_mode: str = "multi_agent"

disable_memory_pool: bool = False

disable_context_retrieval: bool = False

disable_run_test: bool = False

disable_download_test_resources: bool = False

using_ubuntu_only: bool = False

# Skip test_analysis_agent; useful for setup-only pipeline
skip_test_analysis: bool = False

# Skip all docker operations (build/run) but still generate Dockerfile/eval.sh
disable_all_docker: bool = False

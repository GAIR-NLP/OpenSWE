import ast
import contextlib
import glob
import os
import subprocess
from os.path import dirname as pdirname
from os.path import join as pjoin
from pathlib import Path
from subprocess import CalledProcessError
import shutil
from app.log import log_and_print


@contextlib.contextmanager
def cd(newdir):
    """
    Context manager for changing the current working directory
    :param newdir: path to the new directory
    :return: None
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run_command(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run a command in the shell.
    Args:
        - cmd: command to run
    """
    try:
        cp = subprocess.run(cmd, check=True, **kwargs)
    except subprocess.CalledProcessError as e:
        log_and_print(f"Error running command: {cmd}, {e}")
        raise e
    return cp


def is_git_repo() -> bool:
    """
    Check if the current directory is a git repo.
    """
    git_dir = ".git"
    return os.path.isdir(git_dir)


def initialize_git_repo_and_commit(logger=None):
    """
    Initialize the current directory as a git repository and make an initial commit.
    """
    init_cmd = ["git", "init"]
    add_all_cmd = ["git", "add", "."]
    commit_cmd = ["git", "commit", "-m", "Temp commit made by ACR."]
    run_command(init_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(add_all_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(commit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)



def get_current_commit_hash() -> str:
    command = ["git", "rev-parse", "HEAD"]
    cp = subprocess.run(command, text=True, capture_output=True)
    try:
        cp.check_returncode()
        return cp.stdout.strip()
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to get SHA-1 of HEAD: {cp.stderr}") from e


def repo_commit_current_changes():
    """
    Commit the current active changes so that it's safer to do git reset later on.
    Use case: for storing the changes made in pre_install and test_patch in a commit.
    """
    add_all_cmd = ["git", "add", "."]
    commit_cmd = ["git", "commit", "-m", "Temporary commit for storing changes"]
    run_command(add_all_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(commit_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clone_repo(clone_link: str, cloned_dir: str, bare: bool = False):
    """
    Clone a repo to dest_dir (without trailing `/`).

    Returns:
        - path to the newly cloned directory.
    """
    dest_dir = os.path.dirname(cloned_dir)  # 获取目录路径
    cloned_name = os.path.basename(cloned_dir)
    clone_cmd = ["git", "clone", clone_link, cloned_name]
    if bare:
        if os.path.isdir(cloned_dir): # already cloned
            return
        clone_cmd.append("--bare")
    else:
        git_dir = pjoin(cloned_dir, ".git")
        if os.path.isdir(git_dir):
            return
        
    create_dir_if_not_exists(dest_dir)
    with cd(dest_dir):
        try:
            run_command(clone_cmd)
        except Exception:
            # skip if already cloned by another process
            git_dir = pjoin(cloned_dir, ".git")
            if not os.path.isdir(git_dir):
                raise

    # cloned_dir = pjoin(dest_dir, cloned_name)
    # return cloned_dir


def clone_repo_and_checkout(
    clone_link: str, commit_hash: str, cloned_dir: str,
    # dest_dir: str, cloned_name: str
):
    """
    Clone a repo to dest_dir, and checkout to commit `commit_hash`. This will
    remove original directory.
    """
    # cloned_dir = 
    if os.path.isdir(cloned_dir):
        shutil.rmtree(cloned_dir)
    clone_repo(clone_link, cloned_dir)
    if commit_hash != "":
        checkout_cmd = ["git", "reset", "--hard", commit_hash]
        with cd(cloned_dir):
            run_command(checkout_cmd)
    # return cloned_dir

def get_version_by_git(cloned_dir:str)-> str:
    command = ["git"," describe","--tags"]
    info = None
    with cd(cloned_dir):
        info = run_command(command)
    return info.stdout.strip() # FIXME


def repo_clean_changes() -> None:
    """
    Reset repo to HEAD. Basically clean active changes and untracked files on top of HEAD.

    A lightweight version of `repo_reset_and_clean_checkout`. This is also used in different scenarios.
    """
    reset_cmd = ["git", "reset", "--hard"]
    clean_cmd = ["git", "clean", "-fd"]
    run_command(reset_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def repo_reset_and_clean_checkout(commit_hash: str) -> None:
    """
    Run commands to reset repo to the original commit state.
    Cleans both the uncommited changes and the untracked files, and submodule changes.
    Assumption: The current directory is the git repository.
    """
    # NOTE: do these before `git reset`. This is because some of the removed files below
    # may actually be in version control. So even if we deleted such files here, they
    # will be brought back by `git reset`.
    # Clean files that might be in .gitignore, but could have been created by previous runs
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    if os.path.exists("tests/.coveragerc"):
        os.remove("tests/.coveragerc")
    other_cov_files = glob.glob(".coverage.TSS.*", recursive=True)
    for f in other_cov_files:
        os.remove(f)

    reset_cmd = ["git", "reset", "--hard", commit_hash]
    clean_cmd = ["git", "clean", "-fd"]
    checkout_cmd = ["git", "checkout", commit_hash]
    run_command(reset_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_command(clean_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # need to checkout before submodule init. Otherwise submodule may init to another version
    run_command(checkout_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # this is a fail-safe combo to reset any changes to the submodule: first unbind all submodules
    # and then make a fresh checkout of them.
    # Reference: https://stackoverflow.com/questions/10906554/how-do-i-revert-my-changes-to-a-git-submodule
    submodule_unbind_cmd = ["git", "submodule", "deinit", "-f", "."]
    submodule_init_cmd = ["git", "submodule", "update", "--init"]
    run_command(
        submodule_unbind_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    run_command(
        submodule_init_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def run_script_in_conda(
    args: list[str], env_name: str, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a python command in a given conda environment.
    """
    cmd = ["conda", "run", "-n", env_name, "python", *args]
    return subprocess.run(cmd, **kwargs)


def run_string_cmd_in_conda(
    command: str, env_name: str, **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a complete command in a given conda environment, where the command is a string.

    This is useful when the command to be run contains &&, etc.

    NOTE: use `conda activate` instead of `conda run` in this verison, so that we can
          run commands that contain `&&`, etc.
    """
    conda_bin_path = os.getenv("CONDA_EXE")  # for calling conda
    if conda_bin_path is None:
        raise RuntimeError("Env variable CONDA_EXE is not set")
    conda_root_dir = pdirname(pdirname(conda_bin_path))
    conda_script_path = pjoin(conda_root_dir, "etc", "profile.d", "conda.sh")
    conda_cmd = f"source {conda_script_path} ; conda activate {env_name} ; {command} ; conda deactivate"
    log_and_print(f"Running command: {conda_cmd}")
    return subprocess.run(conda_cmd, shell=True, **kwargs)


def create_dir_if_not_exists(dir_path: str):
    """
    Create a directory if it does not exist.
    Args:
        dir_path (str): Path to the directory.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def to_relative_path(file_path: str, project_root: str) -> str:
    """Convert an absolute path to a path relative to the project root.

    Args:
        - file_path (str): The absolute path.
        - project_root (str): Absolute path of the project root dir.

    Returns:
        The relative path.
    """
    if Path(file_path).is_absolute():
        return str(Path(file_path).relative_to(project_root))
    else:
        return file_path


def to_absolute_path(file_path: str, project_root: str) -> str:
    """Convert a relative path to an absolute path.

    Args:
        - file_path (str): The relative path.
        - project_root (str): Absolute path of a root dir.
    """
    return pjoin(project_root, file_path)


def find_file(directory, filename) -> str | None:
    """
    Find a file in a directory. filename can be short name, relative path to the
    directory, or an incomplete relative path to the directory.
    Returns:
        - the relative path to the file if found; None otherwise.
    """

    # Helper method one
    def find_file_exact_relative(directory, filename) -> str | None:
        if os.path.isfile(os.path.join(directory, filename)):
            return filename
        return None

    # Helper method two
    def find_file_shortname(directory, filename) -> str | None:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == filename:
                    return os.path.relpath(os.path.join(root, file), directory)
        return None

    # if the filename is exactly the relative path.
    found = find_file_exact_relative(directory, filename)
    if found is not None:
        return found

    # if the filename is a short name without any directory
    found = find_file_shortname(directory, filename)
    if found is not None:
        return found

    # if the filename has some directory, but is not a relative path to
    # the directory
    parts = filename.split(os.path.sep)
    shortname = parts[-1]
    found = find_file_shortname(directory, shortname)
    if found is None:
        # really cannot find this file
        return None
    # can find this shortname, but we also need to check whether the intermediate
    # directories match
    if filename in found:
        return found
    else:
        return None


def parse_function_invocation(
    invocation_str: str,
) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(invocation_str)
        expr = tree.body[0]
        assert isinstance(expr, ast.Expr)
        call = expr.value
        assert isinstance(call, ast.Call)
        func = call.func
        assert isinstance(func, ast.Name)
        function_name = func.id
        raw_arguments = [ast.unparse(arg) for arg in call.args]
        # clean up spaces or quotes, just in case
        arguments = [arg.strip().strip("'").strip('"') for arg in raw_arguments]

        try:
            new_arguments = [ast.literal_eval(x) for x in raw_arguments]
            if new_arguments != arguments:
                log_and_print(
                    f"Refactored invocation argument parsing gives different result on "
                    f"{invocation_str!r}: old result is {arguments!r}, new result "
                    f" is {new_arguments!r}"
                )
        except Exception as e:
            log_and_print(
                f"Refactored invocation argument parsing failed on {invocation_str!r}: {e!s}"
            )
    except Exception as e:
        raise ValueError(f"Invalid function invocation: {invocation_str}") from e

    return function_name, arguments

def extract_test_patch(patch: str) -> tuple[str, str]:
    """extract golden and test patch from all patch, using filename pattern"""
    def is_test_file(file_path: str) -> bool:
        """Determine if a file path represents a test file."""
        test_keywords = ["test", "e2e", "spec"]
        path_lower = file_path.lower()
        return any(keyword in path_lower for keyword in test_keywords)
    
    if not patch:
        return "", ""
    
    lines = patch.split('\n')
    test_patches = []
    golden_patches = []
    current_patch = []

    is_in_test_file = False
    is_in_file = False
    
    for line in lines:
        if line.startswith('diff --git'):
            is_in_file = True
            if current_patch:
                content = '\n'.join(current_patch)
                if is_in_test_file and current_patch:
                    test_patches.append(content)
                else:
                    golden_patches.append(content)
            current_patch = [line]
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2][2:]
                is_in_test_file = is_test_file(file_path)
            else:
                is_in_test_file = False
        else:
            if is_in_file:
                current_patch.append(line)
    
    if is_in_test_file and current_patch:
        test_patches.append('\n'.join(current_patch))
    
    return '\n'.join(golden_patches), '\n'.join(test_patches)


def git_diff_commits(repo_path: str, base_commit: str, target_commit: str) -> str:
    """
    Generate a git diff between two commits.
    
    Args:
        repo_path: Path to the git repository
        base_commit: The base commit hash
        target_commit: The target commit hash
    
    Returns:
        The diff as a string, or empty string if commits are the same or on error
    """
    if not base_commit or not target_commit:
        return ""
    
    if base_commit == target_commit:
        return ""
    
    try:
        with cd(repo_path):
            diff_cmd = ["git", "diff", base_commit, target_commit]
            result = subprocess.run(
                diff_cmd,
                capture_output=True,
                text=True,
                check=True
            )
        return result.stdout
    except subprocess.CalledProcessError as e:
        log_and_print(f"Error generating diff between {base_commit} and {target_commit}: {e}")
        return ""
    except Exception as e:
        log_and_print(f"Unexpected error in git_diff_commits: {e}")
        return ""

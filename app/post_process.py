"""
Post-process the output of the inference workflow.
"""

import json
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from glob import glob
from os.path import join as pjoin
from shutil import move

from app import utils as apputils
from app.model import common




# track status of patch extraction
class ExtractStatus(str, Enum):
    # APPLICABLE_PATCH = "APPLICABLE_PATCH"
    # MATCHED_BUT_EMPTY_ORIGIN = "MATCHED_BUT_EMPTY_ORIGIN"
    # MATCHED_BUT_EMPTY_DIFF = "MATCHED_BUT_EMPTY_DIFF"
    # RAW_PATCH_BUT_UNMATCHED = "RAW_PATCH_BUT_UNMATCHED"
    # RAW_PATCH_BUT_UNPARSED = "RAW_PATCH_BUT_UNPARSED"
    # NO_PATCH = "NO_PATCH"
    IS_VALID_JSON = "IS_VALID_JSON"
    NOT_VALID_JSON = "NOT_VALID_JSON"
    NO_SETUP = "NO_SETUP"
    APPLICABLE_SETUP = "APPLICABLE_SETUP"

    def __lt__(self, other):
        # order from min to max
        order = [
            self.NO_SETUP,
            # self.RAW_PATCH_BUT_UNPARSED,
            # self.RAW_PATCH_BUT_UNMATCHED,
            # self.MATCHED_BUT_EMPTY_DIFF,
            # self.MATCHED_BUT_EMPTY_ORIGIN,
            self.APPLICABLE_SETUP,
        ]
        self_index = order.index(self)
        other_index = order.index(other)
        return self_index < other_index

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(self.value)

    def to_dir_name(self, expr_dir: str):
        return pjoin(expr_dir, self.value.lower())

    @staticmethod
    def max(statuses):
        return sorted(statuses)[-1]


def record_extract_status(individual_expr_dir: str, extract_status: ExtractStatus):
    """
    Write extract status to file, so that we can read it again when
    classifying patches
    """
    # there is 1-to-1 correspondence between agent_patch_raw and extract_status
    # FIXME: it might be better to record these status in memory so they can be easily managed.
    record_file = pjoin(individual_expr_dir, "extract_status.json")
    if not os.path.isfile(record_file):
        # record for the first time
        with open(record_file, "w") as f:
            json.dump({"extract_status": [extract_status]}, f, indent=4)
    else:
        with open(record_file) as f:
            record = json.load(f)
        record["extract_status"].append(extract_status)
        with open(record_file, "w") as f:
            json.dump(record, f, indent=4)


def read_extract_status(individual_expr_dir: str) -> tuple[ExtractStatus, int]:
    """
    Read extract status from file. If there are multiple status recorded, read the best one.
    Returns:
        - The best extract status
        - The index of the best status in the list of all statuses. (0-based)
    """
    # we should read from the all the record
    record_file = pjoin(individual_expr_dir, "Dockerfile")
    if not os.path.isfile(record_file):
        # if no status file is written, means that we did not even
        # reach the state of extracting patches
        return ExtractStatus.NO_SETUP, -1
    else:
        return ExtractStatus.APPLICABLE_SETUP, 1
    # with open(record_file) as f:
    #     record = json.load(f)
    # # convert string to enum type
    # all_status = [ExtractStatus(s) for s in record["extract_status"]]

    # best_status = ExtractStatus.max(all_status)
    # best_idx = all_status.index(best_status)
    # return best_status, best_idx






def organize_experiment_results(expr_dir: str):
    """
    Assuming patches have already been extracted, organize the experiment result
    directories into a few categories and move them there.
    """
    # (1) find all the task experiment directories
    task_exp_names = [
        x
        for x in os.listdir(expr_dir)
        if os.path.isdir(pjoin(expr_dir, x))
        and "__" in x  # for filtering out other dirs like "applicable_patch"
    ]
    task_exp_dirs = [pjoin(expr_dir, x) for x in task_exp_names]

    # start organizing
    for extract_status in ExtractStatus:
        os.makedirs(extract_status.to_dir_name(expr_dir), exist_ok=True)

    for task_dir in task_exp_dirs:
        extract_status, _ = read_extract_status(task_dir)
        corresponding_dir = extract_status.to_dir_name(expr_dir)
        shutil.move(task_dir, corresponding_dir)



def extract_swe_bench_input(dir: str):
    """
    After diff format patch files have been extracted, this function collects
    them and writes a single file that can be used by swe-bench.

    Returns:
        - path to swe-bench input file.
    """
    # only look into applicable_patch dir, since we have already done
    # the categorization
    applicable_res_dir = pjoin(dir, "applicable_setup")
    # figure out what tasks have applicable patch
    task_dirs = [
        x
        for x in os.listdir(applicable_res_dir)
        if os.path.isdir(pjoin(applicable_res_dir, x))
    ]
    task_dirs = [pjoin(applicable_res_dir, x) for x in task_dirs]
    # patch_files = [pjoin(x, "agent_patch_raw") for x in task_dirs]
    # patch_files = [os.path.abspath(x) for x in patch_files]

    # Diff files have the name extracted_patch_{1,2,3...}.diff
    # We take the one with the largest index. This is because
    # (1) if there is no validation, then there is at most one such file,
    #     so just take it
    # (2) if there is validation, only the one with the largest index may be correct
    docker_files = []
    for x in task_dirs:
        extracted_dockerfile = glob(pjoin(x, "Dockerfile"))
        docker_files.append(extracted_dockerfile[0])

    docker_files = [os.path.abspath(x) for x in docker_files]

    # patch_files = [x for x in patch_files if os.path.isfile(x)]
    docker_files = [x for x in docker_files if os.path.isfile(x)]

    all_results = []
    final_results = []
    for docker_file in docker_files:
        # task_dir = os.path.dirname(os.path.dirname(docker_file))
        task_dir = os.path.dirname(docker_file)
        meta_file = pjoin(task_dir, "meta.json")
        with open(meta_file) as f:
            meta = json.load(f)
        status_file = pjoin(task_dir, "status.json")
        status = NotImplemented
        if os.path.exists(status_file):
            with open(status_file) as f:
                status_meta = json.load(f)
            status = status_meta['is_finish'] 
                
        else:
            continue
        task_id = meta["task_id"]
        this_result = {}
        
        this_result["instance_id"] = task_id
        this_result["model_name_or_path"] = common.SELECTED_MODEL.name
        docker_content = ""
        eval_script_content = ""
        if os.path.exists(docker_file):
            with open(docker_file) as f:
                docker_content = f.read()
        eval_script_file = docker_file.replace('Dockerfile','eval.sh')
        if os.path.exists(eval_script_file):
            with open(eval_script_file) as f:
                eval_script_content = f.read()
        # if not docker_content:
        #     # empty diff file, dont bother sending it to swe-bench
        #     continue
        this_result["dockerfile"] = docker_content
        this_result["eval_script"] = eval_script_content
        this_result['version'] = meta['task_info']['version']
        this_result['repo'] = meta['task_info']['repo']
        this_result['patch'] = meta['task_info']['patch']
        this_result['status'] = status
        all_results.append(this_result)
        if status == True:
            final_results.append(this_result)

    final_predictions_file = pjoin(dir, "predictions.json")
    raw_predictions_file = pjoin(dir, "raw_predictions.json")
    with open(final_predictions_file, "w") as f:
        json.dump(final_results, f, indent=4)

    with open(raw_predictions_file, "w") as f:
        json.dump(all_results, f, indent=4)

    return final_predictions_file


def is_valid_json(json_str: str) -> tuple[ExtractStatus, list | dict | None]:
    """
    Check whether a json string is valid.
    """
    try:
        data = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        return ExtractStatus.NOT_VALID_JSON, None
    return ExtractStatus.IS_VALID_JSON, data


"""
Main entries of the module.
"""



def un_classify_expr_dir(expr_dir: str):
    individual_expr_dirs = []
    for individual_expr_dir in glob(pjoin(expr_dir, "*", "*__*")):
        assert "info.log" in os.listdir(
            individual_expr_dir
        ), f"{individual_expr_dir} has no info.log"
        individual_expr_dirs.append(individual_expr_dir)

    for d in individual_expr_dirs:
        move(d, expr_dir)




def organize_and_form_input(expr_dir):
    """
    Only organize the experiment directories into a few categories.
    Args:
        - expr_dir: the overall experiment directory.
    """
    organize_experiment_results(expr_dir)
    swe_input_file = extract_swe_bench_input(expr_dir)
    return swe_input_file

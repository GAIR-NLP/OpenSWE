import subprocess
import json
import yaml
import os
import shutil
import signal
import pickle
import logging
import time
import psutil
import random
from filelock import FileLock
from datetime import datetime
from logging import Logger
from dataclasses import dataclass
from pydantic import BaseModel, TypeAdapter, ConfigDict
from prometheus_client import Gauge, start_http_server
from functools import cached_property

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prometheus metrics
metrics_completed_tasks = Gauge('worker_completed_tasks_total', 'Total number of completed tasks')
metrics_running_processes = Gauge('worker_running_processes', 'Number of currently running processes')
metrics_unfinished_tasks = Gauge('worker_unfinished_tasks', 'Number of tasks in the unfinished queue')
metrics_failed_restart_tasks = Gauge('worker_failed_restart_tasks', 'Number of failed restart tasks')
metrics_timeout_tasks = Gauge('worker_timeout_tasks', 'Number of tasks that timed out')

class SWETask(BaseModel):
    """Represents a software engineering task."""
    model_config = ConfigDict(frozen=True)

    instance_id: str
    repo: str
    pull_number: int
    base_commit: str
    end_commit: str
    test_patch: str
    problem_statement: str
    repo_id: int
    patch: str
    version: str
    language: str

    def to_jsonl(self, fpath: str):
        """Save task to a JSON file."""
        with open(fpath, 'w') as f:
            f.write(self.model_dump_json())

@dataclass
class RunningTask:
    """Represents a currently running task process."""
    task: SWETask
    process: subprocess.Popen
    stdout_fpath: str
    stderr_fpath: str

    @property
    def stdout(self) -> str:
        with open(self.stdout_fpath, "r") as f:
            return f.read()
    
    @property
    def stderr(self) -> str:
        with open(self.stderr_fpath, "r") as f:
            return f.read()
    
    @classmethod
    def from_task(cls, run_cmd: list[str], task: SWETask, cwd: str, env: dict):
        if os.path.exists(os.path.join(cwd, "output")):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            shutil.move(os.path.join(cwd, "output"), os.path.join(cwd, f"output{timestamp}"))
        
        stdout_fpath = os.path.join(cwd, "stdout.log")
        stderr_fpath = os.path.join(cwd, "stderr.log")
        
        with open(stdout_fpath, "w") as stdout_fd, open(stderr_fpath, "w") as stderr_fd:
            process = subprocess.Popen(
                run_cmd,
                stderr=stderr_fd,
                stdout=stdout_fd,
                start_new_session=True,
                cwd=cwd,
                text=True,
                errors='replace',
                env=env,
            )
        
        return cls(task, process, stdout_fpath, stderr_fpath)

def kill_process_tree(pid: int, timeout: int = 5):
    """Recursively kill process tree, including all children."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Try graceful termination first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            return
        
        # Wait for processes to exit
        gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
        
        # Force kill remaining processes
        for p in alive:
            try:
                logger.warning(f"Force killing process {p.pid}: {p.name()}")
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.error(f"Failed to kill process tree {pid}: {e}")

class OpenAIAPI(BaseModel):
    base_url: str
    api_key: str
    weights: float

    @cached_property
    def env_dic(self) -> dict[str, str]:
        return {"BASEURL": self.base_url, "APIKEY": self.api_key}

class DynamicConfig(BaseModel):
    process_num: int
    openai_apis: list[OpenAIAPI]

    @cached_property
    def openai_weights(self) -> list[float]:
        return [i.weights for i in self.openai_apis]

    @classmethod
    def from_yaml(cls, fpath: str):
        with open(fpath, "r") as f:
            return cls.model_validate(yaml.safe_load(f))
    
    def to_yaml(self, fpath: str):
        with open(fpath, "w") as f:
            yaml.dump(self.model_dump(mode='json'), f)

    def gen_env_dic(self) -> dict[str, str]:
        use_api = random.choices(self.openai_apis, weights=self.openai_weights, k=1)[0]
        return use_api.env_dic

@dataclass
class Worker:
    task_que_dir: str
    dynamic_config: DynamicConfig
    data_dir: str
    run_command: list[str]
    check_gap: float
    prometheus_port: int = 8000

    def __post_init__(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self.tmpdir = os.path.join(self.data_dir, "worktmp")
        if os.path.exists(self.tmpdir):
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            self.old_data_dir = os.path.join(self.data_dir, f"old{timestamp}")
            shutil.move(self.tmpdir, self.old_data_dir)
        os.makedirs(self.tmpdir)

        self.loggers: list[Logger] = []
        for i in range(self.dynamic_config.process_num):
            workdir = self.get_workdir(i)
            os.makedirs(workdir)
            loggeri = logging.getLogger(f'worker_{i}')
            loggeri.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(workdir, "work.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            loggeri.addHandler(file_handler)
            loggeri.propagate = False
            self.loggers.append(loggeri)

        main_log_file = os.path.join(self.data_dir, "work.log")
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        self.processes: dict[int, RunningTask] = {}
        self.free_process: list[int] = list(range(self.dynamic_config.process_num))
        self.unfinished_data_file = os.path.join(self.data_dir, "unfinished.pkl")
        self.completed_tasks_count = 0
        self.failed_restart_file = os.path.join(self.data_dir, "failed_restart.pkl")
        self.failed_restart: list[SWETask] = [] 
        self.load_failed()

        self.config_file = os.path.join(self.data_dir, "dynamic_config.yaml")
        self.dynamic_config.to_yaml(self.config_file)
        self.config_file_lock = FileLock(f"{self.config_file}.lock")
        
        if not os.path.exists(self.unfinished_data_file):
            logger.info(f"No unfinished task file found at {self.unfinished_data_file}")
            self.unfinished_data: set[SWETask] = set()
            self.pending_data: list[SWETask] = []
            self.save_unfinished()
        else:
            self.load_unfinished()
            self.pending_data = list(self.unfinished_data)
        
        logger.info(f"Initialized with {len(self.pending_data)} tasks")
        start_http_server(self.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        
        self.update_metrics()
        self.should_exit = False
    
    def get_workdir(self, pind: int) -> str:
        return os.path.join(self.tmpdir, f"p{pind}")
    
    def get_realworkdir(self, pind: int, task: SWETask) -> str:
        use_path = os.path.join(self.tmpdir, f"p{pind}", task.instance_id)
        os.makedirs(use_path, exist_ok=True)
        return use_path

    def save_unfinished(self):
        if os.path.exists(self.unfinished_data_file):
            shutil.copy2(self.unfinished_data_file, f"{self.unfinished_data_file}.bak")
        
        with open(self.unfinished_data_file, 'wb') as f:
            pickle.dump(self.unfinished_data, f)

    def load_unfinished(self):
        try:
            with open(self.unfinished_data_file, 'rb') as f:
                self.unfinished_data = pickle.load(f)
        except Exception:
            with open(f"{self.unfinished_data_file}.bak", 'rb') as f:
                self.unfinished_data = pickle.load(f)

    def update_metrics(self):
        """Update all Prometheus metrics."""
        metrics_running_processes.set(len(self.processes))
        metrics_unfinished_tasks.set(len(self.unfinished_data))
    
    def update_config(self, new_config: DynamicConfig):
        new_process_num = new_config.process_num
        if new_process_num <= 0:
            logger.error(f"Invalid process number: {new_process_num}, must be positive")
            return
        
        old_process_num = self.dynamic_config.process_num
        self.dynamic_config = new_config
        if new_process_num == old_process_num:
            logger.info(f"Process number unchanged: {new_process_num}")
            return
        
        if new_process_num > old_process_num:
            # Increase process count: create workdirs and loggers
            for i in range(old_process_num, new_process_num):
                workdir = self.get_workdir(i)
                os.makedirs(workdir, exist_ok=True)
                
                loggeri = logging.getLogger(f'worker_{i}')
                loggeri.setLevel(logging.INFO)
                file_handler = logging.FileHandler(os.path.join(workdir, "work.log"))
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
                loggeri.addHandler(file_handler)
                loggeri.propagate = False
                self.loggers.append(loggeri)
                
                self.free_process.append(i)
            
            logger.info(f"Increased process number from {old_process_num} to {new_process_num}")
        else:
            processes_to_remove = []
            for worker_id in range(new_process_num, old_process_num):
                if worker_id in self.processes:
                    # Stop running processes
                    running_task = self.processes[worker_id]
                    logger.info(f"Stopping worker {worker_id} running task {running_task.task.instance_id}")
                    kill_process_tree(running_task.process.pid)
                    
                    # Re-add tasks to pending queue
                    self.pending_data.insert(0, running_task.task)
                    processes_to_remove.append(worker_id)
                
                # Remove from free processes
                if worker_id in self.free_process:
                    self.free_process.remove(worker_id)
            
            # Clean up stopped processes
            for worker_id in processes_to_remove:
                del self.processes[worker_id]
            
            logger.info(f"Decreased process number from {old_process_num} to {new_process_num}")
            if processes_to_remove:
                logger.info(f"Stopped workers: {processes_to_remove}, tasks returned to queue")

        self.update_metrics()
    
    def signal_handler(self, signum, frame):
        """Handle exit signals."""
        print("\nReceived exit signal, exiting gracefully...")
        self.should_exit = True

    def read_task(self):
        buffer_lst = sorted(os.listdir(self.task_que_dir))
        if not buffer_lst:
            return
        use_fname = buffer_lst[0]
        use_fpath = os.path.join(self.task_que_dir, use_fname)
        with open(use_fpath, 'r') as f:
            for fi in f:
                taski = SWETask(**json.loads(fi))
                self.pending_data.append(taski)
                self.unfinished_data.add(taski)
        logger.info(f"Loaded data from {use_fpath}, total pending: {len(self.pending_data)}")
        os.remove(use_fpath)

    def load_failed(self):
        if not os.path.exists(self.failed_restart_file):
            self.failed_restart = []
            self.save_failed()
            return
        with open(self.failed_restart_file, 'rb') as f:
            self.failed_restart = pickle.load(f)

    def save_failed(self):
        if os.path.exists(self.failed_restart_file):
            shutil.copy2(self.failed_restart_file, f"{self.failed_restart_file}.bak")
        with open(self.failed_restart_file, 'wb') as f:
            pickle.dump(self.failed_restart, f)

    def _handle_failed_process(self, worker_id: int, running_task: RunningTask):
        """Handle failed process."""
        try:
            logger.warning(f"Worker {worker_id} with {running_task.task.instance_id} failed")
            self.loggers[worker_id].warning(f"Worker {worker_id} with {running_task.task} failed")
            self.loggers[worker_id].warning(f"stdout=\n{running_task.stdout}")
            self.loggers[worker_id].warning(f"stderr=\n{running_task.stderr}")
        except Exception:
            self.loggers[worker_id].error(f"Could not get logs for worker {worker_id} with {running_task.task}")
        
        self.free_process.append(worker_id)
        self.failed_restart.append(running_task.task)
        self.unfinished_data.remove(running_task.task)
        metrics_failed_restart_tasks.inc()
        self.save_failed()

    def _handle_completed_process(self, worker_id: int, running_task: RunningTask):
        """Handle completed process."""
        logger.info(f"Worker {worker_id} with {running_task.task.instance_id} finished")
        self.loggers[worker_id].info(f"Worker {worker_id} with {running_task.task} finished")
        self.loggers[worker_id].info(f"stdout=\n{running_task.stdout}")
        self.loggers[worker_id].info(f"stderr=\n{running_task.stderr}")
        
        self.free_process.append(worker_id)
        self.completed_tasks_count += 1
        metrics_completed_tasks.inc()
        self.unfinished_data.remove(running_task.task)

    def check_finished(self):
        """Check and handle finished tasks."""
        new_processes = {}
        for worker_id, running_task in self.processes.items():
            returncode = running_task.process.poll()
            if returncode is None:
                new_processes[worker_id] = running_task
                continue
            
            if returncode != 0:
                if returncode == 124:
                    metrics_timeout_tasks.inc()
                self._handle_failed_process(worker_id, running_task)
            else:
                self._handle_completed_process(worker_id, running_task)
        
        self.processes = new_processes

    def _start_task(self, task: SWETask, worker_id: int):
        """Start a new task."""
        workdiri = self.get_realworkdir(worker_id, task)
        task_file = os.path.join(workdiri, "task.jsonl")
        task.to_jsonl(task_file)
        use_env = os.environ.copy()
        use_env.update(self.dynamic_config.gen_env_dic())
        use_env.update({"INFILE": task_file, "TMPWDIR": workdiri})
        self.processes[worker_id] = RunningTask.from_task(
            self.run_command,
            task=task,
            cwd=workdiri,
            env=use_env
        )
    
    def _allocate_tasks(self):
        """Allocate tasks to free processes."""
        while self.free_process and self.pending_data:
            task = self.pending_data.pop(0)
            worker_id = self.free_process.pop(0)
            self._start_task(task, worker_id)
    
    def _cleanup(self):
        """Clean up resources and save state."""
        print("Saving unfinished data...")
        self.save_unfinished()
        
        for running_task in self.processes.values():
            kill_process_tree(running_task.process.pid)
        
        print(f"Exited. Completed: {self.completed_tasks_count}, Failed: {len(self.failed_restart)}")

    def reload_config(self):
        with self.config_file_lock:
            try:
                new_config = DynamicConfig.from_yaml(self.config_file)
                if new_config != self.dynamic_config:
                    self.update_config(new_config)
            except Exception:
                logger.warning(f"Failed to read config, rewriting current config")
                self.dynamic_config.to_yaml(self.config_file)

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        print("Worker started successfully. Press Ctrl+C to exit.")
        
        try:
            while not self.should_exit:
                self.check_finished()
                
                # Read new tasks if pending tasks are low
                if len(self.pending_data) < len(self.free_process):
                    self.read_task()
                
                # Allocate tasks to free processes
                self._allocate_tasks()
                
                self.save_unfinished()
                self.update_metrics()
                self.reload_config()
                time.sleep(self.check_gap)
        finally:
            self._cleanup()

    @classmethod
    def from_yaml(cls, fpath: str, **kwargs):
        with open(fpath, "r") as f:
            data: dict = yaml.safe_load(f)
            data.update(kwargs)
            return TypeAdapter(cls).validate_python(data)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', default='work.yaml', type=str, help='Config file path')
    parser.add_argument('--taskdir', type=str)
    parser.add_argument('--datadir', type=str)
    args = parser.parse_args()

    Worker.from_yaml(args.c, task_que_dir=args.taskdir, data_dir=args.datadir).run()

import os
import shutil
import time
import json
import yaml
import argparse
import signal
from contextlib import contextmanager
from pydantic.dataclasses import dataclass
from prometheus_client import Gauge, start_http_server

@dataclass
class SplitMeta:
    total_num: int
    file_meta: dict[str, dict[str, int]]

@dataclass
class ManagerCfg:
    data_dir: str
    node_num: int
    task_que_dir: str
    check_gap: float
    buffer_size: int
    prometheus_port: int

    def __post_init__(self):
        os.makedirs(self.task_que_dir, exist_ok=True)
        self.task_ques: list[str] = []
        for i in range(self.node_num):
            que_path = os.path.join(self.task_que_dir, f"task{i}")
            self.task_ques.append(que_path)
            os.makedirs(que_path, exist_ok=True)
        
        self.unused_data_file = os.path.join(self.data_dir, "unused.json")

        meta_path = os.path.join(self.data_dir, "meta.json")
        with open(meta_path, "r") as f:
            self.meta_data = SplitMeta(**json.load(f))

        if os.path.exists(self.unused_data_file):
            self.load_unused()
        else:
            print("No unused data found, assuming all files are unused")
            self.unused_file = [os.path.join(self.data_dir, i) for i in self.meta_data.file_meta]
            self.save_unused()

        print(f"Loaded unused data: {len(self.unused_file)} files")
        
        # Initialize Prometheus metrics
        self.total_buffer_gauge = Gauge('task_buffer_total', 'Total number of tasks in all buffers')
        self.tasks_sent_gauge = Gauge('tasks_sent_total', 'Total number of tasks sent to workers')
        self.tasks_sent_gauge.set(0)
        
        # Start Prometheus HTTP server
        start_http_server(self.prometheus_port)
        self.should_exit = False
    
    def load_unused(self):
        with open(self.unused_data_file, "r") as f:
            self.unused_file = json.load(f)

    def save_unused(self):
        with open(self.unused_data_file, "w") as f:
            json.dump(self.unused_file, f)
    
    @contextmanager
    def atomic_task_dispatch(self):
        """Ensure atomicity of task dispatch and unused file update."""
        dispatched_info = []  # List of (task_path, dest_dir)
        try:
            yield dispatched_info
            # Update unused_file only after success
            for task_path, _ in dispatched_info:
                self.unused_file.remove(task_path)
            self.save_unused()
        except Exception as e:
            # Rollback copied files on error
            print(f"Task dispatch error: {e}, rolling back...")
            for task_path, dest_dir in dispatched_info:
                dest_file = os.path.join(dest_dir, os.path.basename(task_path))
                if os.path.exists(dest_file):
                    os.remove(dest_file)
            raise

    def signal_handler(self, signum, frame):
        """Handle exit signals."""
        print("\nReceived exit signal, exiting gracefully...")
        self.should_exit = True

    def _dispatch_tasks_to_queue(self, task_queue: str, current_buffer: int) -> int:
        """Dispatch tasks to queue, return count."""
        if current_buffer >= self.buffer_size or not self.unused_file:
            return 0
        
        tasks_to_send = min(self.buffer_size - current_buffer, len(self.unused_file))
        
        with self.atomic_task_dispatch() as dispatched:
            for i in range(tasks_to_send):
                task_path = self.unused_file[i]
                shutil.copy2(task_path, task_queue)
                dispatched.append((task_path, task_queue))
        
        return tasks_to_send

    def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        tasks_sent = 0
        print("Master started successfully. Press Ctrl+C to exit.")
        
        try:
            while not self.should_exit:
                total_buffer = 0
                for task_queue in self.task_ques:
                    now_buffer = len(os.listdir(task_queue))
                    total_buffer += now_buffer
                    
                    new_tasks = self._dispatch_tasks_to_queue(task_queue, now_buffer)
                    tasks_sent += new_tasks
                
                # Update Prometheus metrics
                self.total_buffer_gauge.set(total_buffer)
                self.tasks_sent_gauge.set(tasks_sent)
                time.sleep(self.check_gap)
        finally:
            print("Saving unused data...")
            self.save_unused()
            print(f"Master exited. Tasks sent: {tasks_sent}")

    @classmethod
    def from_yaml(cls, fpath: str):
        with open(fpath, "r") as f:
            return cls(**yaml.safe_load(f))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="Configuration file")
    args = parser.parse_args()

    ManagerCfg.from_yaml(args.cfg).run()

import os
import subprocess
import sys
import threading
import uuid
from enum import Enum
from typing import Dict, List, Optional

import psutil

from lora_forge.log import log


def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


class TaskStatus(Enum):
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    TERMINATED = 3


class Task:
    def __init__(self, task_id, command, environ=None, run_id=None):
        self.task_id = task_id
        self.lock = threading.Lock()
        self.command = command
        self.status = TaskStatus.CREATED
        self.environ = environ or os.environ
        self.process = None
        # the monitor's record of this task (lora_forge.training.runs); its output is captured there
        self.run_id = run_id
        self.output_pump = None

    def _mark_finished(self):
        # A terminated task stays TERMINATED once its process exits.
        with self.lock:
            if self.status != TaskStatus.TERMINATED:
                self.status = TaskStatus.FINISHED

    def wait(self) -> int:
        """Wait for the process to exit (and its output to be copied); returns its exit code."""
        try:
            returncode = self.process.wait()
        except BaseException:
            self.process.kill()
            raise
        finally:
            if self.output_pump is not None:
                self.output_pump.join(timeout=10)
        self._mark_finished()
        return returncode

    def execute(self) -> bool:
        """Start the process. Returns False if the task was terminated before it started."""
        with self.lock:
            if self.status == TaskStatus.TERMINATED:
                return False
            try:
                if self.run_id:
                    from lora_forge.training.runs import OutputPump

                    self.process = subprocess.Popen(
                        self.command, env=self.environ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    self.output_pump = OutputPump(self.process.stdout, self.run_id, sys.stdout)
                    self.output_pump.start()
                else:
                    self.process = subprocess.Popen(self.command, env=self.environ)
            except Exception:
                # never leave a task RUNNING without a process: it would block every later task
                self.status = TaskStatus.FINISHED
                raise
            self.status = TaskStatus.RUNNING
            return True

    def terminate(self):
        with self.lock:
            if self.status not in (TaskStatus.CREATED, TaskStatus.RUNNING):
                return
            self.status = TaskStatus.TERMINATED
            process = self.process
        if process is None:
            return  # not started yet; execute() sees TERMINATED and does not start it
        try:
            kill_proc_tree(process.pid, False)
        except Exception as e:
            log.error(f"Error when killing process: {e}")


class TaskManager:
    # Keep a bounded history of finished/terminated tasks so a long session does not grow forever.
    MAX_RETAINED_TASKS = 20

    def __init__(self, max_concurrent=1) -> None:
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}

    def _prune(self):
        done = [
            task_id for task_id, task in self.tasks.items()
            if task.status in (TaskStatus.FINISHED, TaskStatus.TERMINATED)
        ]
        for task_id in done[: -self.MAX_RETAINED_TASKS]:
            self.tasks.pop(task_id, None)

    def create_task(self, command: List[str], environ, run_id: Optional[str] = None):
        self._prune()
        # CREATED counts too: a task is CREATED until its worker thread starts the process, and a
        # second /run arriving in that window must not start another training.
        running_tasks = [t for _, t in self.tasks.items() if t.status in (TaskStatus.CREATED, TaskStatus.RUNNING)]
        if len(running_tasks) >= self.max_concurrent:
            log.error(
                f"Unable to create a task because there are already {len(running_tasks)} tasks running, reaching the maximum concurrent limit.")
            return None
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, command=command, environ=environ, run_id=run_id)
        self.tasks[task_id] = task
        # task.execute() # breaking change
        log.info(f"Task {task_id} created")
        return task

    def terminate_task(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.terminate()

    def find_by_run(self, run_id: str) -> Optional[Task]:
        return next((task for task in self.tasks.values() if task.run_id == run_id), None)

    def dump(self) -> List[Dict]:
        return [
            {
                "id": task.task_id,
                "status": task.status.name,
                "run_id": task.run_id,
            }
            for task in self.tasks.values()
        ]


tm = TaskManager()

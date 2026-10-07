"""Asynchronous Background Task Manager for AstroFoundation Long-Running Analyses."""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TaskRecord(BaseModel):
    task_id: str
    status: TaskStatus
    created_at: str
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BackgroundTaskManager:
    """Manages asynchronous background jobs for telescope batch analysis."""

    def __init__(self):
        self._tasks: Dict[str, TaskRecord] = {}

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        record = TaskRecord(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._tasks[task_id] = record
        return task_id

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, TaskRecord]:
        return self._tasks

    async def run_async_job(self, task_id: str, job_fn: Callable[[], Any]):
        """Execute a task in an asyncio worker thread and record the outcome."""
        if task_id not in self._tasks:
            return

        self._tasks[task_id].status = TaskStatus.RUNNING
        try:
            # Run blocking CPU/Network task in default loop executor
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, job_fn)

            self._tasks[task_id].status = TaskStatus.SUCCESS
            self._tasks[task_id].finished_at = datetime.now(timezone.utc).isoformat()
            if hasattr(result, "model_dump"):
                self._tasks[task_id].result = result.model_dump()
            elif isinstance(result, dict):
                self._tasks[task_id].result = result
            else:
                self._tasks[task_id].result = {"output": str(result)}
        except Exception as exc:
            logger.exception("Background task %s failed: %s", task_id, exc)
            self._tasks[task_id].status = TaskStatus.FAILED
            self._tasks[task_id].finished_at = datetime.now(timezone.utc).isoformat()
            self._tasks[task_id].error = str(exc)


# Global task manager instance
task_manager = BackgroundTaskManager()

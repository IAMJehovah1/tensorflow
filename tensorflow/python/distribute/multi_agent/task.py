# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Task definition for the Multi-Agent Task Coordination System."""

import enum
import threading
import time
import uuid


class TaskStatus(enum.Enum):
  """Status of a task in the coordination system."""
  PENDING = "pending"
  RUNNING = "running"
  COMPLETED = "completed"
  FAILED = "failed"
  CANCELLED = "cancelled"


class Task:
  """Represents a unit of work to be executed by an agent.

  A Task encapsulates a callable function along with metadata such as its
  priority, status, and result.  Tasks are enqueued with the
  `TaskCoordinator`, which assigns them to available agents for execution.

  Example usage::

      def my_work(x, y):
          return x + y

      task = Task(fn=my_work, args=(1, 2), priority=5)
      # After execution:
      # task.result == 3
      # task.status == TaskStatus.COMPLETED

  Args:
      fn: A callable that will be invoked by the executing agent.
      args: Positional arguments to pass to *fn*.
      kwargs: Keyword arguments to pass to *fn*.
      priority: An integer priority.  Higher values indicate higher priority.
          Defaults to ``0``.
      task_id: An optional string identifier.  A UUID is generated when not
          provided.
  """

  def __init__(self, fn, args=None, kwargs=None, priority=0, task_id=None):
    self._fn = fn
    self._args = args or ()
    self._kwargs = kwargs or {}
    self._priority = priority
    self._task_id = task_id or str(uuid.uuid4())
    self._status = TaskStatus.PENDING
    self._result = None
    self._exception = None
    self._created_at = time.monotonic()
    self._started_at = None
    self._completed_at = None
    self._lock = threading.Lock()
    self._done_event = threading.Event()

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def task_id(self):
    """Unique identifier for this task."""
    return self._task_id

  @property
  def priority(self):
    """Priority of this task (higher == more urgent)."""
    return self._priority

  @property
  def status(self):
    """Current `TaskStatus` of this task."""
    with self._lock:
      return self._status

  @property
  def result(self):
    """Return value of the task function, or ``None`` if not yet completed."""
    with self._lock:
      return self._result

  @property
  def exception(self):
    """Exception raised during execution, or ``None`` if successful."""
    with self._lock:
      return self._exception

  @property
  def created_at(self):
    """Monotonic timestamp at which the task was created."""
    return self._created_at

  @property
  def started_at(self):
    """Monotonic timestamp at which execution began, or ``None``."""
    with self._lock:
      return self._started_at

  @property
  def completed_at(self):
    """Monotonic timestamp at which the task finished, or ``None``."""
    with self._lock:
      return self._completed_at

  # ------------------------------------------------------------------
  # Internal state transitions (called by agents/coordinator)
  # ------------------------------------------------------------------

  def _mark_running(self):
    with self._lock:
      self._status = TaskStatus.RUNNING
      self._started_at = time.monotonic()

  def _mark_completed(self, result):
    with self._lock:
      self._result = result
      self._status = TaskStatus.COMPLETED
      self._completed_at = time.monotonic()
    self._done_event.set()

  def _mark_failed(self, exception):
    with self._lock:
      self._exception = exception
      self._status = TaskStatus.FAILED
      self._completed_at = time.monotonic()
    self._done_event.set()

  def _mark_cancelled(self):
    with self._lock:
      if self._status == TaskStatus.PENDING:
        self._status = TaskStatus.CANCELLED
        self._completed_at = time.monotonic()
        self._done_event.set()
        return True
      return False

  # ------------------------------------------------------------------
  # Public helpers
  # ------------------------------------------------------------------

  def wait(self, timeout=None):
    """Block until the task reaches a terminal state.

    Args:
        timeout: Maximum seconds to wait.  ``None`` waits indefinitely.

    Returns:
        ``True`` if the task completed within the timeout, ``False`` otherwise.
    """
    return self._done_event.wait(timeout=timeout)

  def _execute(self):
    """Execute the task's callable.  Called internally by agents."""
    self._mark_running()
    try:
      result = self._fn(*self._args, **self._kwargs)
      self._mark_completed(result)
    except Exception as exc:  # pylint: disable=broad-except
      self._mark_failed(exc)

  # ------------------------------------------------------------------
  # Comparison (used by priority queue)
  # ------------------------------------------------------------------

  def __lt__(self, other):
    # Higher priority value → processed first; break ties by creation time.
    if self._priority != other._priority:
      return self._priority > other._priority
    return self._created_at < other._created_at

  def __le__(self, other):
    return self == other or self < other

  def __eq__(self, other):
    if not isinstance(other, Task):
      return NotImplemented
    return self._task_id == other._task_id

  def __hash__(self):
    return hash(self._task_id)

  def __repr__(self):
    return (
        f"Task(id={self._task_id!r}, priority={self._priority}, "
        f"status={self._status.value})"
    )

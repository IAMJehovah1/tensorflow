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
"""TaskCoordinator for the Multi-Agent Task Coordination System.

The coordinator manages a pool of `Agent` workers and a priority queue of
`Task` objects.  Clients submit tasks via `schedule`, optionally wait for
them with `join`, and can inspect real-time statistics through
`get_stats`.

Example usage::

    from tensorflow.python.distribute.multi_agent.coordinator import (
        TaskCoordinator,
    )
    from tensorflow.python.distribute.multi_agent.task import Task

    coordinator = TaskCoordinator(num_agents=4)
    coordinator.start()

    def compute(x):
        return x * x

    tasks = [coordinator.schedule(compute, args=(i,)) for i in range(10)]
    coordinator.join()

    results = [t.result for t in tasks]
    coordinator.stop()
"""

import heapq
import threading

from tensorflow.python.distribute.multi_agent.agent import Agent
from tensorflow.python.distribute.multi_agent.task import Task
from tensorflow.python.distribute.multi_agent.task import TaskStatus


class TaskCoordinator:
  """Coordinates task execution across a pool of agents.

  The coordinator maintains an internal priority queue of `Task` objects and
  dispatches them to idle `Agent` workers running on background threads.
  Tasks with higher ``priority`` values are executed before lower-priority
  ones; ties are broken by creation order (FIFO).

  Args:
      num_agents: Number of agent workers to create.  Must be >= 1.

  Raises:
      ValueError: If *num_agents* < 1.
  """

  def __init__(self, num_agents=1):
    if num_agents < 1:
      raise ValueError(
          f"num_agents must be >= 1, got {num_agents}."
      )
    self._num_agents = num_agents
    self._queue = []          # heap of (Task,) – ordering defined by Task.__lt__
    self._queue_lock = threading.Lock()
    self._queue_not_empty = threading.Condition(self._queue_lock)
    self._pending_count = 0   # tasks in queue + currently running
    self._pending_lock = threading.Lock()
    self._all_done = threading.Event()
    self._all_done.set()      # nothing pending at start
    self._stats_lock = threading.Lock()
    self._total_scheduled = 0
    self._total_completed = 0
    self._total_failed = 0
    self._total_cancelled = 0
    self._agents = [
        Agent(agent_id=str(i), coordinator=self)
        for i in range(num_agents)
    ]
    self._started = False
    self._stopped = False

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def start(self):
    """Start all agent threads.

    This must be called before submitting tasks.  Calling ``start`` on an
    already-started coordinator has no effect.
    """
    if self._started:
      return
    self._started = True
    for agent in self._agents:
      agent.start()

  def stop(self, wait=True, timeout=None):
    """Stop all agents and optionally wait for them to finish.

    Args:
        wait: If ``True`` (default), block until all agents have exited.
        timeout: Per-agent join timeout in seconds.  ``None`` waits
            indefinitely.
    """
    self._stopped = True
    for agent in self._agents:
      agent.stop()
    # Unblock any agents that are blocked on _get_next_task.
    with self._queue_not_empty:
      self._queue_not_empty.notify_all()
    if wait:
      for agent in self._agents:
        agent.join(timeout=timeout)

  # ------------------------------------------------------------------
  # Task submission
  # ------------------------------------------------------------------

  def schedule(self, fn, args=None, kwargs=None, priority=0, task_id=None):
    """Enqueue a task for execution and return the `Task` object.

    Args:
        fn: Callable to execute.
        args: Positional arguments for *fn*.
        kwargs: Keyword arguments for *fn*.
        priority: Integer priority; higher values run first.
        task_id: Optional explicit task ID string.

    Returns:
        The `Task` object representing the submitted work.

    Raises:
        RuntimeError: If the coordinator has been stopped.
    """
    if self._stopped:
      raise RuntimeError("Cannot schedule tasks on a stopped coordinator.")
    task = Task(
        fn=fn,
        args=args,
        kwargs=kwargs,
        priority=priority,
        task_id=task_id,
    )
    with self._stats_lock:
      self._total_scheduled += 1
    with self._pending_lock:
      self._pending_count += 1
      self._all_done.clear()
    with self._queue_not_empty:
      heapq.heappush(self._queue, task)
      self._queue_not_empty.notify()
    return task

  def cancel(self, task):
    """Attempt to cancel a pending task.

    Cancellation is only possible while the task is in ``PENDING`` status.
    Tasks that have already started executing cannot be cancelled.

    Args:
        task: The `Task` to cancel.

    Returns:
        ``True`` if the task was successfully cancelled, ``False`` otherwise.
    """
    cancelled = task._mark_cancelled()  # pylint: disable=protected-access
    if cancelled:
      with self._stats_lock:
        self._total_cancelled += 1
      self._decrement_pending()
    return cancelled

  # ------------------------------------------------------------------
  # Waiting
  # ------------------------------------------------------------------

  def join(self, timeout=None):
    """Block until all currently scheduled tasks have finished.

    Args:
        timeout: Maximum seconds to wait.  ``None`` waits indefinitely.

    Returns:
        ``True`` if all tasks completed within *timeout*, ``False`` if
        the timeout expired while tasks were still pending.
    """
    return self._all_done.wait(timeout=timeout)

  # ------------------------------------------------------------------
  # Statistics
  # ------------------------------------------------------------------

  def get_stats(self):
    """Return a snapshot of coordination statistics.

    Returns:
        A ``dict`` with the following keys:

        - ``"num_agents"``: total number of agents.
        - ``"agents_idle"``: number of agents currently idle.
        - ``"agents_busy"``: number of agents currently executing a task.
        - ``"queue_size"``: number of tasks waiting in the queue.
        - ``"total_scheduled"``: total tasks submitted since start.
        - ``"total_completed"``: tasks finished successfully.
        - ``"total_failed"``: tasks that raised an exception.
        - ``"total_cancelled"``: tasks cancelled before execution.
    """
    from tensorflow.python.distribute.multi_agent.agent import AgentStatus  # pylint: disable=g-import-not-at-top
    with self._stats_lock:
      total_scheduled = self._total_scheduled
      total_completed = self._total_completed
      total_failed = self._total_failed
      total_cancelled = self._total_cancelled
    agents_idle = sum(
        1 for a in self._agents if a.status == AgentStatus.IDLE
    )
    agents_busy = sum(
        1 for a in self._agents if a.status == AgentStatus.BUSY
    )
    with self._queue_lock:
      queue_size = len(self._queue)
    return {
        "num_agents": self._num_agents,
        "agents_idle": agents_idle,
        "agents_busy": agents_busy,
        "queue_size": queue_size,
        "total_scheduled": total_scheduled,
        "total_completed": total_completed,
        "total_failed": total_failed,
        "total_cancelled": total_cancelled,
    }

  @property
  def agents(self):
    """List of `Agent` objects managed by this coordinator."""
    return list(self._agents)

  # ------------------------------------------------------------------
  # Internal helpers (called by agents)
  # ------------------------------------------------------------------

  def _get_next_task(self, timeout=0.1):
    """Return the highest-priority pending task, blocking up to *timeout* s."""
    with self._queue_not_empty:
      deadline = None
      if timeout is not None:
        import time  # pylint: disable=g-import-not-at-top
        deadline = time.monotonic() + timeout
      while not self._queue and not self._stopped:
        if deadline is not None:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
            return None
          self._queue_not_empty.wait(timeout=remaining)
        else:
          self._queue_not_empty.wait()
      if self._stopped and not self._queue:
        return None
      # Skip tasks that were cancelled while in the queue.
      while self._queue:
        task = heapq.heappop(self._queue)
        if task.status != TaskStatus.CANCELLED:
          return task
      return None

  def _on_task_done(self, task):
    """Called by an agent once a task has reached a terminal state."""
    with self._stats_lock:
      if task.status == TaskStatus.COMPLETED:
        self._total_completed += 1
      else:
        self._total_failed += 1
    self._decrement_pending()

  def _decrement_pending(self):
    with self._pending_lock:
      self._pending_count -= 1
      if self._pending_count <= 0:
        self._pending_count = 0
        self._all_done.set()

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
"""Agent definition for the Multi-Agent Task Coordination System."""

import enum
import threading

from tensorflow.python.distribute.multi_agent.task import TaskStatus


class AgentStatus(enum.Enum):
  """Lifecycle status of an agent."""
  IDLE = "idle"
  BUSY = "busy"
  STOPPED = "stopped"


class Agent:
  """A worker that retrieves tasks from a coordinator and executes them.

  Agents run on a dedicated background thread and continuously pull tasks
  from the `TaskCoordinator` task queue until they are stopped.

  Args:
      agent_id: A string identifier for this agent.
      coordinator: The `TaskCoordinator` that this agent belongs to.
  """

  def __init__(self, agent_id, coordinator):
    self._agent_id = agent_id
    self._coordinator = coordinator
    self._status = AgentStatus.IDLE
    self._current_task = None
    self._tasks_completed = 0
    self._tasks_failed = 0
    self._lock = threading.Lock()
    self._thread = threading.Thread(
        target=self._run, name=f"Agent-{agent_id}", daemon=True
    )

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def agent_id(self):
    """Unique identifier for this agent."""
    return self._agent_id

  @property
  def status(self):
    """Current `AgentStatus` of this agent."""
    with self._lock:
      return self._status

  @property
  def current_task(self):
    """The `Task` currently being executed, or ``None`` if idle."""
    with self._lock:
      return self._current_task

  @property
  def tasks_completed(self):
    """Number of tasks successfully completed by this agent."""
    with self._lock:
      return self._tasks_completed

  @property
  def tasks_failed(self):
    """Number of tasks that raised an exception on this agent."""
    with self._lock:
      return self._tasks_failed

  # ------------------------------------------------------------------
  # Lifecycle
  # ------------------------------------------------------------------

  def start(self):
    """Start the agent's background processing thread."""
    self._thread.start()

  def stop(self):
    """Signal the agent to stop after finishing any in-progress task."""
    with self._lock:
      self._status = AgentStatus.STOPPED

  def join(self, timeout=None):
    """Wait for the agent's thread to finish.

    Args:
        timeout: Maximum seconds to wait.
    """
    self._thread.join(timeout=timeout)

  # ------------------------------------------------------------------
  # Internal
  # ------------------------------------------------------------------

  def _run(self):
    """Main loop: pull tasks from the coordinator and execute them."""
    while True:
      with self._lock:
        if self._status == AgentStatus.STOPPED:
          break
      task = self._coordinator._get_next_task(timeout=0.1)  # pylint: disable=protected-access
      if task is None:
        continue
      with self._lock:
        self._status = AgentStatus.BUSY
        self._current_task = task
      try:
        task._execute()  # pylint: disable=protected-access
      finally:
        with self._lock:
          if task.status == TaskStatus.COMPLETED:
            self._tasks_completed += 1
          else:
            self._tasks_failed += 1
          self._current_task = None
          if self._status != AgentStatus.STOPPED:
            self._status = AgentStatus.IDLE
        self._coordinator._on_task_done(task)  # pylint: disable=protected-access

  def __repr__(self):
    return f"Agent(id={self._agent_id!r}, status={self.status.value})"

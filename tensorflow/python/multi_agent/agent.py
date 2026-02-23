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
"""Agent module for the Multi-Agent Task Coordination System.

Defines the `Agent` class which picks up `Task` objects and executes them,
updating each task's status and result fields accordingly.
"""

from tensorflow.python.multi_agent.task import TaskStatus


class Agent:
  """Processes `Task` objects assigned by a `TaskCoordinator`.

  Each agent maintains an ordered list of assigned tasks and processes
  them sequentially via :meth:`process_tasks`.

  Attributes:
    agent_id: Unique identifier string for this agent.
    tasks: List of `Task` objects currently assigned to this agent.
  """

  def __init__(self, agent_id):
    """Initializes a new Agent.

    Args:
      agent_id: Unique string identifier for this agent.
    """
    self.agent_id = agent_id
    self.tasks = []

  def assign_task(self, task):
    """Adds *task* to this agent's work queue.

    Args:
      task: A `Task` instance to be processed later.
    """
    self.tasks.append(task)

  def process_tasks(self):
    """Executes all pending tasks in assignment order.

    For each task in :attr:`tasks` whose status is ``PENDING``:

    * Sets status to ``IN_PROGRESS``.
    * Calls ``task.callable_fn()`` if provided; stores the return value
      in ``task.result``.
    * On success, sets status to ``COMPLETED``.
    * On any exception, records the exception in ``task.error`` and sets
      status to ``FAILED``.
    """
    for task in self.tasks:
      if task.status != TaskStatus.PENDING:
        continue
      task.status = TaskStatus.IN_PROGRESS
      try:
        if task.callable_fn is not None:
          task.result = task.callable_fn()
        task.status = TaskStatus.COMPLETED
      except Exception as exc:  # pylint: disable=broad-except
        task.error = exc
        task.status = TaskStatus.FAILED

  def __repr__(self):
    return f"Agent(id={self.agent_id!r}, tasks={len(self.tasks)})"

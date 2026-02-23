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
"""Coordinator module for the Multi-Agent Task Coordination System.

Implements `TaskCoordinator`, which manages a pool of `Agent` objects,
distributes `Task` objects among them in priority order, and reports on
overall task progress.
"""

from tensorflow.python.multi_agent.agent import Agent
from tensorflow.python.multi_agent.task import TaskStatus


class TaskCoordinator:
  """Manages agents and coordinates task assignment and execution.

  Tasks submitted via :meth:`submit_task` are queued internally.  A call to
  :meth:`assign_tasks` distributes all queued tasks across the available
  agents (highest-priority tasks first, round-robin across agents), and
  :meth:`run` triggers each agent to process its assigned work.

  Attributes:
    agents: Ordered list of registered `Agent` instances.
  """

  def __init__(self):
    """Initializes a new TaskCoordinator with an empty agent pool."""
    self.agents = []
    self._pending_tasks = []

  def add_agent(self, agent):
    """Registers *agent* with this coordinator.

    Args:
      agent: An `Agent` instance to add to the pool.

    Raises:
      ValueError: If an agent with the same ``agent_id`` is already
        registered.
    """
    for existing in self.agents:
      if existing.agent_id == agent.agent_id:
        raise ValueError(
            f"Agent with id {agent.agent_id!r} is already registered.")
    self.agents.append(agent)

  def submit_task(self, task):
    """Adds *task* to the coordinator's pending queue.

    Args:
      task: A `Task` instance to be scheduled.
    """
    self._pending_tasks.append(task)

  def assign_tasks(self):
    """Distributes all pending tasks to agents in priority order.

    Tasks are sorted by descending priority before being distributed
    round-robin across the registered agents.

    Raises:
      RuntimeError: If no agents have been registered.
    """
    if not self.agents:
      raise RuntimeError("No agents registered. Add agents before assigning tasks.")
    sorted_tasks = sorted(
        self._pending_tasks,
        key=lambda t: t.priority,
        reverse=True,
    )
    for index, task in enumerate(sorted_tasks):
      agent = self.agents[index % len(self.agents)]
      agent.assign_task(task)
    self._pending_tasks.clear()

  def run(self):
    """Instructs every registered agent to process its assigned tasks."""
    for agent in self.agents:
      agent.process_tasks()

  def get_task_status_summary(self):
    """Returns a summary of task statuses across all agents.

    Returns:
      A ``dict`` mapping each `TaskStatus` value string to an integer count
      of tasks in that state.
    """
    summary = {status.value: 0 for status in TaskStatus}
    for agent in self.agents:
      for task in agent.tasks:
        summary[task.status.value] += 1
    return summary

  def __repr__(self):
    return (
        f"TaskCoordinator(agents={len(self.agents)}, "
        f"pending={len(self._pending_tasks)})"
    )

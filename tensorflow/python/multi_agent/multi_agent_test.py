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
"""Tests for the Multi-Agent Task Coordination System."""

from tensorflow.python.multi_agent.agent import Agent
from tensorflow.python.multi_agent.coordinator import TaskCoordinator
from tensorflow.python.multi_agent.task import Task
from tensorflow.python.multi_agent.task import TaskPriority
from tensorflow.python.multi_agent.task import TaskStatus
from tensorflow.python.platform import test


class TaskTest(test.TestCase):
  """Tests for the Task class."""

  def test_default_status_is_pending(self):
    task = Task("t1", "do something")
    self.assertEqual(task.status, TaskStatus.PENDING)

  def test_default_priority_is_medium(self):
    task = Task("t1", "do something")
    self.assertEqual(task.priority, TaskPriority.MEDIUM)

  def test_explicit_priority_high(self):
    task = Task("t2", "urgent work", priority=TaskPriority.HIGH)
    self.assertEqual(task.priority, TaskPriority.HIGH)

  def test_result_and_error_initially_none(self):
    task = Task("t3", "work")
    self.assertIsNone(task.result)
    self.assertIsNone(task.error)

  def test_repr_contains_task_id(self):
    task = Task("my-task", "desc")
    self.assertIn("my-task", repr(task))

  def test_callable_fn_stored(self):
    fn = lambda: 42
    task = Task("t4", "compute", callable_fn=fn)
    self.assertIs(task.callable_fn, fn)


class AgentTest(test.TestCase):
  """Tests for the Agent class."""

  def test_agent_starts_with_no_tasks(self):
    agent = Agent("agent-1")
    self.assertEqual(agent.tasks, [])

  def test_assign_task_adds_to_list(self):
    agent = Agent("agent-1")
    task = Task("t1", "work")
    agent.assign_task(task)
    self.assertLen(agent.tasks, 1)

  def test_process_tasks_completes_task(self):
    agent = Agent("agent-1")
    task = Task("t1", "work", callable_fn=lambda: "done")
    agent.assign_task(task)
    agent.process_tasks()
    self.assertEqual(task.status, TaskStatus.COMPLETED)
    self.assertEqual(task.result, "done")

  def test_process_tasks_marks_failed_on_exception(self):
    def failing_fn():
      raise ValueError("boom")

    agent = Agent("agent-1")
    task = Task("t1", "work", callable_fn=failing_fn)
    agent.assign_task(task)
    agent.process_tasks()
    self.assertEqual(task.status, TaskStatus.FAILED)
    self.assertIsInstance(task.error, ValueError)

  def test_process_tasks_skips_non_pending(self):
    agent = Agent("agent-1")
    task = Task("t1", "already done")
    task.status = TaskStatus.COMPLETED
    agent.assign_task(task)
    agent.process_tasks()
    # Status should remain COMPLETED, not be re-processed.
    self.assertEqual(task.status, TaskStatus.COMPLETED)

  def test_process_task_without_callable(self):
    agent = Agent("agent-1")
    task = Task("t1", "no-op")
    agent.assign_task(task)
    agent.process_tasks()
    self.assertEqual(task.status, TaskStatus.COMPLETED)
    self.assertIsNone(task.result)

  def test_repr_contains_agent_id(self):
    agent = Agent("my-agent")
    self.assertIn("my-agent", repr(agent))


class TaskCoordinatorTest(test.TestCase):
  """Tests for the TaskCoordinator class."""

  def test_add_agent_registers_agent(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    self.assertLen(coordinator.agents, 1)

  def test_add_duplicate_agent_raises(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    with self.assertRaises(ValueError):
      coordinator.add_agent(Agent("a1"))

  def test_assign_tasks_without_agents_raises(self):
    coordinator = TaskCoordinator()
    coordinator.submit_task(Task("t1", "work"))
    with self.assertRaises(RuntimeError):
      coordinator.assign_tasks()

  def test_tasks_distributed_round_robin(self):
    coordinator = TaskCoordinator()
    a1 = Agent("a1")
    a2 = Agent("a2")
    coordinator.add_agent(a1)
    coordinator.add_agent(a2)
    for i in range(4):
      coordinator.submit_task(Task(f"t{i}", "work"))
    coordinator.assign_tasks()
    self.assertLen(a1.tasks, 2)
    self.assertLen(a2.tasks, 2)

  def test_higher_priority_tasks_assigned_first(self):
    coordinator = TaskCoordinator()
    agent = Agent("a1")
    coordinator.add_agent(agent)
    low = Task("low", "low", priority=TaskPriority.LOW)
    high = Task("high", "high", priority=TaskPriority.HIGH)
    coordinator.submit_task(low)
    coordinator.submit_task(high)
    coordinator.assign_tasks()
    # HIGH priority task should be first in the agent's task list.
    self.assertEqual(agent.tasks[0].task_id, "high")
    self.assertEqual(agent.tasks[1].task_id, "low")

  def test_run_executes_all_tasks(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    results = []
    for i in range(3):
      coordinator.submit_task(
          Task(f"t{i}", "work", callable_fn=lambda i=i: results.append(i)))
    coordinator.assign_tasks()
    coordinator.run()
    self.assertLen(results, 3)

  def test_get_task_status_summary(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    coordinator.submit_task(Task("t1", "work", callable_fn=lambda: None))
    coordinator.assign_tasks()
    coordinator.run()
    summary = coordinator.get_task_status_summary()
    self.assertEqual(summary[TaskStatus.COMPLETED.value], 1)
    self.assertEqual(summary[TaskStatus.PENDING.value], 0)

  def test_pending_tasks_cleared_after_assign(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    coordinator.submit_task(Task("t1", "work"))
    coordinator.assign_tasks()
    # Internal queue should be empty after assignment.
    self.assertEqual(coordinator._pending_tasks, [])  # pylint: disable=protected-access

  def test_repr_contains_agent_count(self):
    coordinator = TaskCoordinator()
    coordinator.add_agent(Agent("a1"))
    self.assertIn("agents=1", repr(coordinator))


if __name__ == "__main__":
  test.main()

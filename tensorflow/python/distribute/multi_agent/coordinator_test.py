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

import time
import unittest

from tensorflow.python.distribute.multi_agent.agent import Agent
from tensorflow.python.distribute.multi_agent.agent import AgentStatus
from tensorflow.python.distribute.multi_agent.coordinator import TaskCoordinator
from tensorflow.python.distribute.multi_agent.task import Task
from tensorflow.python.distribute.multi_agent.task import TaskStatus


# ---------------------------------------------------------------------------
# Task tests
# ---------------------------------------------------------------------------

class TaskTest(unittest.TestCase):

  def test_initial_status_is_pending(self):
    task = Task(fn=lambda: None)
    self.assertEqual(task.status, TaskStatus.PENDING)

  def test_execute_marks_completed(self):
    task = Task(fn=lambda x: x * 2, args=(21,))
    task._execute()
    self.assertEqual(task.status, TaskStatus.COMPLETED)
    self.assertEqual(task.result, 42)

  def test_execute_with_kwargs(self):
    task = Task(fn=lambda x, y: x + y, kwargs={"x": 3, "y": 4})
    task._execute()
    self.assertEqual(task.result, 7)

  def test_failed_task_captures_exception(self):
    def boom():
      raise ValueError("oops")

    task = Task(fn=boom)
    task._execute()
    self.assertEqual(task.status, TaskStatus.FAILED)
    self.assertIsInstance(task.exception, ValueError)
    self.assertIsNone(task.result)

  def test_cancel_pending_task(self):
    task = Task(fn=lambda: None)
    cancelled = task._mark_cancelled()
    self.assertTrue(cancelled)
    self.assertEqual(task.status, TaskStatus.CANCELLED)

  def test_cancel_running_task_returns_false(self):
    task = Task(fn=lambda: None)
    task._mark_running()
    cancelled = task._mark_cancelled()
    self.assertFalse(cancelled)
    self.assertEqual(task.status, TaskStatus.RUNNING)

  def test_wait_returns_true_after_completion(self):
    task = Task(fn=lambda: None)
    task._execute()
    result = task.wait(timeout=1.0)
    self.assertTrue(result)

  def test_task_id_is_unique_by_default(self):
    t1 = Task(fn=lambda: None)
    t2 = Task(fn=lambda: None)
    self.assertNotEqual(t1.task_id, t2.task_id)

  def test_custom_task_id(self):
    task = Task(fn=lambda: None, task_id="my-task-123")
    self.assertEqual(task.task_id, "my-task-123")

  def test_priority_ordering(self):
    low = Task(fn=lambda: None, priority=0)
    high = Task(fn=lambda: None, priority=10)
    # lower priority task should be considered "greater" in heap terms
    self.assertLess(high, low)

  def test_timestamps(self):
    task = Task(fn=time.sleep, args=(0,))
    self.assertIsNotNone(task.created_at)
    self.assertIsNone(task.started_at)
    task._execute()
    self.assertIsNotNone(task.started_at)
    self.assertIsNotNone(task.completed_at)

  def test_repr(self):
    task = Task(fn=lambda: None, priority=3, task_id="t1")
    self.assertIn("t1", repr(task))
    self.assertIn("3", repr(task))


# ---------------------------------------------------------------------------
# TaskCoordinator tests
# ---------------------------------------------------------------------------

class TaskCoordinatorTest(unittest.TestCase):

  def _make_coordinator(self, num_agents=2):
    coord = TaskCoordinator(num_agents=num_agents)
    coord.start()
    return coord

  def tearDown(self):
    super().tearDown()

  def test_invalid_num_agents_raises(self):
    with self.assertRaises(ValueError):
      TaskCoordinator(num_agents=0)

  def test_schedule_and_join(self):
    coord = self._make_coordinator(num_agents=2)
    results = []

    def fn(x):
      return x * x

    tasks = [coord.schedule(fn, args=(i,)) for i in range(5)]
    joined = coord.join(timeout=5.0)
    coord.stop()
    self.assertTrue(joined)
    for i, task in enumerate(tasks):
      self.assertEqual(task.status, TaskStatus.COMPLETED)
      self.assertEqual(task.result, i * i)

  def test_task_exception_does_not_crash_coordinator(self):
    coord = self._make_coordinator(num_agents=1)

    def bad():
      raise RuntimeError("failure")

    task = coord.schedule(bad)
    joined = coord.join(timeout=5.0)
    coord.stop()
    self.assertTrue(joined)
    self.assertEqual(task.status, TaskStatus.FAILED)
    self.assertIsInstance(task.exception, RuntimeError)

  def test_cancel_pending_task(self):
    coord = TaskCoordinator(num_agents=1)
    # Do NOT start yet so the task stays pending
    task = coord.schedule(lambda: None)
    cancelled = coord.cancel(task)
    self.assertTrue(cancelled)
    self.assertEqual(task.status, TaskStatus.CANCELLED)
    coord.start()
    coord.join(timeout=2.0)
    coord.stop()

  def test_priority_ordering(self):
    """Higher-priority tasks should be executed before lower-priority ones."""
    order = []
    lock_event = __import__("threading").Event()

    def blocker():
      # Keep the single agent busy until we release it
      lock_event.wait()

    def record(label):
      order.append(label)

    coord = TaskCoordinator(num_agents=1)
    coord.start()

    # Fill the agent with a blocking task first
    coord.schedule(blocker, priority=0)
    time.sleep(0.05)  # let the agent pick up the blocker

    # Enqueue low-priority then high-priority
    coord.schedule(record, args=("low",), priority=1)
    coord.schedule(record, args=("high",), priority=10)

    # Release the blocker
    lock_event.set()
    coord.join(timeout=5.0)
    coord.stop()

    self.assertEqual(order, ["high", "low"])

  def test_get_stats(self):
    coord = self._make_coordinator(num_agents=3)
    stats = coord.get_stats()
    self.assertEqual(stats["num_agents"], 3)
    self.assertIn("total_scheduled", stats)
    self.assertIn("total_completed", stats)

    tasks = [coord.schedule(lambda: None) for _ in range(6)]
    coord.join(timeout=5.0)
    coord.stop()

    stats = coord.get_stats()
    self.assertEqual(stats["total_scheduled"], 6)
    self.assertEqual(stats["total_completed"], 6)
    self.assertEqual(stats["total_failed"], 0)

  def test_stop_prevents_new_schedules(self):
    coord = self._make_coordinator(num_agents=1)
    coord.stop()
    with self.assertRaises(RuntimeError):
      coord.schedule(lambda: None)

  def test_agents_property(self):
    coord = self._make_coordinator(num_agents=3)
    agents = coord.agents
    self.assertEqual(len(agents), 3)
    coord.stop()

  def test_multiple_agents_parallel_execution(self):
    """Tasks should run in parallel across multiple agents."""
    started = __import__("threading").Barrier(3)
    results = []

    def slow_fn(val):
      started.wait(timeout=5.0)
      return val

    coord = TaskCoordinator(num_agents=3)
    coord.start()
    tasks = [coord.schedule(slow_fn, args=(i,)) for i in range(3)]
    coord.join(timeout=10.0)
    coord.stop()

    for i, task in enumerate(tasks):
      self.assertEqual(task.status, TaskStatus.COMPLETED)

  def test_join_returns_true_when_no_tasks(self):
    coord = self._make_coordinator(num_agents=2)
    result = coord.join(timeout=1.0)
    self.assertTrue(result)
    coord.stop()


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class AgentTest(unittest.TestCase):

  def test_initial_status_idle(self):
    coord = TaskCoordinator(num_agents=1)
    agent = coord.agents[0]
    self.assertEqual(agent.status, AgentStatus.IDLE)

  def test_repr(self):
    coord = TaskCoordinator(num_agents=1)
    agent = coord.agents[0]
    self.assertIn("Agent", repr(agent))
    self.assertIn("idle", repr(agent))

  def test_agent_counts_completed_tasks(self):
    coord = TaskCoordinator(num_agents=1)
    coord.start()
    for _ in range(5):
      coord.schedule(lambda: None)
    coord.join(timeout=5.0)
    coord.stop()
    total_completed = sum(a.tasks_completed for a in coord.agents)
    self.assertEqual(total_completed, 5)

  def test_agent_counts_failed_tasks(self):
    coord = TaskCoordinator(num_agents=1)
    coord.start()
    coord.schedule(lambda: (_ for _ in ()).throw(ValueError("x")))
    coord.join(timeout=5.0)
    coord.stop()
    total_failed = sum(a.tasks_failed for a in coord.agents)
    self.assertEqual(total_failed, 1)


if __name__ == "__main__":
  unittest.main()

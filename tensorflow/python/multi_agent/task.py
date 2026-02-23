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
"""Task module for the Multi-Agent Task Coordination System.

Defines the `Task` class which encapsulates work items with status tracking
and priority levels for use with `Agent` and `TaskCoordinator`.
"""

import enum


class TaskStatus(enum.Enum):
  """Enumeration of possible task lifecycle states."""
  PENDING = "pending"
  IN_PROGRESS = "in_progress"
  COMPLETED = "completed"
  FAILED = "failed"


class TaskPriority(enum.IntEnum):
  """Numeric priority levels for tasks (higher value = higher priority)."""
  LOW = 1
  MEDIUM = 2
  HIGH = 3
  CRITICAL = 4


class Task:
  """Represents a unit of work to be executed by an `Agent`.

  Attributes:
    task_id: Unique identifier string for this task.
    description: Human-readable description of the work to be done.
    priority: A `TaskPriority` value controlling scheduling order.
    status: Current `TaskStatus` of the task.
    result: The value returned by the callable once the task completes.
    error: Exception instance if the task failed, otherwise ``None``.
  """

  def __init__(self, task_id, description, priority=TaskPriority.MEDIUM,
               callable_fn=None):
    """Initializes a new Task.

    Args:
      task_id: Unique string identifier.
      description: Short human-readable description.
      priority: `TaskPriority` (default: MEDIUM).
      callable_fn: Optional zero-argument callable to execute when the task
        is processed.  If ``None`` the task completes with ``result=None``.
    """
    self.task_id = task_id
    self.description = description
    self.priority = TaskPriority(priority)
    self.callable_fn = callable_fn
    self.status = TaskStatus.PENDING
    self.result = None
    self.error = None

  def __repr__(self):
    return (
        f"Task(id={self.task_id!r}, priority={self.priority.name}, "
        f"status={self.status.value})"
    )

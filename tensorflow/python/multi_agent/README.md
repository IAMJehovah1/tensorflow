# Multi-Agent Task Coordination System

This package (`tensorflow.python.multi_agent`) provides a lightweight
framework for coordinating work across multiple agents.  It is designed to
be standalone and dependency-free (beyond the Python standard library) so
that it can be integrated into any module of the TensorFlow codebase without
disruption.

---

## Package structure

```
tensorflow/python/multi_agent/
├── __init__.py       – public re-exports
├── task.py           – Task, TaskStatus, TaskPriority
├── agent.py          – Agent
├── coordinator.py    – TaskCoordinator
└── multi_agent_test.py
```

---

## Quick-start

```python
from tensorflow.python.multi_agent import (
    Task, TaskPriority, Agent, TaskCoordinator
)

# 1. Create a coordinator and register agents.
coordinator = TaskCoordinator()
coordinator.add_agent(Agent("worker-0"))
coordinator.add_agent(Agent("worker-1"))

# 2. Submit tasks (optionally with a callable and a priority).
coordinator.submit_task(Task("t1", "low priority job",
                             priority=TaskPriority.LOW,
                             callable_fn=lambda: print("low")))
coordinator.submit_task(Task("t2", "critical job",
                             priority=TaskPriority.CRITICAL,
                             callable_fn=lambda: print("critical")))

# 3. Distribute tasks to agents (highest-priority first, round-robin).
coordinator.assign_tasks()

# 4. Execute all assigned tasks.
coordinator.run()

# 5. Inspect results.
print(coordinator.get_task_status_summary())
# {'pending': 0, 'in_progress': 0, 'completed': 2, 'failed': 0}
```

---

## API reference

### `Task`

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | Unique identifier. |
| `description` | `str` | Human-readable description. |
| `priority` | `TaskPriority` | Scheduling priority. |
| `callable_fn` | `callable \| None` | Zero-argument function to execute. |
| `status` | `TaskStatus` | Current lifecycle state. |
| `result` | `Any` | Return value of `callable_fn` after completion. |
| `error` | `Exception \| None` | Exception captured on failure. |

### `TaskStatus` (enum)

`PENDING` → `IN_PROGRESS` → `COMPLETED` / `FAILED`

### `TaskPriority` (IntEnum)

`LOW=1`, `MEDIUM=2`, `HIGH=3`, `CRITICAL=4`

### `Agent`

| Method | Description |
|--------|-------------|
| `assign_task(task)` | Add a task to the agent's queue. |
| `process_tasks()` | Execute all pending tasks sequentially. |

### `TaskCoordinator`

| Method | Description |
|--------|-------------|
| `add_agent(agent)` | Register an agent. |
| `submit_task(task)` | Add a task to the pending queue. |
| `assign_tasks()` | Distribute pending tasks to agents (priority order, round-robin). |
| `run()` | Instruct all agents to process their tasks. |
| `get_task_status_summary()` | Return a `dict` with counts per `TaskStatus`. |

---

## Running the tests

```bash
python -m pytest tensorflow/python/multi_agent/multi_agent_test.py -v
# or via the TensorFlow test runner:
python tensorflow/python/multi_agent/multi_agent_test.py
```

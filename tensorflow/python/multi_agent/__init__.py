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
"""Multi-Agent Task Coordination System.

Public API::

    from tensorflow.python.multi_agent import Task, TaskPriority, TaskStatus
    from tensorflow.python.multi_agent import Agent
    from tensorflow.python.multi_agent import TaskCoordinator
    from tensorflow.python.multi_agent import (
        AuditResult, ConfidenceLevel, FeedbackRating, SelfEvaluationAuditor
    )
"""

from tensorflow.python.multi_agent.agent import Agent
from tensorflow.python.multi_agent.auditor import AuditResult
from tensorflow.python.multi_agent.auditor import ConfidenceLevel
from tensorflow.python.multi_agent.auditor import FeedbackRating
from tensorflow.python.multi_agent.auditor import SelfEvaluationAuditor
from tensorflow.python.multi_agent.coordinator import TaskCoordinator
from tensorflow.python.multi_agent.task import Task
from tensorflow.python.multi_agent.task import TaskPriority
from tensorflow.python.multi_agent.task import TaskStatus

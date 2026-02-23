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
"""AI Self-Evaluation Auditor module.

Provides `SelfEvaluationAuditor`, which analyzes AI-generated responses,
assigns a confidence score, detects potential quality issues, generates an
explanation of its reasoning, and integrates user feedback to improve future
evaluations.

Example usage::

    from tensorflow.python.multi_agent.auditor import (
        SelfEvaluationAuditor, FeedbackRating
    )

    auditor = SelfEvaluationAuditor()
    result = auditor.evaluate("The answer is 42.", query="What is the answer?")
    print(result.confidence_score)   # e.g. 0.85
    print(result.explanation)

    auditor.record_feedback(result.audit_id, FeedbackRating.HELPFUL)
"""

import enum
import uuid


# ---------------------------------------------------------------------------
# Public enumerations
# ---------------------------------------------------------------------------

class ConfidenceLevel(enum.Enum):
  """Qualitative confidence band derived from a numeric score.

  Bands:
    HIGH   – score ≥ 0.80
    MEDIUM – 0.50 ≤ score < 0.80
    LOW    – score < 0.50
  """
  HIGH = "high"
  MEDIUM = "medium"
  LOW = "low"


class FeedbackRating(enum.Enum):
  """User-supplied quality rating for an audited response."""
  HELPFUL = "helpful"
  INCOMPLETE = "incomplete"
  INACCURATE = "inaccurate"


# ---------------------------------------------------------------------------
# AuditResult data class
# ---------------------------------------------------------------------------

class AuditResult:
  """Holds the outcome of a single evaluation performed by `SelfEvaluationAuditor`.

  Attributes:
    audit_id: Unique identifier for this audit (UUID string).
    confidence_score: Float in ``[0.0, 1.0]`` representing estimated quality.
    confidence_level: `ConfidenceLevel` band derived from the score.
    flags: List of human-readable strings describing detected issues.
    explanation: Narrative summary of how the score was reached.
    requires_human_review: ``True`` when the auditor recommends expert review.
  """

  def __init__(self, audit_id, confidence_score, flags, explanation):
    self.audit_id = audit_id
    self.confidence_score = confidence_score
    self.confidence_level = _score_to_level(confidence_score)
    self.flags = flags
    self.explanation = explanation
    self.requires_human_review = self.confidence_level == ConfidenceLevel.LOW

  def __repr__(self):
    return (
        f"AuditResult(id={self.audit_id!r}, "
        f"score={self.confidence_score:.2f}, "
        f"level={self.confidence_level.value})"
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _score_to_level(score):
  """Map a numeric score to a `ConfidenceLevel` band."""
  if score >= 0.80:
    return ConfidenceLevel.HIGH
  if score >= 0.50:
    return ConfidenceLevel.MEDIUM
  return ConfidenceLevel.LOW


# Uncertainty markers that reduce confidence when found in a response.
_UNCERTAINTY_MARKERS = frozenset([
    "maybe", "perhaps", "possibly", "i'm not sure", "i am not sure",
    "i don't know", "i do not know", "uncertain", "unclear", "unsure",
    "not certain", "might be", "could be",
])

# Minimum and maximum "healthy" response lengths (in characters).
_MIN_HEALTHY_LEN = 20
_MAX_HEALTHY_LEN = 5000


# ---------------------------------------------------------------------------
# SelfEvaluationAuditor
# ---------------------------------------------------------------------------

class SelfEvaluationAuditor:
  """Evaluates AI-generated responses and integrates user feedback.

  The auditor applies a set of lightweight heuristics to estimate response
  quality without requiring external dependencies:

  * **Completeness** – penalizes empty or very short responses.
  * **Verbosity** – penalizes excessively long responses.
  * **Uncertainty markers** – detects hedging language that signals low
    confidence.
  * **Query coverage** – when a *query* is supplied, checks that at least
    one query keyword appears in the response.
  * **Feedback adjustment** – recorded feedback shifts the base score for
    similar future evaluations.

  Attributes:
    _feedback_history: List of ``(audit_id, FeedbackRating, correction)``
      tuples accumulated via :meth:`record_feedback`.
    _feedback_score_adjustment: Running adjustment applied to the base score
      derived from aggregate feedback signal.
  """

  def __init__(self):
    self._feedback_history = []
    self._feedback_score_adjustment = 0.0

  # ------------------------------------------------------------------
  # Public methods
  # ------------------------------------------------------------------

  def evaluate(self, response, query=None):
    """Analyse *response* and return an :class:`AuditResult`.

    Args:
      response: String containing the AI-generated response to evaluate.
      query: Optional string with the original user query.  When provided,
        query-coverage checks are performed.

    Returns:
      An :class:`AuditResult` instance.
    """
    if not isinstance(response, str):
      raise TypeError(
          f"response must be a str, got {type(response).__name__!r}")

    flags = []
    score_components = []

    # -- Completeness check ------------------------------------------
    if not response.strip():
      flags.append("Response is empty.")
      explanation = self._build_explanation(0.0, flags)
      return AuditResult(
          audit_id=str(uuid.uuid4()),
          confidence_score=0.0,
          flags=flags,
          explanation=explanation,
      )
    elif len(response) < _MIN_HEALTHY_LEN:
      flags.append("Response is very short and may be incomplete.")
      score_components.append(0.4)
    else:
      score_components.append(1.0)

    # -- Verbosity check ---------------------------------------------
    if len(response) > _MAX_HEALTHY_LEN:
      flags.append("Response is excessively long; consider condensing.")
      score_components.append(0.7)
    else:
      score_components.append(1.0)

    # -- Uncertainty markers -----------------------------------------
    lower_response = response.lower()
    found_markers = [
        m for m in _UNCERTAINTY_MARKERS if m in lower_response
    ]
    if found_markers:
      flags.append(
          f"Uncertainty language detected: {', '.join(sorted(found_markers))}.")
      score_components.append(0.6)
    else:
      score_components.append(1.0)

    # -- Query coverage ----------------------------------------------
    if query and query.strip():
      keywords = [
          w.lower().strip("?.,!") for w in query.split() if len(w) > 3
      ]
      if keywords:
        covered = sum(1 for kw in keywords if kw in lower_response)
        coverage_ratio = covered / len(keywords)
        if coverage_ratio < 0.5:
          flags.append(
              "Response may not fully address the query (low keyword coverage).")
        score_components.append(max(0.3, coverage_ratio))

    # -- Aggregate score ---------------------------------------------
    base_score = sum(score_components) / len(score_components)
    adjusted_score = min(1.0, max(0.0,
                                  base_score + self._feedback_score_adjustment))

    # -- Explanation -------------------------------------------------
    explanation = self._build_explanation(adjusted_score, flags)

    audit_id = str(uuid.uuid4())
    return AuditResult(
        audit_id=audit_id,
        confidence_score=adjusted_score,
        flags=flags,
        explanation=explanation,
    )

  def record_feedback(self, audit_id, rating, correction=None):
    """Record user feedback for a previously evaluated response.

    Feedback is stored in :attr:`_feedback_history` and the aggregate
    score adjustment is updated so that future evaluations benefit from
    the signal.

    Args:
      audit_id: String ID returned by a prior :meth:`evaluate` call.
      rating: A :class:`FeedbackRating` value.
      correction: Optional free-text correction provided by the user.

    Raises:
      TypeError: If *rating* is not a :class:`FeedbackRating` instance.
    """
    if not isinstance(rating, FeedbackRating):
      raise TypeError(
          f"rating must be a FeedbackRating, got {type(rating).__name__!r}")
    self._feedback_history.append((audit_id, rating, correction))
    self._update_score_adjustment()

  def get_feedback_summary(self):
    """Return a summary of all recorded feedback.

    Returns:
      A ``dict`` mapping each :class:`FeedbackRating` value string to the
      count of times that rating has been recorded.  Also includes a
      ``"total"`` key with the overall feedback count.
    """
    summary = {rating.value: 0 for rating in FeedbackRating}
    for _, rating, _ in self._feedback_history:
      summary[rating.value] += 1
    summary["total"] = len(self._feedback_history)
    return summary

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  def _update_score_adjustment(self):
    """Recalculate the feedback-driven score adjustment.

    HELPFUL ratings contribute +0.02 each, INACCURATE ratings contribute
    −0.05 each, and INCOMPLETE ratings contribute −0.02 each.  The
    cumulative adjustment is clamped to ``[−0.20, +0.20]``.
    """
    adjustment = 0.0
    for _, rating, _ in self._feedback_history:
      if rating == FeedbackRating.HELPFUL:
        adjustment += 0.02
      elif rating == FeedbackRating.INACCURATE:
        adjustment -= 0.05
      elif rating == FeedbackRating.INCOMPLETE:
        adjustment -= 0.02
    self._feedback_score_adjustment = max(-0.20, min(0.20, adjustment))

  def _build_explanation(self, score, flags):
    """Compose a human-readable explanation string."""
    level = _score_to_level(score)
    pct = int(round(score * 100))
    parts = [f"Confidence: {pct}%."]
    if flags:
      parts.append("Issues detected: " + " ".join(flags))
    else:
      parts.append("No quality issues detected.")
    if level == ConfidenceLevel.LOW:
      parts.append(
          "This response has low confidence and may require expert review.")
    elif level == ConfidenceLevel.MEDIUM:
      parts.append("Response quality is acceptable but could be improved.")
    else:
      parts.append("Response quality appears high.")
    if self._feedback_score_adjustment != 0.0:
      direction = "up" if self._feedback_score_adjustment > 0 else "down"
      parts.append(
          f"Score adjusted {direction} based on prior user feedback.")
    return " ".join(parts)

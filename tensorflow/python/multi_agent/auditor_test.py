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
"""Tests for the AI Self-Evaluation Auditor."""

from tensorflow.python.multi_agent.auditor import AuditResult
from tensorflow.python.multi_agent.auditor import ConfidenceLevel
from tensorflow.python.multi_agent.auditor import FeedbackRating
from tensorflow.python.multi_agent.auditor import SelfEvaluationAuditor
from tensorflow.python.platform import test


class AuditResultTest(test.TestCase):
  """Tests for the AuditResult data class."""

  def _make_result(self, score, flags=None, explanation=""):
    return AuditResult(
        audit_id="test-id",
        confidence_score=score,
        flags=flags or [],
        explanation=explanation,
    )

  def test_high_confidence_level(self):
    result = self._make_result(0.90)
    self.assertEqual(result.confidence_level, ConfidenceLevel.HIGH)
    self.assertFalse(result.requires_human_review)

  def test_medium_confidence_level(self):
    result = self._make_result(0.65)
    self.assertEqual(result.confidence_level, ConfidenceLevel.MEDIUM)
    self.assertFalse(result.requires_human_review)

  def test_low_confidence_level_requires_review(self):
    result = self._make_result(0.30)
    self.assertEqual(result.confidence_level, ConfidenceLevel.LOW)
    self.assertTrue(result.requires_human_review)

  def test_boundary_score_0_80_is_high(self):
    result = self._make_result(0.80)
    self.assertEqual(result.confidence_level, ConfidenceLevel.HIGH)

  def test_boundary_score_0_50_is_medium(self):
    result = self._make_result(0.50)
    self.assertEqual(result.confidence_level, ConfidenceLevel.MEDIUM)

  def test_boundary_score_below_0_50_is_low(self):
    result = self._make_result(0.49)
    self.assertEqual(result.confidence_level, ConfidenceLevel.LOW)

  def test_repr_contains_audit_id(self):
    result = self._make_result(0.75)
    self.assertIn("test-id", repr(result))

  def test_flags_stored(self):
    flags = ["Response is empty.", "Uncertainty language detected."]
    result = self._make_result(0.20, flags=flags)
    self.assertEqual(result.flags, flags)


class SelfEvaluationAuditorEvaluateTest(test.TestCase):
  """Tests for SelfEvaluationAuditor.evaluate()."""

  def setUp(self):
    super().setUp()
    self.auditor = SelfEvaluationAuditor()

  def test_empty_response_low_confidence(self):
    result = self.auditor.evaluate("")
    self.assertEqual(result.confidence_level, ConfidenceLevel.LOW)
    self.assertTrue(any("empty" in f.lower() for f in result.flags))

  def test_whitespace_only_response_low_confidence(self):
    result = self.auditor.evaluate("   \n\t  ")
    self.assertEqual(result.confidence_level, ConfidenceLevel.LOW)

  def test_very_short_response_flagged(self):
    result = self.auditor.evaluate("Yes.")
    self.assertTrue(any("short" in f.lower() for f in result.flags))

  def test_good_response_high_confidence(self):
    response = (
        "The capital of France is Paris. It is a major European city "
        "and the seat of the French government, located in the north "
        "of the country along the Seine river."
    )
    result = self.auditor.evaluate(response)
    self.assertGreaterEqual(result.confidence_score, 0.80)
    self.assertEqual(result.confidence_level, ConfidenceLevel.HIGH)

  def test_uncertainty_markers_reduce_confidence(self):
    certain = "The boiling point of water is 100 degrees Celsius at sea level."
    uncertain = "Maybe the boiling point is perhaps around 100 degrees."
    result_certain = self.auditor.evaluate(certain)
    result_uncertain = self.auditor.evaluate(uncertain)
    self.assertGreater(result_certain.confidence_score,
                       result_uncertain.confidence_score)
    self.assertTrue(any("uncertainty" in f.lower()
                        for f in result_uncertain.flags))

  def test_excessively_long_response_flagged(self):
    long_response = "word " * 1500  # > 5000 chars
    result = self.auditor.evaluate(long_response)
    self.assertTrue(any("long" in f.lower() for f in result.flags))

  def test_query_coverage_low_penalizes_score(self):
    query = "What is the population of Tokyo and its main industries?"
    response = "The weather is nice today."
    result = self.auditor.evaluate(response, query=query)
    self.assertTrue(any("query" in f.lower() for f in result.flags))

  def test_query_coverage_high_does_not_add_flag(self):
    query = "What is the population of Tokyo?"
    response = (
        "The population of Tokyo is approximately 14 million people in the "
        "city proper, making it one of the most populous cities in the world."
    )
    result = self.auditor.evaluate(response, query=query)
    self.assertFalse(any("query" in f.lower() for f in result.flags))

  def test_returns_audit_result_instance(self):
    result = self.auditor.evaluate("Some response text.")
    self.assertIsInstance(result, AuditResult)

  def test_each_evaluation_has_unique_audit_id(self):
    r1 = self.auditor.evaluate("Response one.")
    r2 = self.auditor.evaluate("Response two.")
    self.assertNotEqual(r1.audit_id, r2.audit_id)

  def test_non_string_response_raises_type_error(self):
    with self.assertRaises(TypeError):
      self.auditor.evaluate(42)

  def test_explanation_contains_confidence_percentage(self):
    result = self.auditor.evaluate(
        "The mitochondria is the powerhouse of the cell.")
    self.assertIn("Confidence:", result.explanation)
    self.assertIn("%", result.explanation)

  def test_low_confidence_explanation_mentions_review(self):
    result = self.auditor.evaluate("")
    self.assertIn("review", result.explanation.lower())

  def test_evaluate_without_query(self):
    result = self.auditor.evaluate(
        "Paris is the capital of France.", query=None)
    self.assertIsNotNone(result)

  def test_score_clamped_to_0_1(self):
    result = self.auditor.evaluate("x" * 6000)  # triggers verbosity flag
    self.assertGreaterEqual(result.confidence_score, 0.0)
    self.assertLessEqual(result.confidence_score, 1.0)


class SelfEvaluationAuditorFeedbackTest(test.TestCase):
  """Tests for feedback recording and score adjustment."""

  def setUp(self):
    super().setUp()
    self.auditor = SelfEvaluationAuditor()

  def test_record_feedback_stores_entry(self):
    result = self.auditor.evaluate("Some response.")
    self.auditor.record_feedback(result.audit_id, FeedbackRating.HELPFUL)
    summary = self.auditor.get_feedback_summary()
    self.assertEqual(summary[FeedbackRating.HELPFUL.value], 1)
    self.assertEqual(summary["total"], 1)

  def test_multiple_feedback_entries(self):
    r1 = self.auditor.evaluate("Response 1.")
    r2 = self.auditor.evaluate("Response 2.")
    self.auditor.record_feedback(r1.audit_id, FeedbackRating.HELPFUL)
    self.auditor.record_feedback(r2.audit_id, FeedbackRating.INACCURATE)
    summary = self.auditor.get_feedback_summary()
    self.assertEqual(summary[FeedbackRating.HELPFUL.value], 1)
    self.assertEqual(summary[FeedbackRating.INACCURATE.value], 1)
    self.assertEqual(summary["total"], 2)

  def test_helpful_feedback_increases_future_scores(self):
    response = "A detailed and accurate response about machine learning."
    baseline = self.auditor.evaluate(response)
    # Record many HELPFUL ratings to push adjustment up.
    for _ in range(5):
      self.auditor.record_feedback("dummy-id", FeedbackRating.HELPFUL)
    adjusted = self.auditor.evaluate(response)
    self.assertGreaterEqual(adjusted.confidence_score,
                            baseline.confidence_score)

  def test_inaccurate_feedback_decreases_future_scores(self):
    response = "A detailed and accurate response about machine learning."
    baseline = self.auditor.evaluate(response)
    # Record many INACCURATE ratings to push adjustment down.
    for _ in range(5):
      self.auditor.record_feedback("dummy-id", FeedbackRating.INACCURATE)
    adjusted = self.auditor.evaluate(response)
    self.assertLessEqual(adjusted.confidence_score, baseline.confidence_score)

  def test_invalid_rating_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      self.auditor.record_feedback("some-id", "helpful")

  def test_feedback_summary_all_ratings_present(self):
    summary = self.auditor.get_feedback_summary()
    for rating in FeedbackRating:
      self.assertIn(rating.value, summary)
    self.assertIn("total", summary)

  def test_feedback_summary_initial_all_zero(self):
    summary = self.auditor.get_feedback_summary()
    self.assertEqual(summary["total"], 0)
    for rating in FeedbackRating:
      self.assertEqual(summary[rating.value], 0)

  def test_score_adjustment_clamped(self):
    # Record 50 INACCURATE ratings – adjustment should not drop below -0.20.
    for _ in range(50):
      self.auditor.record_feedback("dummy", FeedbackRating.INACCURATE)
    self.assertGreaterEqual(
        self.auditor._feedback_score_adjustment, -0.20)  # pylint: disable=protected-access

  def test_correction_text_stored_in_history(self):
    result = self.auditor.evaluate("Some response.")
    self.auditor.record_feedback(
        result.audit_id, FeedbackRating.INCOMPLETE,
        correction="Please include more detail.")
    _, _, correction = self.auditor._feedback_history[-1]  # pylint: disable=protected-access
    self.assertEqual(correction, "Please include more detail.")

  def test_feedback_adjustment_mentioned_in_explanation(self):
    response = "A sufficiently long response about neural networks."
    for _ in range(5):
      self.auditor.record_feedback("dummy", FeedbackRating.HELPFUL)
    result = self.auditor.evaluate(response)
    self.assertIn("feedback", result.explanation.lower())


if __name__ == "__main__":
  test.main()

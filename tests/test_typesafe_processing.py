import unittest

from benchmark.processing_models import (
    DEFAULT_BASE_PROMPT,
    estimate_usage_cost,
    get_model_provider,
)
from benchmark.processing_prompts import build_system_prompt
from benchmark.processing_runner import (
    _build_typesafe_batch_request,
    _extract_batch_predictions,
    _extract_typesafe_scores,
)


class TypeSafeProcessingTests(unittest.TestCase):
    def setUp(self):
        self.records_a = {
            "a1": {"name": "Acme Widget", "sku": "W-1"},
            "a2": {"name": "Other Product", "sku": "P-2"},
        }
        self.records_b = {
            "b1": {"name": "ACME Widget", "sku": "W1"},
            "b2": {"name": "Different Widget", "sku": "W-9"},
        }
        self.system_prompt = build_system_prompt(DEFAULT_BASE_PROMPT, "product")

    def _build(self, batch):
        return _build_typesafe_batch_request(
            batch,
            self.records_a,
            self.records_b,
            "labels_inline",
            "jev-1.13.0",
            self.system_prompt,
        )

    def test_model_provider_and_input_only_price(self):
        self.assertEqual(get_model_provider("jev-1.13.0"), "typesafe")
        cost = estimate_usage_cost("jev-1.13.0", 1_000_000, 1_000_000)
        self.assertEqual(cost["input_cost_usd"], 0.042)
        self.assertEqual(cost["output_cost_usd"], 0.0)
        self.assertEqual(cost["total_cost_usd"], 0.042)

    def test_pairwise_request_uses_one_noul_question(self):
        bundle = self._build({"mode": "pairwise", "pairs": [("a1", "b1")]})
        payload = bundle["request"]["request_kwargs"]["json"]

        self.assertEqual(payload["model"], "jev-1.13.0")
        self.assertEqual(payload["questions"]["pair_0"]["type"], "noul")
        self.assertEqual(bundle["pair_lookup"], {"pair_0": ("a1", "b1")})
        self.assertEqual(set(payload["state"]), {"record_a", "record_b"})

    def test_multi_pair_request_maps_each_question_to_its_pair(self):
        pairs = [("a1", "b1"), ("a2", "b2")]
        bundle = self._build({"mode": "multi_pair", "pairs": pairs})
        payload = bundle["request"]["request_kwargs"]["json"]

        self.assertEqual(len(payload["questions"]), 2)
        self.assertEqual(bundle["pair_lookup"]["pair_1"], ("a2", "b2"))
        self.assertEqual(len(payload["state"]["candidate_pairs"]), 2)

    def test_one_to_many_request_shares_source_record(self):
        pairs = [("a1", "b1"), ("a1", "b2")]
        bundle = self._build(
            {"mode": "one_to_many", "id_a": "a1", "pairs": pairs}
        )
        payload = bundle["request"]["request_kwargs"]["json"]

        self.assertIn("source_record", payload["state"])
        self.assertEqual(len(payload["state"]["candidates"]), 2)
        self.assertEqual(len(payload["questions"]), 2)

    def test_noul_scores_threshold_at_one_half_and_remain_available(self):
        payload = {
            "answers": {
                "pair_0": {"type": "noul", "noul": 0.5},
                "pair_1": {"type": "noul", "noul": 0.49},
            }
        }
        lookup = {"pair_0": ("a1", "b1"), "pair_1": ("a1", "b2")}

        predicted = _extract_batch_predictions("one_to_many", payload, lookup, {})
        scores = _extract_typesafe_scores(payload, lookup)

        self.assertEqual(predicted, {("a1", "b1")})
        self.assertEqual(scores[("a1", "b1")], 0.5)
        self.assertEqual(scores[("a1", "b2")], 0.49)


if __name__ == "__main__":
    unittest.main()

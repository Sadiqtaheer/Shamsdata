"""Check temporal boundaries, one-class inference, and forecast selection isolation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from train import chronological_validation_split, positive_probability, probability_metrics, evaluate_forecast


class MLChecks(unittest.TestCase):
    def test_timestamps_stay_together(self):
        frame = pd.DataFrame({"date": [1, 1, 2, 2, 3, 3]})
        fit, holdout = chronological_validation_split(frame, "date", .5)
        self.assertLess(fit.date.max(), holdout.date.min())
        with self.assertRaises(ValueError):
            chronological_validation_split(frame.assign(date=1), "date")

    def test_single_class_probability(self):
        X = pd.DataFrame({"x": [0, 1, 2]})
        for label in (0, 1):
            model = ExtraTreesClassifier(n_estimators=2).fit(X, [label] * 3)
            np.testing.assert_array_equal(positive_probability(model, X), [label] * 3)
            self.assertEqual(probability_metrics([label] * 3, np.array([label] * 3))["brier_score"], 0)

    def test_selection_does_not_use_final_holdout(self):
        history = pd.Series(np.tile(np.arange(1, 8), 40), dtype=float)
        def poor_model(series, horizon):
            return None, np.full(horizon, 100.)
        with patch("train.recursive_forecast", side_effect=poor_model):
            selected, metrics = evaluate_forecast(history, 30)
            changed = history.copy()
            changed.iloc[-30:] = 100
            selected_changed, changed_metrics = evaluate_forecast(changed, 30)
        self.assertEqual(selected, "weekly_baseline")
        self.assertEqual(selected_changed, selected)
        self.assertEqual(metrics["selection_candidate_mae"], changed_metrics["selection_candidate_mae"])
        self.assertEqual(metrics["mae"], 0)
        self.assertGreater(changed_metrics["mae"], 0)


if __name__ == "__main__":
    unittest.main()

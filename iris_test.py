from sklearn.metrics import accuracy_score
from pathlib import Path
import pandas as pd
import unittest


METRICS = {
    "accuracy": accuracy_score,
}

def compute_metric(predicition_path: Path, test_path: Path, metric_func):
    pred = pd.read_csv(predicition_path).predictions.values
    test = pd.read_csv(test_path).target.values
    return metric_func(pred, test)


def compare_with_min_score(metric_value: float, min_score: float):
    return metric_value >= min_score


class MetricTest(unittest.TestCase):
    predictions = Path("data/predict.csv")
    test_path = Path("data/test.csv")
    metric = METRICS["accuracy"]
    min_score = 0.9

    def test_metric_greater_than_min_score(self):
        metric_value = compute_metric(
            MetricTest.predictions, 
            MetricTest.test_path, MetricTest.metric
        )
        ok = compare_with_min_score(metric_value, MetricTest.min_score)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
    
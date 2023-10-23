from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from pathlib import Path
import pandas as pd
import unittest

from iris_report import Report
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial


METRICS = {
    "accuracy": accuracy_score,
    "precision": partial(precision_score, average = "macro")
}

def compute_metric(predicition_path: Path, test_path: Path, metric_func):
    pred = pd.read_csv(predicition_path).predictions.values
    test = pd.read_csv(test_path).target.values
    return metric_func(pred, test)


def get_confusion_matrix(predicition_path: Path, test_path: Path):
    pred = pd.read_csv(predicition_path).predictions.values
    test = pd.read_csv(test_path).target.values
    
    cm = confusion_matrix(test, pred)
    df_cfm = pd.DataFrame(
        cm, index = ["0", "1", "2"], columns = ["0", "1", "2"])
    plt.figure(figsize = (10, 7))
    cfm_plot = sn.heatmap(df_cfm, annot=True)
    return cfm_plot.figure


def compare_with_min_score(metric_value: float, min_score: float):
    return metric_value >= min_score


class MetricTest(unittest.TestCase):
    report = Report(Path("data"))
    predictions = Path("data/predict.csv")
    test_path = Path("data/test.csv")
    metric = METRICS
    min_score = 0.9
    
    def test_metric_greater_than_min_score(self):
        for metric_name, metric_func in MetricTest.metric.items():
            metric_value = compute_metric(
                MetricTest.predictions, 
                MetricTest.test_path, metric_func
            )
            MetricTest.report.add(f"{metric_name} = {metric_value}", "metrics.txt")
            ok = compare_with_min_score(metric_value, MetricTest.min_score)
            self.assertTrue(ok)

        cm = get_confusion_matrix(MetricTest.predictions, MetricTest.test_path)
        MetricTest.report.add(cm, "confusion_matrix.png")


if __name__ == "__main__":
    unittest.main()
    
import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from decent_bench.benchmark import MetricResult
from decent_bench.metrics import metric_library as ml
from decent_bench.metrics import metric_utils
from decent_bench.metrics._metric import Metric


class _AlgorithmStub:
    def __init__(self, name: str) -> None:
        self.name = name


class _MetricStub(Metric):
    def __init__(self, description: str) -> None:
        super().__init__([np.average], fmt=".2e", y_log=False)
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    def get_data_from_trial(self, agents, problem, iteration):  # noqa: D102
        return [0.0]


class _CostStub:
    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def predict(self, x: object, data: list[object]) -> list[int]:  # noqa: ARG002
        return self.predictions


class _AgentViewStub:
    def __init__(self, *, is_server: bool, predictions: list[int]) -> None:
        self.is_server = is_server
        self.cost = _CostStub(predictions)
        self.x_history = {-1: np.array([0.0])}


class _ProblemStub:
    def __init__(self, labels: list[int]) -> None:
        self.test_data = [(np.array([index]), label) for index, label in enumerate(labels)]


def test_metric_result_to_dataframe_converts_table_and_plot_results() -> None:
    pd = pytest.importorskip("pandas")

    algorithm_a = _AlgorithmStub("A")
    algorithm_b = _AlgorithmStub("B")
    metric_one = _MetricStub("one")
    metric_two = _MetricStub("two")

    metrics_result = MetricResult(
        agent_metrics=None,
        table_metrics=[metric_one, metric_two],
        plot_metrics=[metric_one, metric_two],
        table_results={
            algorithm_a: {
                metric_one: {"avg": (1.0, 0.1), "mdn": (1.5, 0.2)},
            },
            algorithm_b: {
                metric_two: {"avg": (2.0, 0.3)},
            },
        },
        plot_results={
            algorithm_a: {
                metric_one: ([0.0, 1.0], [1.0, 2.0], [0.5, 1.5], [1.5, 2.5]),
            },
            algorithm_b: {
                metric_two: ([2.0], [3.0], [2.5], [3.5]),
            },
        },
    )

    table_df, plot_df = metrics_result.to_dataframe()

    expected_table = pd.DataFrame(
        [
            {"algorithm": "A", "metric": "one", "statistic": "avg", "mean": 1.0, "margin_of_error": 0.1},
            {"algorithm": "A", "metric": "one", "statistic": "mdn", "mean": 1.5, "margin_of_error": 0.2},
            {"algorithm": "B", "metric": "two", "statistic": "avg", "mean": 2.0, "margin_of_error": 0.3},
        ],
        columns=["algorithm", "metric", "statistic", "mean", "margin_of_error"],
    )
    expected_plot = pd.DataFrame(
        [
            {"algorithm": "A", "metric": "one", "x": 0.0, "y_mean": 1.0, "y_min": 0.5, "y_max": 1.5},
            {"algorithm": "A", "metric": "one", "x": 1.0, "y_mean": 2.0, "y_min": 1.5, "y_max": 2.5},
            {"algorithm": "B", "metric": "two", "x": 2.0, "y_mean": 3.0, "y_min": 2.5, "y_max": 3.5},
        ],
        columns=["algorithm", "metric", "x", "y_mean", "y_min", "y_max"],
    )

    pd.testing.assert_frame_equal(table_df, expected_table)
    pd.testing.assert_frame_equal(plot_df, expected_plot)


def test_metric_result_to_dataframe_handles_missing_table_or_plot_results() -> None:
    pytest.importorskip("pandas")

    algorithm = _AlgorithmStub("A")
    metric = _MetricStub("one")

    table_only_result = MetricResult(
        agent_metrics=None,
        table_metrics=[metric],
        plot_metrics=None,
        table_results={algorithm: {metric: {"avg": (1.0, 0.0)}}},
        plot_results=None,
    )
    plot_only_result = MetricResult(
        agent_metrics=None,
        table_metrics=None,
        plot_metrics=[metric],
        table_results=None,
        plot_results={algorithm: {metric: ([0.0], [1.0], [0.5], [1.5])}},
    )

    table_only_df, table_only_plot_df = table_only_result.to_dataframe()
    plot_only_table_df, plot_only_df = plot_only_result.to_dataframe()

    assert table_only_plot_df is None
    assert plot_only_table_df is None
    assert list(table_only_df.columns) == ["algorithm", "metric", "statistic", "mean", "margin_of_error"]
    assert list(plot_only_df.columns) == ["algorithm", "metric", "x", "y_mean", "y_min", "y_max"]


def test_balanced_accuracy_uses_per_class_recall() -> None:
    metric_utils._clear_caches()
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    problem = _ProblemStub(labels)
    agent = _AgentViewStub(is_server=False, predictions=predictions)

    result = ml.BalancedAccuracy().get_data_from_trial([agent], problem, -1)

    assert result == [balanced_accuracy_score(labels, predictions)]


def test_server_balanced_accuracy_uses_server_state_predictions() -> None:
    metric_utils._clear_caches()
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 0]
    problem = _ProblemStub(labels)
    server = _AgentViewStub(is_server=True, predictions=[])
    client = _AgentViewStub(is_server=False, predictions=predictions)

    result = ml.ServerBalancedAccuracy().get_data_from_trial([server, client], problem, -1)

    assert result == (balanced_accuracy_score(labels, predictions),)

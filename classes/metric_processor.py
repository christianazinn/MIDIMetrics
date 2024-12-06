from abc import abstractmethod
from pathlib import Path

from classes.metric import Metric
from classes.metric_config import MetricConfig
from metrics.pitch_metrics import BarPitchVarietyMetric

from symusic import Score

class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig

    def __init__(self,
                 metric_config: MetricConfig):
        self.metric_config = metric_config

        self.metrics=[]
        if metric_config.bar_pitch_variety:
            self.metrics.append(BarPitchVarietyMetric())

    def compute_metrics(self, midi_file: str | Path):

        score = Score(midi_file)

        for metric in self.metrics:
            metric.compute_metric(metric_config=self.metric_config, score = score)
            metric.plot_metric()





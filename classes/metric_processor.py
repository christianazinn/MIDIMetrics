from abc import abstractmethod
from pathlib import Path

import numpy as np
from miditok.utils import get_bars_ticks

from classes.metric import Metric
from classes.metric_config import MetricConfig
from metrics.pitch_metrics import BarPitchVarietyMetric, BarAbsolutePitchesMetric

from symusic import Score

from metrics.rythm_metrics import BarNoteDensity, BarNoteDensityMetric, NoteDurationsMetric


class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig

    def __init__(self,
                 metric_config: MetricConfig):
        self.metric_config = metric_config

        self.metrics=[]
        if metric_config.bar_absolute_pitches:
            self.metrics.append(BarAbsolutePitchesMetric())
        if metric_config.bar_pitch_variety:
            self.metrics.append(BarPitchVarietyMetric())
        if metric_config.bar_note_density:
            self.metrics.append(BarNoteDensityMetric())
        if metric_config.note_durations:
            self.metrics.append(NoteDurationsMetric())

    def compute_metrics(self, midi_file: str | Path):

        self.score = Score(midi_file)

        _window_bars_ticks = self._get_window_bars_ticks()

        for metric in self.metrics:
            metric.compute_metric(metric_config=self.metric_config,
                                  score = self.score,
                                  window_bars_ticks = _window_bars_ticks)
            #metric.plot_metric()

    def _get_window_bars_ticks(self,):
        bars_ticks = np.array(get_bars_ticks(self.score))

        infilling_start_idx = self.metric_config.infilled_bars[0]
        infilling_end_idx = self.metric_config.infilled_bars[1]
        infilling_length = infilling_end_idx - infilling_start_idx

        # infilling_bars_ticks = bars_ticks[infilling_start_idx:infilling_end_idx]
        return bars_ticks[infilling_start_idx - self.metric_config.context_size
                                       :infilling_end_idx + self.metric_config.context_size + 1]






import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
from miditok.utils import get_bars_ticks

from classes.metric import Metric, MetricType
from classes.metric_config import MetricConfig
from metrics.pitch_metrics import BarPitchVarietyMetric, BarAbsolutePitchesMetric

from symusic import Score

from metrics.rythm_metrics import BarNoteDensityMetric, NoteDurationsMetric


class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig

    def __init__(self,
                 metric_config: MetricConfig):
        self.metric_config = metric_config

        self.metrics=[]
        if metric_config.bar_absolute_pitches:
            self.metrics.append(BarAbsolutePitchesMetric(MetricType.BAR))
        if metric_config.bar_pitch_variety:
            self.metrics.append(BarPitchVarietyMetric(MetricType.BAR))
        if metric_config.bar_note_density:
            self.metrics.append(BarNoteDensityMetric(MetricType.BAR))
        if metric_config.note_durations:
            self.metrics.append(NoteDurationsMetric(MetricType.GLOBAL))

    def compute_metrics(self, midi_file: str | Path):

        self.score = Score(midi_file)

        _window_bars_ticks = self._get_window_bars_ticks()

        if _window_bars_ticks is None:
            msg = ("[ERROR] MetricsProcessor::compute_metrics Couldn't compute"
                   f" bars ticks values for midi file: {midi_file}")
            raise ValueError(msg)

        _context_ticks = (_window_bars_ticks[0], _window_bars_ticks[-1])
        _infilling_ticks = (_window_bars_ticks[self.metric_config.context_size],
                _window_bars_ticks[-self.metric_config.context_size - 1])

        start_time = time.time()

        for metric in self.metrics:
            metric.compute_metric(metric_config=self.metric_config,
                                  score = self.score,
                                  window_bars_ticks = _window_bars_ticks)
            #metric.plot_metric()
        end_time = time.time()

        print(f"Time to compute metrics on midi file {midi_file}: {end_time - start_time} seconds")

    def _get_window_bars_ticks(self):
        bars_ticks = np.array(get_bars_ticks(self.score))

        infilling_start_idx = self.metric_config.infilled_bars[0]
        infilling_end_idx = self.metric_config.infilled_bars[1]

        return bars_ticks[infilling_start_idx - self.metric_config.context_size
                                       :infilling_end_idx + self.metric_config.context_size + 1]





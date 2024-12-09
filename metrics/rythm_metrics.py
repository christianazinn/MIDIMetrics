import numpy as np
from symusic import Score

from classes.metric import Metric
from classes.metric_config import MetricConfig


class BarNoteDensityMetric(Metric):
    def compute_metric(self, metric_config: MetricConfig, score: Score, window_bars_ticks: np.array):
        infilling_length = metric_config.infilled_bars[1] - metric_config.infilled_bars[0]

        self.context_distribution = []
        self.infilling_distribution = []
        for idx, track in enumerate(score.tracks):
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks) - 1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i + 1]))[0]
                if idx == metric_config.infilled_track_idx and i in range(metric_config.context_size,
                                                                          infilling_length + metric_config.context_size):
                    self.infilling_distribution.append(len(note_idxs))
                else:
                    track_distribution.append(len(note_idxs))  # Add number of different pitches

            self.context_distribution.append(track_distribution)

        return

    def write_to_json(self):
        return

class NoteDurationsMetric(Metric):
    """
        NoteDurationsMetric class

        Computes the occurrence of every duration over the context
        and the infilling section.

    """
    def compute_metric(self, metric_config: MetricConfig, score: Score, window_bars_ticks: np.array):
        infilling_length = metric_config.infilled_bars[1] - metric_config.infilled_bars[0]

        self.context_durations_frquency = {}  # Durations in all context
        self.track_context_durations_frquency = {} # Durations only in the context of the infilling track
        self.infilling_durations_frquency = {}
        for idx, track in enumerate(score.tracks):
            times = np.array([note.time for note in track.notes])
            if idx == metric_config.infilled_track_idx:
                for i in range(len(window_bars_ticks) - 1):
                    note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i + 1]))[0]
                    durations = np.array([note.duration for note in track.notes[note_idxs]])
                    if i in range(metric_config.context_size,
                               infilling_length + metric_config.context_size):
                        unique_values, counts = np.unique(durations, return_counts=True)
                        for value, count in zip(unique_values, counts):
                            self.infilling_durations_frquency[value] += count
                    unique_values, counts = np.unique(durations, return_counts=True)
                    for value, count in zip(unique_values, counts):
                        self.track_context_durations_frquency[value] += count
            else:
                note_idxs = np.where((times >= window_bars_ticks[0]) & (times < window_bars_ticks[-1]))[0]
                durations = np.array([note.duration for note in track.notes[note_idxs]])
                unique_values, counts = np.unique(durations, return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.context_durations_frquency[value] += count

        return
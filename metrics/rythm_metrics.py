from collections import defaultdict

import numpy as np
from symusic import Score

from classes.metric import Metric
from classes.generation_config import GenerationConfig

import pandas as pd


class BarNoteDensityMetric(Metric):

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        self.context_distribution = []
        self.infilling_distribution = []
        self.track_context_distribution = []
        for idx, track in enumerate(score.tracks):
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks) - 1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i + 1]))[0]
                if idx == generation_config.infilled_track_idx:
                    if i in range(generation_config.context_size,infilling_length + generation_config.context_size):
                        self.infilling_distribution.append(len(note_idxs))
                    else:
                        self.track_context_distribution.append(len(note_idxs))
                        track_distribution.append(len(note_idxs))
                else:
                    track_distribution.append(len(note_idxs))  # Add number of different pitches

            self.context_distribution.append(track_distribution)

        return (self.track_context_distribution,
                self.infilling_distribution,
                None)

class NoteDurationsMetric(Metric):
    """
        NoteDurationsMetric class

        Computes the occurrence of every duration over the context
        and the infilling section.

    """
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        context_start_ticks = window_bars_ticks[0]
        context_end_ticks = window_bars_ticks[-1]
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        self.context_durations_frquency = defaultdict(int)  # Durations in all context
        self.track_context_durations_frquency = defaultdict(int) # Durations only in the context of the infilling track
        self.infilling_durations_frquency = defaultdict(int)
        for idx, track in enumerate(score.tracks):

            durations = np.array([note.duration for note in track.notes])
            times = np.array([note.time for note in track.notes])

            if idx == generation_config.infilled_track_idx:
                track_context_note_idxs = np.where(
                    ((times >= context_start_ticks) & (times < infilling_start_ticks)) |
                    ((times >= infilling_end_ticks) & (times < context_end_ticks))
                )[0]
                track_context_durations = durations[track_context_note_idxs]
                unique_values, counts = np.unique(track_context_durations, return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.track_context_durations_frquency[value] += count
                    self.context_durations_frquency[value] += count

                infilling_note_idxs = np.where((times >= infilling_start_ticks) & (times < infilling_end_ticks))[0]
                infilling_durations = durations[infilling_note_idxs]
                unique_values, counts = np.unique(infilling_durations, return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.infilling_durations_frquency[value] += count

            else:
                note_idxs = np.where((times >= context_start_ticks) & (times < context_end_ticks))[0]
                durations = durations[note_idxs]
                unique_values, counts = np.unique(durations, return_counts=True)
                for value, count in zip(unique_values, counts):
                    self.context_durations_frquency[value] += count

        return (self.track_context_durations_frquency,
                self.infilling_durations_frquency,
                self.context_durations_frquency)
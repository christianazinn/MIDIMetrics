from lib2to3.fixes.fix_input import context
from multiprocessing.managers import Value

import numpy as np
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.generation_config import GenerationConfig

import matplotlib.pyplot as plt
from itertools import chain


class BarAbsolutePitchesMetric(Metric):
    """
        barAbsolutePitchesMetric class

        Computes the number of notes for each absolute pitch
    """

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        self.context_pitch_class_set = set() # Pitches in all context
        self.track_context_pitch_class_set = set() # Pitches only in the context of the infilling track
        self.infilling_pitch_class_set = set() # Pitches in the infilled section
        for idx, track in enumerate(score.tracks):
            pitches = np.array([note.pitch for note in track.notes])
            times = np.array([note.time for note in track.notes])

            for i in range(len(window_bars_ticks) - 1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i + 1]))[0]
                if idx == generation_config.infilled_track_idx:
                    pitch_classes = np.unique(pitches[note_idxs]) % 12
                    if i in range(generation_config.context_size, infilling_length + generation_config.context_size):
                        self.infilling_pitch_class_set.update(pitch_classes)
                    else:
                        self.track_context_pitch_class_set.update(pitch_classes)
                        self.context_pitch_class_set.update(pitch_classes)
                else:
                    pitch_classes = np.unique(pitches[note_idxs]) % 12
                    self.context_pitch_class_set.update(pitch_classes)

        return self.track_context_pitch_class_set, self.infilling_pitch_class_set, self.context_pitch_class_set

class BarPitchVarietyMetric(Metric):
    """
        barPitchVarietyMetric class

        Computes the distribution of the number of different pitches
        across all bars.
    """
    @override
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        self.context_distribution = []
        self.infilling_distribution = []
        self.track_context_distribution = []
        for idx, track in enumerate(score.tracks):
            pitches = np.array([note.pitch for note in track.notes])
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks)-1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i+1]))[0]
                if idx == generation_config.infilled_track_idx:
                    if i in range(generation_config.context_size,infilling_length+generation_config.context_size):
                        self.infilling_distribution.append(len(np.unique(pitches[note_idxs])))
                    else:
                        self.track_context_distribution.append(len(np.unique(pitches[note_idxs])))
                else:
                    track_distribution.append(len(np.unique(pitches[note_idxs]))) # Add number of different pitches
                    self.context_distribution.append(track_distribution)

        return (self.track_context_distribution,
                self.infilling_distribution,
                None)
    
    def plot_metric(self):
        # Flatten the list using itertools.chain
        flattened_data1 = list(chain.from_iterable(self.context_distribution))
        data2 = self.infilling_distribution

        # Create subplots (1 row, 2 columns)
        plt.subplot(1, 2, 1)  # (rows, columns, subplot index)
        plt.hist(flattened_data1, bins=np.arange(min(flattened_data1), max(flattened_data1) + 2) - 0.5,
                 edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Occurrences')
        plt.title('Occurrence Histogram (Data1)')
        plt.xticks(np.arange(min(flattened_data1), max(flattened_data1) + 1))
        plt.grid(True)

        # Plot the second data vector in the second subplot
        plt.subplot(1, 2, 2)  # (rows, columns, subplot index)
        plt.hist(data2, bins=np.arange(min(data2), max(data2) + 2) - 0.5, edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Occurrences')
        plt.title('Occurrence Histogram (Data2)')
        plt.xticks(np.arange(min(data2), max(data2) + 1))
        plt.grid(True)

        # Show the figure
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()












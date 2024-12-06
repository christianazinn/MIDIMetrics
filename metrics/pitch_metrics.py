from lib2to3.fixes.fix_input import context

import numpy as np
from miditok.utils import get_bars_ticks
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.metric_config import MetricConfig

import matplotlib.pyplot as plt
from itertools import chain


class BarAbsolutePitchesMetric:
    """
        barAbsolutePitchesMetric class

        Computes the number of notes for each absolute pitch
    """

    def __init__(self):
        pass

class BarPitchVarietyMetric(Metric):
    """
        barPitchVarietyMetric class

        Computes the distribution of the number of different pitches
        across all bars.
    """
    @override
    def compute_metric(self, metric_config: MetricConfig, score: Score):
        bars_ticks = np.array(get_bars_ticks(score))

        infilling_start_idx = metric_config.infilled_bars[0]
        infilling_end_idx = metric_config.infilled_bars[1]
        infilling_length = infilling_end_idx - infilling_start_idx

        #infilling_bars_ticks = bars_ticks[infilling_start_idx:infilling_end_idx]
        window_bars_ticks = bars_ticks[infilling_start_idx - metric_config.context_size
                                       :infilling_end_idx + metric_config.context_size+1]

        self.context_distribution = []
        self.infilling_distribution = []
        for idx, track in enumerate(score.tracks):
            pitches = np.array([note.pitch for note in track.notes])
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks)-1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i+1]))[0]
                if idx == metric_config.infilled_track_idx and i in range(metric_config.context_size,
                                                                     infilling_length+metric_config.context_size):
                    self.infilling_distribution.append(len(np.unique(pitches[note_idxs])))
                else:
                    track_distribution.append(len(np.unique(pitches[note_idxs]))) # Add number of different pitches

            self.context_distribution.append(track_distribution)
    
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












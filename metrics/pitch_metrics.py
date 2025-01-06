import os
from pathlib import Path

import numpy as np
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.generation_config import GenerationConfig

import matplotlib.pyplot as plt
from itertools import chain

from collections import Counter


class BarAbsolutePitchesMetric(Metric):
    """
        barAbsolutePitchesMetric class

        Computes the number of notes for each absolute pitch
    """
    def __init__(self):
        super().__init__()
        self.infilling_vs_track_context_differences = []
        self.infilling_vs_context_differences = []
        self.individual_results = []

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        self.context_pitch_class_set = set()  # Pitches in all context
        self.track_context_pitch_class_set = set()  # Pitches only in the context of the infilling track
        self.infilling_pitch_class_set = set()  # Pitches in the infilled section
        for idx, track in enumerate(score.tracks):
            pitches = np.array([note.pitch for note in track.notes]).astype(int)
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

        self.finalize_computation(generation_config)

    def finalize_computation(self, generation_config: GenerationConfig):
        # Compute differences
        infilling_vs_track_context_diff = len(
            self.infilling_pitch_class_set - self.track_context_pitch_class_set
        )
        infilling_vs_context_diff = len(
            self.infilling_pitch_class_set - self.context_pitch_class_set
        )

        # Save results for distribution calculation
        self.infilling_vs_track_context_differences.append(infilling_vs_track_context_diff)
        self.infilling_vs_context_differences.append(infilling_vs_context_diff)

        # Save individual file results
        self.individual_results.append((
            generation_config.filename,
            infilling_vs_track_context_diff,
            infilling_vs_context_diff
        ))

    def analysis(self):
        """
        Finalizes the analysis by computing frequency distributions of differences
        """
        self.track_context_distribution = Counter(self.infilling_vs_track_context_differences)
        self.context_distribution = Counter(self.infilling_vs_context_differences)

    def output_results(self, output_folder: Path | str):

        output_folder = Path(output_folder) / "BarAbsolutePitchesMetric"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.results_to_txt(output_folder)

    def results_to_txt(self, output_folder: Path | str):
        """
        Writes the results to a text file in the specified output folder.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        results_file = os.path.join(output_folder, "BarAbsolutePitchesMetric.txt")

        with open(results_file, "w") as f:
            # Write Final Results
            f.write("##### FINAL RESULTS #######\n")
            f.write("Track Context Distribution:\n")
            for k, v in sorted(self.track_context_distribution.items()):
                f.write(f"{k}: {v}\n")
            f.write("Context Distribution:\n")
            for k, v in sorted(self.context_distribution.items()):
                f.write(f"{k}: {v}\n")
            f.write("#########################\n\n")

            # Write Individual Results
            f.write("##### INDIVIDUAL FILES #######\n")
            for filename, track_context_diff, context_diff in self.individual_results:
                f.write(f"File: {filename}\n")
                f.write(f"  Infilling vs Track Context Differences: {track_context_diff}\n")
                f.write(f"  Infilling vs Context Differences: {context_diff}\n")
                f.write("-------------------------\n")


class BarPitchVarietyMetric(Metric):
    """
        barPitchVarietyMetric class (should rename UPC as in
        https://arxiv.org/pdf/2011.06801)

        Computes the distribution of the number of different pitches
        across all bars.
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.file_statistics = []

    @override
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        self.context_distribution = []
        self.infilling_distribution = []
        self.track_context_distribution = []
        for idx, track in enumerate(score.tracks):
            pitches = np.array([note.pitch for note in track.notes]).astype(int)
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks)-1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i+1]))[0]
                if idx == generation_config.infilled_track_idx:
                    if i in range(generation_config.context_size,infilling_length+generation_config.context_size):
                        self.infilling_distribution.append(len(np.unique(pitches[note_idxs])))
                    else:
                        self.track_context_distribution.append(len(np.unique(pitches[note_idxs])))
                        track_distribution.append(len(np.unique(pitches[note_idxs])))
                else:
                    track_distribution.append(len(np.unique(pitches[note_idxs]))) # Add number of different pitches
            self.context_distribution.append(track_distribution)

        # Compute statistics for the current file
        self.file_statistics.append({
            "filename": generation_config.filename,
            "track_context_stats": self.compute_statistics(self.track_context_distribution),
            "infilling_stats": self.compute_statistics(self.infilling_distribution),
        })

    def compute_statistics(self, values):
        """Compute mean, std. dev, median, min, and max for a list of values."""
        return {
            "mean": np.mean(values) if values else 0,
            "std_dev": np.std(values) if values else 0,
            "median": np.median(values) if values else 0,
            "min": np.min(values) if values else 0,
            "max": np.max(values) if values else 0,
        }

    def analysis(self):
        # No additional global analysis needed, as metrics are computed per file
        return

    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "BarPitchVarietyMetric"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Plot each statistic
        for field in ['mean', 'std_dev', 'median', 'min', 'max']:
            self.plot(output_folder, field)

        self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Outputs the filename, track context stats, and infilling stats to a text file.
        """
        with open(output_folder / "pitch_variety_stats.txt", 'w') as file:
            # Write a header
            file.write("##### TRACK CONTEXT vs INFILLING STATS #####\n")
            file.write(
                "Filename | Context Mean | Context Std. Dev. | Context Max | Context Min | Infilling Mean | Infilling Std. Dev. | Infilling Max | Infilling Min\n")
            file.write(
                "--------------------------------------------------------------------------------------------------------\n")

            # Write stats for each MIDI file
            for stats in self.file_statistics:
                filename = stats['filename']
                track_context_stats = stats['track_context_stats']
                infilling_stats = stats['infilling_stats']

                line = f"{filename} | " \
                       f"{track_context_stats['mean']:.2f} | " \
                       f"{track_context_stats['std_dev']:.2f} | " \
                       f"{track_context_stats['max']} | " \
                       f"{track_context_stats['min']} | " \
                       f"{infilling_stats['mean']:.2f} | " \
                       f"{infilling_stats['std_dev']:.2f} | " \
                       f"{infilling_stats['max']} | " \
                       f"{infilling_stats['min']}\n"
                file.write(line)

            # Close the file
            file.write("###############################################\n")

    def plot(self, output_folder: Path | str, field: str = None):
        """
        Plots the statistics of context and infilling distributions for each MIDI file.
        Red = Context, Blue = Infilling
        """
        if field is None:
            msg = f"{BarPitchVarietyMetric.__class__} internal error in plot function. field is None"
            raise ValueError(msg)

        # Extract data
        context_values = [stats['track_context_stats'][field] for stats in self.file_statistics]
        infilling_values = [stats['infilling_stats'][field] for stats in self.file_statistics]
        filenames = [stats['filename'] for stats in self.file_statistics]

        # Use indices as x-axis labels instead of filenames
        indices = list(range(len(filenames)))

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot context values (without connections between dots)
        plt.plot(indices, context_values, 'ro', label=f'Context {field.capitalize()}')

        # Plot infilling values (without connections between dots)
        plt.plot(indices, infilling_values, 'bo', label=f'Infilling {field.capitalize()}')

        # Add horizontal lines connecting context and infilling points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]], [context_values[i], infilling_values[i]], 'k--', lw=1)

        # Annotate plot
        plt.title(f'{field.capitalize()} of Context and Infilling Pitch Variety')
        plt.xlabel('MIDI File Index')
        plt.ylabel(f'{field.capitalize()} Pitch Variety')
        plt.xticks(indices, [f"File {i}" for i in indices], rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.tight_layout()

        # Save plot with an appropriate name
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_folder / f"context_vs_infilling_{field}.png")
        plt.close()







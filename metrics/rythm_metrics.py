from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.generation_config import GenerationConfig



class BarNoteDensityMetric(Metric):

    """
        Computes the number of notes on each bar.

        Given the window context, the number of notes is computed for each bar
        in the infilling region, in the context in the same track, and in the context
        of all tracks. Descriptive statistics (mean, std. dev, median, min, max)
        is then computed for infilling and context distributions.
        Plots can be used to visualize the results
    """

    def __init__(self):
        super().__init__()
        # Store individual descriptive statistics for each MIDI file
        self.file_statistics = []


    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_length = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        context_distribution = []
        infilling_distribution = []
        track_context_distribution = []
        for idx, track in enumerate(score.tracks):
            times = np.array([note.time for note in track.notes])

            track_distribution = []

            for i in range(len(window_bars_ticks) - 1):
                note_idxs = np.where((times >= window_bars_ticks[i]) & (times < window_bars_ticks[i + 1]))[0]
                if idx == generation_config.infilled_track_idx:
                    if i in range(generation_config.context_size,infilling_length + generation_config.context_size):
                        infilling_distribution.append(len(note_idxs))
                    else:
                        track_context_distribution.append(len(note_idxs))
                        track_distribution.append(len(note_idxs))
                else:
                    track_distribution.append(len(note_idxs))  # Add number of different pitches

            context_distribution.append(track_distribution)

        # Compute descriptive statistics for this file
        track_stats = self._compute_descriptive_statistics(track_context_distribution)
        infilling_stats = self._compute_descriptive_statistics(infilling_distribution)

        # Store stats along with the filename
        self.file_statistics.append({
            "filename": generation_config.filename,
            "track_context_stats": track_stats,
            "infilling_stats": infilling_stats,
        })

    def _compute_descriptive_statistics(self, data):
        if not data:
            return {"mean": 0, "std_dev": 0, "median": 0, "max": 0, "min": 0}

        return {
            "mean": np.mean(data),
            "std_dev": np.std(data, ddof=1) if len(data) > 1 else 0,
            "median": np.median(data),
            "max": np.max(data),
            "min": np.min(data),
        }

    def analysis(self):
        return

    def output_results(self, output_folder: Path | str):

        output_folder = Path(output_folder) / "BarNoteDensityMetric"

        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        # Plot each statistic
        for field in ['mean', 'std_dev', 'max', 'min', 'median']:
            self.plot(output_folder, field)

        self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Outputs the filename, track context stats, and infilling stats to a text file.
        """
        with open(output_folder / "track_context_vs_infilling_stats.txt", 'w') as file:
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
            msg = f"{BarNoteDensityMetric.__class__} internal error in plot function. field is None"
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
        plt.title(f'{field.capitalize()} of Context and Infilling Note Densities')
        plt.xlabel('MIDI File Index')
        plt.ylabel(f'{field.capitalize()} Note Density')
        plt.xticks(indices, [f"File {i}" for i in indices], rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.tight_layout()

        # Save plot with an appropriate name
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_folder / f"context_vs_infilling_{field}.png")
        plt.close()

class NoteDurationsSetMetric(Metric):

    """
        Computes durations values in the infilling, track context
        and whole context regions. Then the count of durations
        present in the infilling but not the context is counted.
        Descriptive statistics is used to analyze the overall data
        across all MIDI files (e.g. avg value in output means
        the average number of durations present in the infilling
        but not the context)
    """

    def __init__(self):
        super().__init__()
        # Store individual descriptive statistics for each MIDI file
        self.file_statistics = []

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        context_start_ticks = window_bars_ticks[0]
        context_end_ticks = window_bars_ticks[-1]
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        # Create sets instead of frequency dictionaries
        self.context_durations_set = set()  # Durations in all context
        self.track_context_durations_set = set()  # Durations only in the context of the infilling track
        self.infilling_durations_set = set()  # Durations in the infilling region

        for idx, track in enumerate(score.tracks):

            durations = np.array([note.duration for note in track.notes])
            times = np.array([note.time for note in track.notes])

            if idx == generation_config.infilled_track_idx:
                # Track context: notes that are in the context of the infilling track
                track_context_note_idxs = np.where(
                    ((times >= context_start_ticks) & (times < infilling_start_ticks)) |
                    ((times >= infilling_end_ticks) & (times < context_end_ticks))
                )[0]
                track_context_durations = durations[track_context_note_idxs]
                # Add durations to the track context set
                self.track_context_durations_set.update(track_context_durations)
                # Also add them to the global context set
                self.context_durations_set.update(track_context_durations)

                # Infilling region: notes that are in the infilling region
                infilling_note_idxs = np.where((times >= infilling_start_ticks) & (times < infilling_end_ticks))[0]
                infilling_durations = durations[infilling_note_idxs]
                # Add durations to the infilling set
                self.infilling_durations_set.update(infilling_durations)

            else:
                # Context region for non-infilling tracks
                note_idxs = np.where((times >= context_start_ticks) & (times < context_end_ticks))[0]
                context_durations = durations[note_idxs]
                # Add durations to the global context set
                self.context_durations_set.update(context_durations)

        durations_not_in_track_ctx_count = self.infilling_durations_set - self.track_context_durations_set
        durations_not_in_whole_ctx_count = self.infilling_durations_set - self.context_durations_set

        # Store individual file statistics
        self.file_statistics.append({
            "filename": generation_config.filename,
            "durations_not_in_track_ctx_count": len(durations_not_in_track_ctx_count),
            "durations_not_in_whole_ctx_count": len(durations_not_in_whole_ctx_count),
        })

    def analysis(self):
        # Collect all the durations counts for analysis
        durations_not_in_track_ctx_counts = [file_stats["durations_not_in_track_ctx_count"] for file_stats in self.file_statistics]
        durations_not_in_whole_ctx_counts = [file_stats["durations_not_in_whole_ctx_count"] for file_stats in self.file_statistics]

        # Compute statistics for durations_not_in_track_ctx_count
        self.track_ctx_mean = np.mean(durations_not_in_track_ctx_counts)
        self.track_ctx_std = np.std(durations_not_in_track_ctx_counts)
        self.track_ctx_median = np.median(durations_not_in_track_ctx_counts)
        self.track_ctx_min = np.min(durations_not_in_track_ctx_counts)
        self.track_ctx_max = np.max(durations_not_in_track_ctx_counts)

        # Compute statistics for durations_not_in_whole_ctx_count
        self.whole_ctx_mean = np.mean(durations_not_in_whole_ctx_counts)
        self.whole_ctx_std = np.std(durations_not_in_whole_ctx_counts)
        self.whole_ctx_median = np.median(durations_not_in_whole_ctx_counts)
        self.whole_ctx_min = np.min(durations_not_in_whole_ctx_counts)
        self.whole_ctx_max = np.max(durations_not_in_whole_ctx_counts)

    def output_results(self, output_folder: Path | str):
        # Output the results to a text file
        output_folder = Path(output_folder) / "NoteDurationsSetMetric"
        output_folder.mkdir(parents=True, exist_ok=True)

        with open(output_folder / "infilling_durations_analysis.txt", "w") as f:
            # Write the global insights at the beginning of the file
            f.write("Global Statistics for Infilling Durations\n")
            f.write(f"Durations not in Track Context:\n")
            f.write(f"  Mean: {self.track_ctx_mean}\n")
            f.write(f"  Std. Dev: {self.track_ctx_std}\n")
            f.write(f"  Median: {self.track_ctx_median}\n")
            f.write(f"  Min: {self.track_ctx_min}\n")
            f.write(f"  Max: {self.track_ctx_max}\n")

            f.write(f"\nDurations not in Whole Context:\n")
            f.write(f"  Mean: {self.whole_ctx_mean}\n")
            f.write(f"  Std. Dev: {self.whole_ctx_std}\n")
            f.write(f"  Median: {self.whole_ctx_median}\n")
            f.write(f"  Min: {self.whole_ctx_min}\n")
            f.write(f"  Max: {self.whole_ctx_max}\n")

            # Write the header for individual file results
            f.write("\n\nFilename, Durations Not in Track Context Count, Durations Not in Whole Context Count\n")

            # Write the data for each file
            for file_stat in self.file_statistics:
                f.write(f"{file_stat['filename']}, "
                        f"{file_stat['durations_not_in_track_ctx_count']}, "
                        f"{file_stat['durations_not_in_whole_ctx_count']}\n")


class PolyphonyMinMaxMetric(Metric):

    """
        Computes the minimum and maximum polyphony values
        for the infilling and track context regions. Through the plots
        we can see if there are significant differences between the
        infilling and context portions.
    """

    def __init__(self):
        super().__init__()
        self.file_statistics = []

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        context_start_ticks = window_bars_ticks[0]
        context_end_ticks = window_bars_ticks[-1]
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        max_infilling_polyphony = 0
        min_infilling_polyphony = 0
        max_track_context_polyphony = 0
        min_track_context_polyphony = 0

        track = score.tracks[generation_config.infilled_track_idx]

        times = np.array([note.time for note in track.notes])

        # Track context: notes that are in the context of the infilling track
        track_context_note_idxs = np.where(
            ((times >= context_start_ticks) & (times < infilling_start_ticks)) |
            ((times >= infilling_end_ticks) & (times < context_end_ticks))
        )[0]

        # Find min and max polyphony for context here
        context_note_times = times[track_context_note_idxs]
        if len(context_note_times) > 0:
            unique_timestamps, counts = np.unique(context_note_times, return_counts=True)

            max_context_polyphony, min_context_polyphony = max(counts), min(counts)
        else:
            max_context_polyphony, min_context_polyphony = np.nan, np.nan

        # Infilling region: notes that are in the infilling region
        infilling_note_idxs = np.where((times >= infilling_start_ticks) & (times < infilling_end_ticks))[0]

        # Find min and max polyphony for infilling
        infilling_note_times = times[infilling_note_idxs]
        if len(infilling_note_times) > 0:
            unique_timestamps, counts = np.unique(infilling_note_times, return_counts=True)

            max_infilling_polyphony, min_infilling_polyphony = max(counts), min(counts)
        else:
            max_infilling_polyphony, min_infilling_polyphony = np.nan, np.nan

        # Store results for the current MIDI file
        self.file_statistics.append({
            'filename': generation_config.filename,
            'max_context_polyphony': max_context_polyphony,
            'min_context_polyphony': min_context_polyphony,
            'max_infilling_polyphony': max_infilling_polyphony,
            'min_infilling_polyphony': min_infilling_polyphony,
        })

    def analysis(self):
        return

    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "PolyphonyMinMaxMetric"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Create plots for max and min polyphony
        self.plot(output_folder, 'max')
        self.plot(output_folder, 'min')

    def plot(self, output_folder: Path, polyphony_type: str):
        """
        Create a plot for max or min polyphony values across all MIDI files.
        Ignores np.nan values from the plot and connects context and infilling points with vertical lines.
        """
        if polyphony_type not in ['max', 'min']:
            raise ValueError("polyphony_type must be 'max' or 'min'")

        # Extract data
        context_values = [stats[f"{polyphony_type}_context_polyphony"] for stats in self.file_statistics]
        infilling_values = [stats[f"{polyphony_type}_infilling_polyphony"] for stats in self.file_statistics]
        filenames = [stats['filename'] for stats in self.file_statistics]

        # Filter out np.nan values
        valid_indices = ~np.isnan(context_values) & ~np.isnan(infilling_values)
        context_values = np.array(context_values)[valid_indices]
        infilling_values = np.array(infilling_values)[valid_indices]
        filenames = np.array(filenames)[valid_indices]

        # Use indices as x-axis labels instead of filenames
        indices = list(range(len(filenames)))

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot only the points (remove the lines by using 'o' marker style)
        plt.plot(indices, context_values, 'ro', label=f'Context {polyphony_type.capitalize()} Polyphony')
        plt.plot(indices, infilling_values, 'bo', label=f'Infilling {polyphony_type.capitalize()} Polyphony')

        # Connect the points with vertical lines
        for i in indices:
            plt.plot([i, i], [context_values[i], infilling_values[i]], 'k-', lw=1)  # 'k-' is black line

        # Annotate plot
        plt.title(f'{polyphony_type.capitalize()} Polyphony in Context and Infilling Regions')
        plt.xlabel('MIDI File Index')
        plt.ylabel(f'{polyphony_type.capitalize()} Polyphony')
        plt.xticks(indices, [f"File {i}" for i in indices], rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / f"{polyphony_type}_polyphony_plot.png")
        plt.close()

#TODO: Maybe to be removed, gotta understand how to compare the 2 dictionaries
class NoteDurationsFrequencyMetric(Metric):
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

    def analysis(self):
        return

    def output_results(self, output_folder: Path | str):
        return


class RV(Metric):
    """
        Rythm Variations class

        RV measures how many distinct note durations the model plays within a sequence.
        As in https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf,
        it is computed as the average ratio across all sequences of
        the number of distinct note durations to the total number
        of notes in the sequence.
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.compare_with_original = True
        self.file_statistics = []
        self.rv_original = None
        self.rv_infilled = None
        # Add new structure to store analysis results
        self.analysis_results = {
            'average_difference': None,
            'differences': []
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        durations = np.array([note.duration for note in track.notes
                              if note.time >= infilling_start_ticks and note.time < infilling_end_ticks])

        # Calculate rhythm variations ratio
        if len(durations) == 0:
            rv = np.nan
        else:
            unique_durations = len(np.unique(durations))
            total_notes = len(durations)
            rv = unique_durations / total_notes

        if kwargs.get("is_original", False):
            self.rv_original = rv
            self.file_statistics.append({
                'filename': generation_config.filename,
                'rv_original': self.rv_infilled,
                'rv_infilled': self.rv_original
            })
        else:
            self.rv_infilled = rv

    @override
    def analysis(self):
        """Compute average difference between original and infilled RV values."""
        differences = []
        for stats in self.file_statistics:
            if not (np.isnan(stats['rv_original']) or np.isnan(stats['rv_infilled'])):
                diff = abs(stats['rv_original'] - stats['rv_infilled'])
                differences.append(diff)

        self.analysis_results['differences'] = differences
        self.analysis_results['average_difference'] = np.mean(differences) if differences else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "RV"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.plot(output_folder)
        self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot rhythm variation values for original and infilled MIDI files.
        """
        # Extract data, filtering out NaN values
        valid_stats = [(stats['rv_original'], stats['rv_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['rv_original']) or np.isnan(stats['rv_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original RV values
        plt.plot(indices, original_values, 'ro', label='Original RV')

        # Plot infilled RV values
        plt.plot(indices, infilled_values, 'bo', label='Infilled RV')

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=1)

        # Annotate plot
        plt.title('Rhythm Variations (RV) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('RV Value (Unique Durations / Total Notes)')
        plt.xticks(indices, [f"File {i}" for i in indices],
                   rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "rv_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the RV values for each file to a text file.
        """
        output_file = Path(output_folder) / "rv_results.txt"

        with output_file.open(mode='w') as file:
            file.write("Filename\tRV Original\tRV Infilled\tDifference\n")
            for i, stats in enumerate(self.file_statistics):
                if not (np.isnan(stats['rv_original']) or np.isnan(stats['rv_infilled'])):
                    difference = self.analysis_results['differences'][i]
                    file.write(f"{stats['filename']}\t"
                               f"{stats['rv_original']:.4f}\t"
                               f"{stats['rv_infilled']:.4f}\t"
                               f"{difference:.4f}\n")
                else:
                    file.write(f"{stats['filename']}\tNaN\tNaN\tNaN\n")

            # Add average difference at the end
            file.write(f"\nAverage Difference: "
                       f"{self.analysis_results['average_difference']:.4f}")

class QR(Metric):
    """
        Qualified Rhythm frequency class

        QR measures how many distinct note durations the model plays within a sequence,
        considering only qualified note durations (from 1/32 up to 1 bar long notes).
        As in https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf,
        it is computed as the average ratio across all sequences of
        the number of distinct qualified note durations to the total number of notes in the sequence.
        In https://arxiv.org/pdf/1709.06298 every note above 1/32 is considered
        qualified (definition too weak imo).
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.compare_with_original = True
        self.file_statistics = []
        self.qr_original = None
        self.qr_infilled = None
        # Add new structure to store analysis results
        self.analysis_results = {
            'average_difference': None,
            'differences': []
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        durations = np.array([note.duration for note in track.notes
                              if note.time >= infilling_start_ticks and note.time < infilling_end_ticks])

        # Define qualified durations from 1/32 to 1 bar long notes
        qualified_durations = [score.tpq / x for x in [2**i for i in range(-5, 3)]]

        # Calculate qualified rhythm variations ratio
        if len(durations) == 0:
            qr = np.nan
        else:
            qualified_durations = len([d for d in durations if d in qualified_durations])
            total_notes = len(durations)
            qr = qualified_durations / total_notes

        if kwargs.get("is_original", False):
            self.qr_original = qr
            self.file_statistics.append({
                'filename': generation_config.filename,
                'qr_original': self.qr_infilled,
                'qr_infilled': self.qr_original
            })
        else:
            self.qr_infilled = qr

    @override
    def analysis(self):
        """Compute average difference between original and infilled QR values."""
        differences = []
        for stats in self.file_statistics:
            if not (np.isnan(stats['qr_original']) or np.isnan(stats['qr_infilled'])):
                diff = abs(stats['qr_original'] - stats['qr_infilled'])
                differences.append(diff)

        self.analysis_results['differences'] = differences
        self.analysis_results['average_difference'] = np.mean(differences) if differences else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "QR"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.plot(output_folder)
        self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot qualified rhythm variation values for original and infilled MIDI files.
        """
        # Extract data, filtering out NaN values
        valid_stats = [(stats['qr_original'], stats['qr_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['qr_original']) or np.isnan(stats['qr_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original QR values
        plt.plot(indices, original_values, 'ro', label='Original QR')

        # Plot infilled QR values
        plt.plot(indices, infilled_values, 'bo', label='Infilled QR')

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=1)

        # Annotate plot
        plt.title('Qualified Rhythm Variations (QR) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('QR Value (Qualified Unique Durations / Total Notes)')
        plt.xticks(indices, [f"File {i}" for i in indices],
                   rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "qr_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the QR values for each file to a text file.
        """
        output_file = Path(output_folder) / "qr_results.txt"

        with output_file.open(mode='w') as file:
            file.write("Filename\tQR Original\tQR Infilled\tDifference\n")
            for i, stats in enumerate(self.file_statistics):
                if not (np.isnan(stats['qr_original']) or np.isnan(stats['qr_infilled'])):
                    difference = self.analysis_results['differences'][i]
                    file.write(f"{stats['filename']}\t"
                               f"{stats['qr_original']:.4f}\t"
                               f"{stats['qr_infilled']:.4f}\t"
                               f"{difference:.4f}\n")
                else:
                    file.write(f"{stats['filename']}\tNaN\tNaN\tNaN\n")

            # Add average difference at the end
            file.write(f"\nAverage Difference: "
                       f"{self.analysis_results['average_difference']:.4f}")

class GrooveConsistency(Metric):
    """
    GrooveConsistency class

    Originally presented in https://arxiv.org/pdf/2008.01307 (with the
    name of Grooving Pattern Similarity), helps in measuring the
    music’s rhythmicity. If a piece possesses a clear sense of
    rhythm, the grooving patterns between pairs of bars should
    be similar, thereby producing high GS scores; on the other
    hand, if the rhythm feels unsteady, the grooving patterns
    across bars should be erratic, resulting in low GS scores.
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.compare_with_original = True
        self.file_statistics = []
        self.groove_original = None
        self.groove_infilled = None
        # Add new structure to store analysis results
        self.analysis_results = {
            'average_difference': None,
            'differences': []
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        context_size = generation_config.context_size

        infilled_bars = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        track = score.tracks[generation_config.infilled_track_idx]
        times = np.array([note.time for note in track.notes])

        ticks_per_bar = window_bars_ticks[-1] - window_bars_ticks[-2]

        # Initialize the grooving pattern matrix
        grooving_pattern_matrix = np.zeros((infilled_bars, ticks_per_bar), dtype=bool)

        # Fill in the grooving pattern matrix for each bar
        for i in range(infilled_bars):
            bar_start = window_bars_ticks[i + context_size]
            bar_end = window_bars_ticks[i + context_size + 1]
            bar_times = times[(times >= bar_start) & (times < bar_end)] - bar_start

            grooving_pattern_matrix[i, bar_times] = 1

        # Compute pairwise grooving pattern similarities between adjacent bars
        hamming_distance = np.count_nonzero(
            grooving_pattern_matrix[:-1] != grooving_pattern_matrix[1:]
        )

        groove_consistency = 1 - hamming_distance / (ticks_per_bar * infilled_bars)

        # Store the result for original or infilled based on the flag
        if kwargs.get("is_original", False):
            self.groove_original = groove_consistency
            self.file_statistics.append({
                'filename': generation_config.filename,
                'groove_original': self.groove_original,
                'groove_infilled': self.groove_infilled
            })
        else:
            self.groove_infilled = groove_consistency

    @override
    def analysis(self):
        """Compute average difference between original and infilled Groove Consistency values."""
        differences = []
        for stats in self.file_statistics:
            if not (np.isnan(stats['groove_original']) or np.isnan(stats['groove_infilled'])):
                diff = abs(stats['groove_original'] - stats['groove_infilled'])
                differences.append(diff)

        self.analysis_results['differences'] = differences
        self.analysis_results['average_difference'] = np.mean(differences) if differences else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "GrooveConsistency"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.plot(output_folder)
        self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot Groove Consistency values for original and infilled MIDI files.
        """
        # Extract data, filtering out NaN values
        valid_stats = [(stats['groove_original'], stats['groove_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['groove_original']) or np.isnan(stats['groove_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original Groove Consistency values
        plt.plot(indices, original_values, 'ro', label='Original Groove Consistency')

        # Plot infilled Groove Consistency values
        plt.plot(indices, infilled_values, 'bo', label='Infilled Groove Consistency')

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=1)

        # Annotate plot
        plt.title('Groove Consistency of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('Groove Consistency (Average GS)')
        plt.xticks(indices, [f"File {i}" for i in indices],
                   rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "groove_consistency_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the Groove Consistency values for each file to a text file.
        """
        output_file = Path(output_folder) / "groove_consistency_results.txt"

        with output_file.open(mode='w') as file:

            # Add average difference at the end
            file.write(f"\nAverage Difference: "
                       f"{self.analysis_results['average_difference']:.4f}\n")

            file.write("Filename\tGroove Original\tGroove Infilled\tDifference\n")
            for i, stats in enumerate(self.file_statistics):
                if not (np.isnan(stats['groove_original']) or np.isnan(stats['groove_infilled'])):
                    difference = self.analysis_results['differences'][i]
                    file.write(f"{stats['filename']}\t"
                               f"{stats['groove_original']:.4f}\t"
                               f"{stats['groove_infilled']:.4f}\t"
                               f"{difference:.4f}\n")
                else:
                    file.write(f"{stats['filename']}\tNaN\tNaN\tNaN\n")

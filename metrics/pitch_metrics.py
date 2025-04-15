import os
from pathlib import Path

import numpy as np
import symusic
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.generation_config import GenerationConfig
from classes.constants import STEP, MEAN_LINES_WIDTH, POINT_DIM
import matplotlib.pyplot as plt
from itertools import chain

from collections import Counter

class UPC(Metric):
    """
        Computes the number of used pitch classes per bar (from 0 to 12).
        All the values are then averaged to obtain the final UPC value.
        See https://arxiv.org/pdf/1709.06298 for reference
    """

    def __init__(self):
        super().__init__()
        self.compare_with_original = True
        self.file_statistics = []
        self.original_upc = None
        self.infilled_upc = None
        self.analysis_results = {
            'avg_original': None,
            'std_original': None,
            'avg_infilled': None,
            'std_infilled': None,
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        context_size = generation_config.context_size

        infilled_bars = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]

        track = score.tracks[generation_config.infilled_track_idx]
        pitches = np.array([note.pitch for note in track.notes]).astype(int)
        times = np.array([note.time for note in track.notes])

        pitches_counts = []
        for i in range(infilled_bars):
            note_idxs = np.where(
                (times >= window_bars_ticks[i + context_size]) & (times < window_bars_ticks[i + context_size + 1]))[0]
            infilling_pitches = pitches[note_idxs]
            count = 0
            is_used = [False] * 12
            for pitch in infilling_pitches:
                pitch_class = pitch % 12
                if not is_used[pitch_class]:
                    is_used[pitch_class] = True
                    count += 1
            pitches_counts.append(count)

        upc = np.mean(np.array(pitches_counts))

        if kwargs.get("is_original", False):
            self.original_upc = upc
            self.file_statistics.append({
                'filename': generation_config.filename,
                'upc_original': self.original_upc,
                'upc_infilled': self.infilled_upc
            })
        else:
            self.infilled_upc = upc

    @override
    def analysis(self):
        """Compute average and standard deviation of original and infilled UPC values."""
        original_values = [stats['upc_original'] for stats in self.file_statistics if not np.isnan(stats['upc_original'])]
        infilled_values = [stats['upc_infilled'] for stats in self.file_statistics if not np.isnan(stats['upc_infilled'])]

        self.analysis_results['avg_original'] = np.mean(original_values) if original_values else np.nan
        self.analysis_results['std_original'] = np.std(original_values) if original_values else np.nan
        self.analysis_results['avg_infilled'] = np.mean(infilled_values) if infilled_values else np.nan
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else np.nan

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "UPC"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['avg_original'] - self.analysis_results['avg_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"UPC: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot UPC values for original and infilled MIDI files.
        """
        valid_stats = [(stats['upc_original'], stats['upc_infilled'], stats['filename'])
                      for stats in self.file_statistics
                      if not (np.isnan(stats['upc_original']) or np.isnan(stats['upc_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Retrieve analysis results
        avg_original = self.analysis_results['avg_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['avg_infilled']
        std_infilled = self.analysis_results['std_infilled']

        plt.figure(figsize=(10, 6))

        # Plot original UPC values
        plt.plot(indices, original_values, 'ro', label='Original UPC', markersize = POINT_DIM)

        # Plot infilled UPC values
        plt.plot(indices, infilled_values, 'bo', label='Infilled UPC', markersize = POINT_DIM)

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=0.2)

        # Plot mean and standard deviation for original values
        plt.axhline(avg_original, color='r', linestyle='-', linewidth=MEAN_LINES_WIDTH,
                   label=f'Mean Original ({avg_original:.4f})')
        plt.fill_between(indices, avg_original - std_original, avg_original + std_original,
                        color='r', alpha=0.2, label='Std Dev Original')

        # Plot mean and standard deviation for infilled values
        plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=MEAN_LINES_WIDTH,
                   label=f'Mean Infilled ({avg_infilled:.4f})')
        plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled,
                        color='b', alpha=0.2, label='Std Dev Infilled')

        plt.title('UPC (Used Pitch Classes) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('UPC Value')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_folder / "upc_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the UPC values for each file to a text file.
        """
        output_file = Path(output_folder) / "upc_results.txt"

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['avg_original'] - self.analysis_results['avg_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"UPC: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

        with output_file.open(mode='w') as file:
            file.write(f"Average Original UPC: {self.analysis_results['avg_original']:.4f}\n")
            file.write(f"Std Dev Original UPC: {self.analysis_results['std_original']:.4f}\n")
            file.write(f"Average Infilled UPC: {self.analysis_results['avg_infilled']:.4f}\n")
            file.write(f"Std Dev Infilled UPC: {self.analysis_results['std_infilled']:.4f}\n\n")

            file.write("Filename\tUPC Original\tUPC Infilled\n")
            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                          f"{stats['upc_original']:.4f}\t"
                          f"{stats['upc_infilled']:.4f}\n")

class Polyphony(Metric):
    """
    Return the average number of pitches being played concurrently.

    The polyphony is defined as the average number of pitches being
    played at the same time, evaluated only at time steps where at least
    one pitch is on. Drum tracks are ignored. Return NaN if no note is
    found. Used here for example https://arxiv.org/pdf/2212.11134
    """
    def __init__(self):
        super().__init__()
        self.compare_with_original = True
        self.file_statistics = []
        self.p_original = None
        self.p_infilled = None
        self.analysis_results = {
            'avg_original': None,
            'std_original': None,
            'avg_infilled': None,
            'std_infilled': None,
        }

    def _get_pianoroll(self, notes: list[symusic.Note], infilling_start: int, length: int) -> np.ndarray:
        """Return the binary pianoroll matrix."""
        pianoroll = np.zeros((length, 128), bool)
        for note in notes:
            if note.end > infilling_start + length:
                pianoroll[note.time - infilling_start: length, note.pitch] = 1
            else:
                pianoroll[note.time - infilling_start: note.end - infilling_start, note.pitch] = 1
        return pianoroll

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        notes = [note for note in track.notes
                 if infilling_start_ticks <= note.time < infilling_end_ticks]
        length = infilling_end_ticks - infilling_start_ticks

        pianoroll = self._get_pianoroll(notes, infilling_start_ticks, length)
        denominator = np.count_nonzero(pianoroll.sum(1) > 0)
        if denominator < 1:
            p = np.nan
        else:
            p = pianoroll.sum() / denominator

        if kwargs.get("is_original", False):
            self.p_original = p
            self.file_statistics.append({
                'filename': generation_config.filename,
                'p_original': self.p_original,
                'p_infilled': self.p_infilled
            })
        else:
            self.p_infilled = p

    @override
    def analysis(self):
        """Compute average and standard deviation of original and infilled polyphony values."""
        original_values = [stats['p_original'] for stats in self.file_statistics if not np.isnan(stats['p_original'])]
        infilled_values = [stats['p_infilled'] for stats in self.file_statistics if not np.isnan(stats['p_infilled'])]

        self.analysis_results['avg_original'] = np.mean(original_values) if original_values else np.nan
        self.analysis_results['std_original'] = np.std(original_values) if original_values else np.nan
        self.analysis_results['avg_infilled'] = np.mean(infilled_values) if infilled_values else np.nan
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else np.nan

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "Polyphony"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['avg_original'] - self.analysis_results['avg_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5
        
        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"Polyphony: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")


        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot polyphony values for original and infilled MIDI files.
        """
        valid_stats = [(stats['p_original'], stats['p_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['p_original']) or np.isnan(stats['p_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Retrieve analysis results
        avg_original = self.analysis_results['avg_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['avg_infilled']
        std_infilled = self.analysis_results['std_infilled']

        plt.figure(figsize=(10, 6))

        # Plot original polyphony values
        plt.plot(indices, original_values, 'ro', label='Original Polyphony', markersize = POINT_DIM)

        # Plot infilled polyphony values
        plt.plot(indices, infilled_values, 'bo', label='Infilled Polyphony', markersize = POINT_DIM)

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=0.2)

        # Plot mean and standard deviation for original values
        plt.axhline(avg_original, color='r', linestyle='-', linewidth=MEAN_LINES_WIDTH,
                   label=f'Mean Original ({avg_original:.4f})')
        plt.fill_between(indices, avg_original - std_original, avg_original + std_original,
                        color='r', alpha=0.2, label='Std Dev Original')

        # Plot mean and standard deviation for infilled values
        plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=MEAN_LINES_WIDTH,
                   label=f'Mean Infilled ({avg_infilled:.4f})')
        plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled,
                        color='b', alpha=0.2, label='Std Dev Infilled')

        # Annotate plot
        plt.title('Polyphony of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('Polyphony Value')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_folder / "polyphony_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the polyphony values for each file to a text file.
        """
        output_file = Path(output_folder) / "polyphony_results.txt"

        with output_file.open(mode='w') as file:
            file.write(f"Average Original Polyphony: {self.analysis_results['avg_original']:.4f}\n")
            file.write(f"Std Dev Original Polyphony: {self.analysis_results['std_original']:.4f}\n")
            file.write(f"Average Infilled Polyphony: {self.analysis_results['avg_infilled']:.4f}\n")
            file.write(f"Std Dev Infilled Polyphony: {self.analysis_results['std_infilled']:.4f}\n\n")

            file.write("Filename\tPolyphony Original\tPolyphony Infilled\n")
            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['p_original']:.4f}\t"
                           f"{stats['p_infilled']:.4f}\n")


class PR(Metric):
    """
    PolyphonyRatio class

    Computes the ratio of the number of time steps where more than one pitch
    is played to the total number of time steps. See https://arxiv.org/pdf/2011.06801
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.compare_with_original = True
        self.file_statistics = []
        self.pr_original = None
        self.pr_infilled = None
        self.analysis_results = {
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }

    def _get_pianoroll(self, notes: list[symusic.Note], infilling_start: int, length: int) -> np.ndarray:
        """Return the binary pianoroll matrix."""
        pianoroll = np.zeros((length, 128), bool)
        for note in notes:
            if note.end > infilling_start + length:
                pianoroll[note.time - infilling_start: length, note.pitch] = 1
            else:
                pianoroll[note.time - infilling_start: note.end - infilling_start, note.pitch] = 1
        return pianoroll

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        notes = [note for note in track.notes
                 if infilling_start_ticks <= note.time < infilling_end_ticks]
        length = infilling_end_ticks - infilling_start_ticks

        pianoroll = self._get_pianoroll(notes, infilling_start_ticks, length)
        denominator = np.count_nonzero(pianoroll.sum(1) > 0)
        if denominator < 1:
            pr = np.nan
        else:
            pr = np.count_nonzero(pianoroll.sum(1) > 1) / len(pianoroll)

        if kwargs.get("is_original", False):
            self.pr_original = pr
            self.file_statistics.append({
                'filename': generation_config.filename,
                'pr_original': self.pr_original,
                'pr_infilled': self.pr_infilled
            })
        else:
            self.pr_infilled = pr

    @override
    def analysis(self):
        """Compute statistics for original and infilled PR values."""
        original_values = [stats['pr_original'] for stats in self.file_statistics
                           if not np.isnan(stats['pr_original'])]
        infilled_values = [stats['pr_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['pr_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "PR"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['average_original'] - self.analysis_results['average_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5
        
        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"PR: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot PR values for original and infilled MIDI files.
        """
        # Extract data, filtering out NaN values
        valid_stats = [(stats['pr_original'], stats['pr_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['pr_original']) or np.isnan(stats['pr_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Retrieve analysis results
        avg_original = self.analysis_results['average_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['average_infilled']
        std_infilled = self.analysis_results['std_infilled']

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original PR values
        plt.plot(indices, original_values, 'ro', label='Original PR', markersize = POINT_DIM)

        # Plot infilled PR values
        plt.plot(indices, infilled_values, 'bo', label='Infilled PR', markersize = POINT_DIM)

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=0.2)

        # Plot mean and standard deviation for original values
        plt.axhline(avg_original, color='r', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Original ({avg_original:.4f})')
        plt.fill_between(indices, avg_original - std_original, avg_original + std_original, color='r', alpha=0.2)

        # Plot mean and standard deviation for infilled values
        plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Infilled ({avg_infilled:.4f})')
        plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled, color='b', alpha=0.2)

        # Annotate plot
        plt.title('Polyphony Ratio (PR) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('PR Value')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
        plt.legend(title=None)  # Remove the title from the legend
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "pr_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the PR values for each file to a text file.
        """
        output_file = Path(output_folder) / "pr_results.txt"

        with output_file.open(mode='w') as file:
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            file.write("Filename\tPR Original\tPR Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['pr_original']:.4f}\t"
                           f"{stats['pr_infilled']:.4f}\n")

class PV(Metric):
    """
        Pitch Variations class

        PV measures how many distinct pitches the model plays within a sequence.
        As in https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf,
        it is computes as the average ratio across all sequences of the
        number of distinct pitches to the total number of notes in the
        sequence.
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.file_statistics = []
        self.pv_original = None
        self.pv_infilled = None
        self.analysis_results = {
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }
        self.compare_with_original = True

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        pitches = np.array([note.pitch for note in track.notes
                            if note.time >= infilling_start_ticks and note.time < infilling_end_ticks])

        # Calculate pitch variations ratio
        if len(pitches) == 0:
            pv = np.nan
        else:
            unique_pitches = len(np.unique(pitches))
            total_notes = len(pitches)
            pv = unique_pitches / total_notes

        if kwargs.get("is_original", False):
            self.pv_original = pv
            self.file_statistics.append({
                'filename': generation_config.filename,
                'pv_original': self.pv_infilled,
                'pv_infilled': self.pv_original
            })
        else:
            self.pv_infilled = pv

    def analysis(self):
        """Compute statistics for original and infilled PV values."""
        original_values = [stats['pv_original'] for stats in self.file_statistics
                           if not np.isnan(stats['pv_original'])]
        infilled_values = [stats['pv_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['pv_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0

        return self.analysis_results

    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "PV"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['average_original'] - self.analysis_results['average_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5
        
        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"PV: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """
        Plot pitch variation values for original and infilled MIDI files.
        """
        # Extract data, filtering out NaN values
        valid_stats = [(stats['pv_original'], stats['pv_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['pv_original']) or np.isnan(stats['pv_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Retrieve analysis results
        avg_original = self.analysis_results['average_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['average_infilled']
        std_infilled = self.analysis_results['std_infilled']

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original PV values
        plt.plot(indices, original_values, 'ro', label='Original PV', markersize = POINT_DIM)

        # Plot infilled PV values
        plt.plot(indices, infilled_values, 'bo', label='Infilled PV', markersize = POINT_DIM)

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=0.2)

        # Plot mean and standard deviation for original values
        plt.axhline(avg_original, color='r', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Original ({avg_original:.4f})')
        plt.fill_between(indices, avg_original - std_original, avg_original + std_original, color='r', alpha=0.2)

        # Plot mean and standard deviation for infilled values
        plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Infilled ({avg_infilled:.4f})')
        plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled, color='b', alpha=0.2)

        # Annotate plot
        plt.title('Pitch Variations (PV) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('PV Value (Unique Pitches / Total Notes)')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
        plt.legend(title=None)  # Remove the title from the legend
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "pv_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the PV values for each file to a text file.
        """
        output_file = Path(output_folder) / "pv_results.txt"

        with output_file.open(mode='w') as file:
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            file.write("Filename\tPV Original\tPV Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['pv_original']:.4f}\t"
                           f"{stats['pv_infilled']:.4f}\n")

class PitchClassHistogramEntropy(Metric):
    """
    PitchClassHistogramEntropy class

    Implemented in https://arxiv.org/pdf/2008.01307
    The entropy, in information theory, is a measure of “uncertainty” of a probability distribution,
    hence we adopt it here as a metric to help assess the music’s quality in tonality.
    If a piece’s tonality is clear, several pitch classes should dominate the pitch histogram
    (e.g., the tonic and the dominant), resulting in a low-entropy.
    """

    def __init__(self):
        super().__init__()
        # Store statistics for each MIDI file
        self.compare_with_original = True
        self.file_statistics = []
        self.pch_original = None
        self.pch_infilled = None
        # Add new structure to store analysis results
        self.analysis_results = {
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }

    def _entropy(self, prob):
        """Calculate entropy of a probability distribution."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.nansum(prob * np.log2(prob))

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        """Compute the Pitch Class Histogram Entropy (PCH)."""
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        pitches = np.array([note.pitch for note in track.notes
                            if infilling_start_ticks <= note.time < infilling_end_ticks])

        # Compute histogram
        counter = np.zeros(12)
        for pitch in pitches:
            counter[pitch % 12] += 1

        # Normalize to get probabilities
        denominator = counter.sum()
        if denominator < 1:
            pch = np.nan
        else:
            prob = counter / denominator
            pch = self._entropy(prob)

        # Save results
        if kwargs.get("is_original", False):
            self.pch_original = pch
            self.file_statistics.append({
                'filename': generation_config.filename,
                'pch_original': self.pch_original,
                'pch_infilled': self.pch_infilled
            })
        else:
            self.pch_infilled = pch

    def analysis(self):
        """Compute statistics of original and infilled PCH values."""
        original_values = [stats['pch_original'] for stats in self.file_statistics
                           if not np.isnan(stats['pch_original'])]
        infilled_values = [stats['pch_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['pch_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0

        return self.analysis_results

    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "PitchClassHistogramEntropy"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['average_original'] - self.analysis_results['average_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"PCH: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

    def plot(self, output_folder: Path | str):
        """Plot PCH entropy for original and infilled MIDI files."""
        # Extract data, filtering out NaN values
        valid_stats = [(stats['pch_original'], stats['pch_infilled'], stats['filename'])
                       for stats in self.file_statistics
                       if not (np.isnan(stats['pch_original']) or np.isnan(stats['pch_infilled']))]

        if not valid_stats:
            print("No valid data points to plot")
            return

        original_values, infilled_values, filenames = zip(*valid_stats)
        indices = list(range(len(filenames)))

        # Retrieve analysis results
        avg_original = self.analysis_results['average_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['average_infilled']
        std_infilled = self.analysis_results['std_infilled']

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original PCH entropy values
        plt.plot(indices, original_values, 'ro', label='Original PCH Entropy', markersize = POINT_DIM)

        # Plot infilled PCH entropy values
        plt.plot(indices, infilled_values, 'bo', label='Infilled PCH Entropy', markersize = POINT_DIM)

        # Add connecting lines between original and infilled points
        for i in range(len(indices)):
            plt.plot([indices[i], indices[i]],
                     [original_values[i], infilled_values[i]],
                     'k--', lw=0.2)

        # Plot mean and standard deviation for original values
        plt.axhline(avg_original, color='r', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Original ({avg_original:.4f})')
        plt.fill_between(indices, avg_original - std_original, avg_original + std_original, color='r', alpha=0.2, label='Std Dev Original')

        # Plot mean and standard deviation for infilled values
        plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=MEAN_LINES_WIDTH, label=f'Mean Infilled ({avg_infilled:.4f})')
        plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled, color='b', alpha=0.2, label='Std Dev Infilled')

        # Annotate plot
        plt.title('Pitch Class Histogram Entropy (PCH) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('PCH Entropy Value')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels
        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "pch_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """Write the PCH entropy values for each file to a text file."""
        output_file = Path(output_folder) / "pch_results.txt"

        with output_file.open(mode='w') as file:
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            file.write("Filename\tPCH Original\tPCH Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['pch_original']:.4f}\t"
                           f"{stats['pch_infilled']:.4f}\n")


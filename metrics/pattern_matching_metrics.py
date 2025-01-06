from pathlib import Path
from typing import Set, Tuple

import numpy as np
from symusic import Score
from typing_extensions import override

from classes.generation_config import GenerationConfig
from classes.metric import Metric


class NGramsRepetitions(Metric):

    """
        Computes the number 2-3-4 n-gram repetition in common between the infilling and context regions.
    """

    def __init__(self):
        super().__init__()
        self.file_statistics = []

    @override
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        context_start_ticks = window_bars_ticks[0]
        context_end_ticks = window_bars_ticks[-1]
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        infilled_track = score.tracks[generation_config.infilled_track_idx]

        infilling_notes = [note for note in infilled_track.notes
                           if infilling_start_ticks <= note.time < infilling_end_ticks
                           ]

        context_notes_before = [note for note in infilled_track.notes
                           if context_start_ticks <= note.time < infilling_start_ticks
                           ]
        context_notes_after = [note for note in infilled_track.notes
                           if  infilling_end_ticks <= note.time < context_end_ticks
                           ]

        # Compute the n-gram repetitions
        output = self.compute_common_patterns(
            np.array(context_notes_before),
            np.array(infilling_notes),
            np.array(context_notes_after)
        )

        # Save file-specific statistics
        self.file_statistics.append({
            "filename": generation_config.filename,
            "ngrams": output  # Example: {'2-note': 2, '3-note': 2, '4-note': 2}
        })

    def extract_ngrams(self, notes: np.array, n: int, absolute: bool = False) -> Set[Tuple[Tuple[float, int], ...]]:
        """
        Extract n-grams of size n from a sequence of notes sorted by time.
        Each note is represented as (time, pitch).

        Args:
            notes (np.array): Array of notes, where each note is (time, pitch, duration).
            n (int): Size of the n-gram (e.g., 2, 3, 4).

        Returns:
            Set[Tuple[Tuple[float, int], ...]]: Set of n-grams containing (time, pitch).
        """

        ngrams = set()
        for i in range(len(notes) - n + 1):
            base_time = notes[i].time  # Get the base time for the n-gram
            if absolute:
                # Use absolute pitch values
                ngram = tuple((notes[i + j].time - base_time, notes[i + j].pitch) for j in range(n))
            else:
                # Use pitch differences
                base_pitch = notes[i].pitch
                ngram = tuple((notes[i + j].time - base_time, notes[i + j].pitch - base_pitch) for j in range(n))
            ngrams.add(ngram)
        return ngrams

    def compute_common_patterns(self, context_notes_before: np.array, infill_notes: np.array, context_notes_after: np.array, absolute=False) -> dict:
        """
        Compute the number of common 2, 3, and 4-note patterns (n-grams) between context and infilling regions.
        Comparison is based on pitch and time.

        Args:
            context_notes (np.array): Notes from the context region.
            infill_notes (np.array): Notes from the infilling region.

        Returns:
            dict: Dictionary with counts of common patterns for n-grams of size 2, 3, and 4.
        """
        result = {}
        for n in [2, 3, 4]:
            # Extract n-grams from context and infilling regions
            if absolute:
                context_ngrams_before_abs = self.extract_ngrams(context_notes_before, n, absolute=True)
                infill_ngrams_abs = self.extract_ngrams(infill_notes, n, absolute=True)
                context_ngrams_after_abs = self.extract_ngrams(context_notes_after, n, absolute=True)

                context_ngrams_abs = context_ngrams_before_abs.union(context_ngrams_after_abs)

                common_ngrams_abs = context_ngrams_abs.intersection(infill_ngrams_abs)

                result[f"{n}-note-abs"] = len(common_ngrams_abs)

            context_before_ngrams = self.extract_ngrams(context_notes_before, n)
            infill_ngrams = self.extract_ngrams(infill_notes, n)
            context_after_ngrams = self.extract_ngrams(context_notes_after, n)

            context_ngrams = context_before_ngrams.union(context_after_ngrams)

            # Compute the intersection
            common_ngrams = context_ngrams.intersection(infill_ngrams)

            # Store the count
            result[f"{n}-note"] = len(common_ngrams)
            if len(context_ngrams) > 0:
                result[f"{n}-note-ratio"] = (len(common_ngrams) / len(context_ngrams)) * 100

        return result

    def compute_statistics(self, key):
        """Compute mean, std. dev, median, min, and max for a specific n-gram type."""
        values = [file_data['ngrams'].get(key, 0) for file_data in self.file_statistics]
        return {
            "mean": np.mean(values) if values else 0,
            "std_dev": np.std(values) if values else 0,
            "median": np.median(values) if values else 0,
            "min": np.min(values) if values else 0,
            "max": np.max(values) if values else 0,
        }

    def analysis(self):
        return

    def output_results(self, output_folder: Path | str):
        """
        Outputs n-gram statistics, including ratios, to a text file with global statistics.
        """
        output_folder = Path(output_folder) / "NGramsRepetitions"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Prepare the file
        with open(output_folder / "ngrams_repetitions.txt", 'w') as file:
            # Write global statistics at the top
            file.write("##### GLOBAL STATISTICS #####\n")
            file.write("N-gram | Mean | Std. Dev. | Median | Min | Max\n")
            file.write("---------------------------------------------\n")

            for key in ['2-note', '3-note', '4-note', '2-note-ratio', '3-note-ratio', '4-note-ratio']:
                stats = self.compute_statistics(key)
                line = f"{key} | {stats['mean']:.2f} | {stats['std_dev']:.2f} | {stats['median']:.2f} | {stats['min']} | {stats['max']}\n"
                file.write(line)

            file.write("\n##### FILE-SPECIFIC N-GRAM REPETITIONS #####\n")
            file.write("Filename | 2-note | 3-note | 4-note | 2-note-ratio | 3-note-ratio | 4-note-ratio\n")
            file.write("-------------------------------------------------------------------------\n")

            for stats in self.file_statistics:
                filename = stats['filename']
                ngrams = stats['ngrams']
                line = f"{filename} | {ngrams.get('2-note', 0)} | {ngrams.get('3-note', 0)} | {ngrams.get('4-note', 0)} | " \
                       f"{ngrams.get('2-note-ratio', 0):.2f} | {ngrams.get('3-note-ratio', 0):.2f} | {ngrams.get('4-note-ratio', 0):.2f}\n"
                file.write(line)
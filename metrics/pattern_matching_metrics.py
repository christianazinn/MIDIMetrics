from pathlib import Path
from typing import Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from symusic import Score
from typing_extensions import override

from scipy.spatial.distance import cosine

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

class ContentPreservationMetric(Metric):

    """
        ContentPreservationMetric class

        Describes how much content it retains from the original. Computed by correlating
        the chroma representation of the generated segment with that of the
        corresponding segment in the source style.
        See https://ismir2018.ismir.net/doc/pdfs/107_Paper.pdf for deeper
        explanation.
    """

    def __init__(self):
        super().__init__()
        self.file_statistics = []
        self.original_chroma_vectors = None
        self.infilled_chroma_vectors = None
        self.compare_with_original = True
        self.analysis_results = {
            'average_difference': None,
            'differences': []
        }

    @override
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):

        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        pitches = np.array([note.pitch for note in track.notes]).astype(int)
        durations = np.array([note.duration for note in track.notes]).astype(int)
        times = np.array([note.time for note in track.notes])

        # 8 time steps for each half bar, as in https://ismir2018.ismir.net/doc/pdfs/107_Paper.pdf
        time_steps_per_bar = 16
        time_steps = ((generation_config.infilled_bars[1] - generation_config.infilled_bars[0])
                      *time_steps_per_bar)

        # Divide the infilling region into time frames
        frame_ticks = np.linspace(infilling_start_ticks, infilling_end_ticks, num=time_steps + 1, endpoint=True)

        # Compute pitches for each time frame
        frame_pitches = []
        for i in range(len(frame_ticks) - 1):
            frame_start = frame_ticks[i]
            frame_end = frame_ticks[i + 1]

            # Find notes that either start in this frame or overlap with it
            notes_in_frame_idxs = np.where(
                (times < frame_end) & (times + durations > frame_start)
            )[0]
            frame_pitches.append(pitches[notes_in_frame_idxs])

        # Convert pitches to chroma vectors for all frames
        chroma_vectors = self._pitches_to_chroma(frame_pitches, time_steps_per_bar)

        # Save chroma vectors based on the input type (original or infilled)
        if kwargs.get("is_original", False):
            self.original_chroma_vectors = chroma_vectors
            # Compute cosine similarity when both chroma matrices have been computed
            self._compute_similarity(generation_config)
        else:
            self.infilled_chroma_vectors = chroma_vectors

    def _compute_similarity(self, generation_config: GenerationConfig):
        # Compute cosine similarity for each time step
        similarities = [
            # RuntimeWarning if a vector in the cosine similarity is all 0
            # because of square root computation(See below)
            1 - cosine(self.original_chroma_vectors[i], self.infilled_chroma_vectors[i])
            for i in range(min(len(self.original_chroma_vectors), len(self.infilled_chroma_vectors)))
        ]

        # So we set the similarity in that case to 0. Compatible with
        # cosine similarity definition (if a vector is 0 similarity is 0)
        similarities = [0.0 if np.isnan(sim) else sim for sim in similarities]

        # Compute and return the average similarity as the metric
        average_similarity = np.mean(similarities)

        self.file_statistics.append({
            "filename": generation_config.filename,
            "content_preservation_score": average_similarity,
            "similarities": similarities
        })

    @override
    def analysis(self):
        """
        Analyze the content preservation scores across all files and compute statistics.

        Returns:
            A dictionary containing the average content preservation score, minimum score,
            maximum score, and a list of all scores for further analysis.
        """
        scores = [stats["content_preservation_score"] for stats in self.file_statistics]

        if not scores:
            self.analysis_results = {
                "average_score": None,
                "min_score": None,
                "max_score": None,
                "scores": []
            }
            return

        self.analysis_results = {
            "average_score": np.mean(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "scores": scores
        }

        return self.analysis_results

    def _pitches_to_chroma(self, frame_pitches, time_steps_per_bar):
        """
        Convert a list of pitches per frame to smoothed chroma vectors.

        Args:
            frame_pitches: A list of lists, where each inner list contains the pitches active in a specific time frame.
            time_steps_per_bar: Number of time steps per bar.

        Returns:
            A 2D numpy array where each row represents the smoothed chroma vector for a frame.
        """
        num_pitches = 12  # 12 pitch classes in a chroma representation

        # Initialize chroma vectors for all frames
        chroma_vectors = []
        for frame in frame_pitches:
            # Initialize a chroma vector for the frame
            chroma_vector = np.zeros(num_pitches)
            for pitch in frame:
                chroma_vector[pitch % num_pitches] += 1
            chroma_vectors.append(chroma_vector)

        chroma_vectors = np.array(chroma_vectors)

        # Normalize chroma vectors (frame-wise)
        chroma_vectors = chroma_vectors / (np.linalg.norm(chroma_vectors, axis=1, keepdims=True) + 1e-8)

        # Compute moving average based on dataset frame size
        frame_size = time_steps_per_bar // 2  # From the reference, smoothing uses half-bar frames
        smoothed_chroma = np.array([
            np.mean(chroma_vectors[max(0, i - frame_size // 2):i + frame_size // 2], axis=0)
            if len(chroma_vectors[max(0, i - frame_size // 2):i + frame_size // 2]) > 0 else np.zeros(num_pitches)
            for i in range(chroma_vectors.shape[0])
        ])

        return smoothed_chroma

    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "ContentPreservationMetric"
        output_folder.mkdir(parents=True, exist_ok=True)
        self.plot(output_folder)
        self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Outputs the filename, track context stats, and infilling stats to a text file.
        """

        average_score = self.analysis_results["average_score"]
        min_score = self.analysis_results["min_score"]
        max_score = self.analysis_results["max_score"]


        with open(output_folder / "content_preservation.txt", 'w') as file:
            file.write("Summary Statistics:\n")
            file.write(f"Average Score: {average_score:.2f}\n" if average_score is not None else "Average Score: N/A\n")
            file.write(f"Minimum Score: {min_score:.2f}\n" if min_score is not None else "Minimum Score: N/A\n")
            file.write(f"Maximum Score: {max_score:.2f}\n" if max_score is not None else "Maximum Score: N/A\n")
            file.write("\n")

            file.write(
                "Filename | content_preservation_score | similarities \n")

            for stats in self.file_statistics:
                filename = stats['filename']
                content_preservation_score = stats['content_preservation_score']
                similarities = stats['similarities']

                line = f"{filename} | " \
                       f"{content_preservation_score:.2f} | " \
                       f"{similarities}\n"

                file.write(line)

    def plot(self, output_folder: Path | str):
        """
        Plot smoothed chroma vectors for both original and infilled data.

        Args:
            original_chroma_vectors: 2D numpy array of original chroma vectors (time steps x pitch classes).
            infilled_chroma_vectors: 2D numpy array of infilled chroma vectors (time steps x pitch classes).
            output_folder: Directory where the plot will be saved.
        """

        # Set up the plot
        plt.figure(figsize=(16, 8))

        # Plot original chroma vectors
        plt.subplot(2, 1, 1)
        plt.imshow(self.original_chroma_vectors.T, aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Intensity")
        plt.yticks(ticks=np.arange(12), labels=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        plt.xlabel("Time Steps")
        plt.ylabel("Pitch Classes")
        plt.title("Original Chroma Vectors")

        # Plot infilled chroma vectors
        plt.subplot(2, 1, 2)
        plt.imshow(self.infilled_chroma_vectors.T, aspect='auto', origin='lower', cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label="Intensity")
        plt.yticks(ticks=np.arange(12), labels=["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
        plt.xlabel("Time Steps")
        plt.ylabel("Pitch Classes")
        plt.title("Infilled Chroma Vectors")

        plt.tight_layout()
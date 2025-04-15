from pathlib import Path
from typing import Set, Tuple

import numpy as np
from matplotlib import pyplot as plt
from symusic import Score
from typing_extensions import override

from scipy.spatial.distance import cosine

from classes.generation_config import GenerationConfig
from classes.metric import Metric

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
            'std_score': np.std(scores),
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

        global_output_file = output_folder.parent / "summary.txt"

        mean = self.analysis_results["average_score"]
        std_dev = self.analysis_results["std_score"]


        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"CP: mean: {mean:.5f} std_dev: {std_dev:.5f}\n")

        #self.plot(output_folder)
        #self.output_to_txt(output_folder)

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

        # Save the plot
        output_path = output_folder / "chroma_vectors_plot.png"
        plt.savefig(output_path, dpi=300)
        plt.close()  # Close the figure to free memory

        print(f"Plot saved to {output_path}")

class F1Onsets(Metric):

    def __init__(self):
        super().__init__()
        self.file_statistics = []
        self.compare_with_original = True
        self.analysis_results = {
            'average_difference': None
        }

    @override
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)

        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]

        notes = [note for note in track.notes
                 if infilling_start_ticks <= note.time < infilling_end_ticks]

        pitches = np.array([note.pitch for note in notes]).astype(int)
        times = np.array([note.time - infilling_start_ticks for note in notes])

        max_offset_ticks = score.tpq//8

        if kwargs.get("is_original", False):
            self.original_notes = np.column_stack((pitches, times))
            # Compute cosine similarity when both chroma matrices have been computed
            self._compute_F1(generation_config, max_offset_ticks)
        else:
            self.infilled_notes = np.column_stack((pitches, times))

    def _compute_F1(self, generation_config, max_offset_ticks):
        # Create a matrix of pitch comparisons (y-axis pitches of notes from original section,
        # x-axis pitches from infilled section)
        pitch_matches = self.original_notes[:, 0, np.newaxis] == self.infilled_notes[:, 0]

        # Create a matrix of tick differences (y-axis onsets of notes from original section,
        # x-axis onsets from infilled section)
        tick_diffs = np.abs(self.original_notes[:, 1, np.newaxis] - self.infilled_notes[:, 1])

        # Combine both conditions
        valid_matches = np.logical_and(pitch_matches, tick_diffs <= max_offset_ticks)

        matches = np.sum(valid_matches)


        onset_precision = float(matches) / len(self.infilled_notes)
        onset_recall = float(matches) / len(self.original_notes)

        if onset_precision == 0 and onset_recall == 0:
            f_score = 0
        else:
            f_score = 2 * onset_precision * onset_recall / (onset_precision + onset_recall)

        self.file_statistics.append({
            "filename": generation_config.filename,
            "f_score": f_score,
        })

    @override
    def analysis(self):
        """
        Computes the average, minimum, and maximum F1 scores across all processed files.
        """
        if not self.file_statistics:
            return

        # Extract all f_scores from file_statistics
        f_scores = [stats['f_score'] for stats in self.file_statistics if 'f_score' in stats]

        if f_scores:
            self.analysis_results = {
                'average_f_score': np.mean(f_scores),
                'min_f_score': np.min(f_scores),
                'max_f_score': np.max(f_scores),
                'std_f_score': np.std(f_scores)
            }
        else:
            self.analysis_results = {
                'average_f_score': None,
                'min_f_score': None,
                'max_f_score': None,
                'std_f_score': None
            }

    @override
    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "F1Onsets"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = self.analysis_results['average_f_score']
        std_dev = self.analysis_results['std_f_score']

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"F1: mean: {mean_diff:.4f}, std.dev: {std_dev:.4f}\n")

        #self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Outputs the F1 score results to a text file including summary statistics and per-file metrics.
        """
        average_f_score = self.analysis_results["average_f_score"]
        min_f_score = self.analysis_results["min_f_score"]
        max_f_score = self.analysis_results["max_f_score"]
        std_f_score = self.analysis_results["std_f_score"]

        with open(output_folder / "f1_onsets.txt", 'w') as file:
            file.write("Summary Statistics:\n")
            file.write(
                f"Average F1 Score: {average_f_score:.4f}\n" if average_f_score is not None else "Average F1 Score: N/A\n")
            file.write(
                f"Minimum F1 Score: {min_f_score:.4f}\n" if min_f_score is not None else "Minimum F1 Score: N/A\n")
            file.write(
                f"Maximum F1 Score: {max_f_score:.4f}\n" if max_f_score is not None else "Maximum F1 Score: N/A\n")
            file.write(
                f"Standard Deviation: {std_f_score:.4f}\n" if std_f_score is not None else "Standard Deviation: N/A\n")
            file.write("\n")

            file.write("Filename | F1 Score\n")
            file.write("-" * 50 + "\n")

            for stats in self.file_statistics:
                filename = stats['filename']
                f_score = stats['f_score']

                line = f"{filename} | {f_score:.4f}\n"
                file.write(line)


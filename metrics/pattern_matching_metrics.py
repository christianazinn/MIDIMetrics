from pathlib import Path
from typing import Set, Tuple

import re
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
    def analysis(self, comparison="compare"):
        """
        Analyze the content preservation scores across all files and compute statistics.

        Returns:
            A dictionary containing the average content preservation score, minimum score,
            maximum score, and a list of all scores for further analysis.
        """
        scores = [stats["content_preservation_score"] for stats in self.file_statistics]

        # Organize scores by model and prompt
        base_scores = {}
        comparison_scores = {}
        
        for stats in self.file_statistics:
            filename = stats['filename']
            f_score = stats['content_preservation_score']
            
            # Extract the prompt identifier using regex to handle varying generationtime values
            # Format: 277_track0_infill_bars24_26_context_8_generationtime_0.971_comparison.mid
            # We want to extract everything before "_generationtime_"
            match = re.match(r'^(.+?)_generationtime_.+?(base|epoch\d+)\.mid$', filename)
            
            if match:
                prompt_id = match.group(1)
                model_type = match.group(2)
                
                if model_type == comparison:
                    comparison_scores[prompt_id] = f_score
                else:
                    base_scores[prompt_id] = f_score

        # Find common prompts for paired analysis
        common_prompts = sorted(set(base_scores.keys()) & set(comparison_scores.keys()))

        if not common_prompts:
            self.analysis_results = {
                "average_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
                'std_score': np.std(scores),
                "scores": scores,
                "paired_analysis": {
                    "paired_test_performed": False,
                    "reason": "No common prompts found between base and comparison models"
                }
            }
            return self.analysis_results
        
        # Create paired arrays
        base_paired = [base_scores[prompt] for prompt in common_prompts]
        comparison_paired = [comparison_scores[prompt] for prompt in common_prompts]

        print(f"number of pairs: {len(base_paired)}")
        
        # Calculate differences for Wilcoxon signed-rank test
        differences = [comparison - base for base, comparison in zip(base_paired, comparison_paired)]
        
        # Perform statistical tests
        from scipy import stats
        
        # Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        try:
            w_stat, w_p_value = stats.wilcoxon(differences)
        except ValueError:  # Can happen with zero differences
            w_stat, w_p_value = None, None
        
        # Calculate effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Overall model statistics
        base_mean = np.mean(base_paired)
        base_std = np.std(base_paired)
        comparison_mean = np.mean(comparison_paired)
        comparison_std = np.std(comparison_paired)
        
        # Prepare paired analysis results
        paired_analysis = {
            "paired_test_performed": True,
            "n_pairs": len(common_prompts),
            "common_prompts": common_prompts,
            "base_mean": base_mean,
            "base_std": base_std,
            "comparison_mean": comparison_mean,
            "comparison_std": comparison_std,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_value": w_p_value,
            "cohens_d": cohens_d,
            "paired_data": {
                prompt: {"base": base_scores[prompt], "comparison": comparison_scores[prompt]}
                for prompt in common_prompts
            },
        }
        
        # Store overall statistics along with paired analysis
        self.analysis_results = {
            "average_score": np.mean(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            'std_score': np.std(scores),
            "scores": scores,
            "paired_analysis": paired_analysis
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
        
        paired_analysis = self.analysis_results.get("paired_analysis", {})
        paired_test_performed = paired_analysis.get("paired_test_performed", False)

        with open(output_folder / "content_preservation.txt", 'w') as file:
            file.write("Summary Statistics:\n")
            file.write(f"Average Score: {average_score:.2f}\n" if average_score is not None else "Average Score: N/A\n")
            file.write(f"Minimum Score: {min_score:.2f}\n" if min_score is not None else "Minimum Score: N/A\n")
            file.write(f"Maximum Score: {max_score:.2f}\n" if max_score is not None else "Maximum Score: N/A\n")
            file.write("\n")
            
            # Add paired analysis results if performed
            if paired_test_performed:
                file.write("Paired Statistical Analysis (Base vs comparison):\n")
                file.write(f"Number of paired samples: {paired_analysis['n_pairs']}\n")
                file.write(f"Base model mean: {paired_analysis['base_mean']:.4f} (std: {paired_analysis['base_std']:.4f})\n")
                file.write(f"comparison model mean: {paired_analysis['comparison_mean']:.4f} (std: {paired_analysis['comparison_std']:.4f})\n")
                file.write(f"Mean difference (comparison - Base): {paired_analysis['mean_difference']:.4f} (std: {paired_analysis['std_difference']:.4f})\n")
                file.write("\n")
                
                # Only include Wilcoxon test results if available
                if paired_analysis.get('wilcoxon_p_value') is not None:
                    file.write("Wilcoxon signed-rank test (non-parametric):\n")
                    file.write(f"W-statistic: {paired_analysis['wilcoxon_statistic']}\n")
                    file.write(f"p-value: {paired_analysis['wilcoxon_p_value']:.6f}\n")
                    file.write("\n")
                
                file.write(f"Effect size (Cohen's d): {paired_analysis['cohens_d']:.4f}\n")
            else:
                if "reason" in paired_analysis:
                    file.write(f"Paired statistical analysis not performed: {paired_analysis['reason']}\n\n")
                else:
                    file.write("Paired statistical analysis not performed\n\n")

            file.write("Individual File Statistics:\n")
            file.write("Filename | content_preservation_score | similarities\n")
            file.write("-" * 80 + "\n")

            for stats in self.file_statistics:
                filename = stats['filename']
                content_preservation_score = stats['content_preservation_score']
                # Only print first 3 and last 3 similarities to keep the output reasonable
                similarities = stats['similarities']
                if len(similarities) > 6:
                    similarities_str = f"{similarities[:3]}...{similarities[-3:]}"
                else:
                    similarities_str = str(similarities)

                line = f"{filename} | {content_preservation_score:.4f} | {similarities_str}\n"
                file.write(line)
            
            # Add paired comparison table if performed
            if paired_test_performed:
                file.write("\n\nPaired Comparison Table:\n")
                file.write("Prompt | Base Score | comparison Score | Difference (comparison - Base)\n")
                file.write("-" * 80 + "\n")
                
                paired_data = paired_analysis["paired_data"]
                for prompt in paired_analysis["common_prompts"]:
                    base_score = paired_data[prompt]["base"]
                    comparison_score = paired_data[prompt]["comparison"]
                    difference = comparison_score - base_score
                    file.write(f"{prompt} | {base_score:.4f} | {comparison_score:.4f} | {difference:.4f}\n")
        
        # Write summary to global output file
        with open(output_folder.parent / "summary.txt", mode='a', encoding='utf-8') as f:
            f.write(f"CP: mean: {average_score:.5f} std_dev: {self.analysis_results['std_score']:.5f}\n")

            if paired_test_performed:
                f.write(f"CP Paired Analysis: base_mean: {paired_analysis['base_mean']:.5f} comparison_mean: {paired_analysis['comparison_mean']:.5f}\n")
                f.write(f"CP Effect Size: Cohen's d: {paired_analysis['cohens_d']:.5f} ({effect_size_interp})\n")
            if paired_analysis.get("wilcoxon_p_value", False):
                f.write(f"CP Wilcoxon P: {paired_analysis['wilcoxon_p_value']}\n")

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
    def analysis(self, comparison="compare"):
        """
        Computes the average, minimum, and maximum F1 scores across all processed files.
        Performs paired statistical analysis between base and comparison models.
        """
        if not self.file_statistics:
            self.analysis_results = {
                'average_f_score': None,
                'min_f_score': None,
                'max_f_score': None,
                'std_f_score': None
            }
            return self.analysis_results
        
        # Extract all f_scores from file_statistics
        f_scores = [stats['f_score'] for stats in self.file_statistics if 'f_score' in stats]
        
        # Extract consistently the prompt identifier from filenames
        base_scores = {}
        comparison_scores = {}
        
        import re
        
        for stats in self.file_statistics:
            filename = stats['filename']
            f_score = stats['f_score']
            
            # Extract the prompt identifier using regex to handle varying generationtime values
            # Format: 277_track0_infill_bars24_26_context_8_generationtime_0.971_comparison.mid
            # We want to extract everything before "_generationtime_"
            match = re.match(r'^(.+?)_generationtime_.+?(base|epoch\d+)\.mid$', filename)
            
            if match:
                prompt_id = match.group(1)
                model_type = match.group(2)
                
                if model_type == comparison:
                    comparison_scores[prompt_id] = f_score
                else:
                    base_scores[prompt_id] = f_score
        
        # Find common prompts for paired analysis
        common_prompts = sorted(set(base_scores.keys()) & set(comparison_scores.keys()))
        
        if not common_prompts:
            self.analysis_results = {
                'average_f_score': np.mean(f_scores) if f_scores else None,
                'min_f_score': np.min(f_scores) if f_scores else None,
                'max_f_score': np.max(f_scores) if f_scores else None,
                'std_f_score': np.std(f_scores) if f_scores else None,
            }
            return self.analysis_results
        
        # Create paired arrays
        base_paired = [base_scores[prompt] for prompt in common_prompts]
        comparison_paired = [comparison_scores[prompt] for prompt in common_prompts]
        
        # Calculate differences for Wilcoxon signed-rank test
        differences = [comparison - base for base, comparison in zip(base_paired, comparison_paired)]
        
        # Perform statistical tests
        from scipy import stats
        
        try:
            w_stat, w_p_value = stats.wilcoxon(differences)
        except ValueError:  # Can happen with zero differences
            w_stat, w_p_value = None, None
        
        # Calculate effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0
        
        # Overall model statistics
        base_mean = np.mean(base_paired)
        base_std = np.std(base_paired)
        comparison_mean = np.mean(comparison_paired)
        comparison_std = np.std(comparison_paired)
        
        # Prepare paired analysis results
        paired_analysis = {
            "paired_test_performed": True,
            "n_pairs": len(common_prompts),
            "common_prompts": common_prompts,
            "base_mean": base_mean,
            "base_std": base_std,
            "comparison_mean": comparison_mean,
            "comparison_std": comparison_std,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "wilcoxon_statistic": w_stat,
            "wilcoxon_p_value": w_p_value,
            "cohens_d": cohens_d,
            "paired_data": {
                prompt: {"base": base_scores[prompt], "comparison": comparison_scores[prompt]}
                for prompt in common_prompts
            },
        }
        
        # Store overall statistics along with paired analysis
        self.analysis_results = {
            'average_f_score': np.mean(f_scores) if f_scores else None,
            'min_f_score': np.min(f_scores) if f_scores else None,
            'max_f_score': np.max(f_scores) if f_scores else None,
            'std_f_score': np.std(f_scores) if f_scores else None,
            "paired_analysis": paired_analysis
        }
        
        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        output_folder = Path(output_folder) / "F1Onsets"
        output_folder.mkdir(parents=True, exist_ok=True)

        self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Outputs the F1 score results to a text file including summary statistics and per-file metrics.
        Includes paired statistical analysis between base and comparison models.
        """
        average_f_score = self.analysis_results.get("average_f_score")
        min_f_score = self.analysis_results.get("min_f_score")
        max_f_score = self.analysis_results.get("max_f_score")
        std_f_score = self.analysis_results.get("std_f_score")
        
        paired_analysis = self.analysis_results.get("paired_analysis", {})
        paired_test_performed = paired_analysis.get("paired_test_performed", False)

        with open(output_folder / "f1_onsets.txt", 'w') as file:
            file.write("Summary Statistics:\n")
            file.write(f"Average F1 Score: {average_f_score:.4f}\n" if average_f_score is not None else "Average F1 Score: N/A\n")
            file.write(f"Minimum F1 Score: {min_f_score:.4f}\n" if min_f_score is not None else "Minimum F1 Score: N/A\n")
            file.write(f"Maximum F1 Score: {max_f_score:.4f}\n" if max_f_score is not None else "Maximum F1 Score: N/A\n")
            file.write(f"Standard Deviation: {std_f_score:.4f}\n" if std_f_score is not None else "Standard Deviation: N/A\n")
            file.write("\n")
            
            # Add paired analysis results if performed
            if paired_test_performed:
                file.write("Paired Statistical Analysis (Base vs comparison):\n")
                file.write(f"Number of paired samples: {paired_analysis['n_pairs']}\n")
                file.write(f"Base model mean: {paired_analysis['base_mean']:.4f} (std: {paired_analysis['base_std']:.4f})\n")
                file.write(f"comparison model mean: {paired_analysis['comparison_mean']:.4f} (std: {paired_analysis['comparison_std']:.4f})\n")
                file.write(f"Mean difference (comparison - Base): {paired_analysis['mean_difference']:.4f} (std: {paired_analysis['std_difference']:.4f})\n")
                file.write("\n")
                
                # Only include Wilcoxon test results if available
                if paired_analysis.get('wilcoxon_p_value') is not None:
                    file.write("Wilcoxon signed-rank test (non-parametric):\n")
                    file.write(f"W-statistic: {paired_analysis['wilcoxon_statistic']}\n")
                    file.write(f"p-value: {paired_analysis['wilcoxon_p_value']:.6f}\n")
                
                file.write(f"Effect size (Cohen's d): {paired_analysis['cohens_d']:.4f}\n")
                
            else:
                if "reason" in paired_analysis:
                    file.write(f"Paired statistical analysis not performed: {paired_analysis['reason']}\n\n")
                else:
                    file.write("Paired statistical analysis not performed\n\n")

            file.write("Individual File Statistics:\n")
            file.write("Filename | F1 Score\n")
            file.write("-" * 50 + "\n")

            for stats in self.file_statistics:
                filename = stats['filename']
                f_score = stats['f_score']

                line = f"{filename} | {f_score:.4f}\n"
                file.write(line)
            
            # Add paired comparison table if performed
            if paired_test_performed:
                file.write("\n\nPaired Comparison Table:\n")
                file.write("Prompt | Base Score | comparison Score | Difference (comparison - Base)\n")
                file.write("-" * 80 + "\n")
                
                paired_data = paired_analysis["paired_data"]
                for prompt in paired_analysis["common_prompts"]:
                    base_score = paired_data[prompt]["base"]
                    comparison_score = paired_data[prompt]["comparison"]
                    difference = comparison_score - base_score
                    # Truncate the prompt ID to avoid long lines
                    display_prompt = prompt
                    if len(display_prompt) > 35:
                        display_prompt = display_prompt[:32] + "..."
                    file.write(f"{display_prompt} | {base_score:.4f} | {comparison_score:.4f} | {difference:.4f}\n")
        
        # Write summary to global output file
        with open(output_folder.parent / "summary.txt", mode='a', encoding='utf-8') as f:
            f.write(f"F1: mean: {average_f_score:.4f}, std.dev: {std_f_score:.4f}\n")

            if paired_test_performed:
                f.write(f"F1 Paired Analysis: base_mean: {paired_analysis['base_mean']:.4f} comparison_mean: {paired_analysis['comparison_mean']:.4f}\n")
                f.write(f"F1 Effect Size: Cohen's d: {paired_analysis['cohens_d']:.4f} ({effect_size_interp})\n")
            if paired_analysis.get("wilcoxon_p_value", False):
                f.write(f"F1 Wilcoxon P: {paired_analysis['wilcoxon_p_value']}\n")


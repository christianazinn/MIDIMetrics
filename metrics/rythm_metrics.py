from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
from symusic import Score
from typing_extensions import override

from classes.metric import Metric
from classes.generation_config import GenerationConfig

from classes.constants import STEP, POINT_DIM, MEAN_LINES_WIDTH


class RV(Metric):
    """
        Rhythm Variations class

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
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        durations = np.array([note.duration for note in track.notes
                              if infilling_start_ticks <= note.time < infilling_end_ticks])

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
                'rv_original': self.rv_original,
                'rv_infilled': self.rv_infilled
            })
        else:
            self.rv_infilled = rv

    @override
    def analysis(self):
        """Compute statistics of original and infilled RV values."""
        original_values = [stats['rv_original'] for stats in self.file_statistics
                           if not np.isnan(stats['rv_original'])]
        infilled_values = [stats['rv_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['rv_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "RV"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['average_original'] - self.analysis_results['average_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"RV: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

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

        # Retrieve analysis results
        avg_original = self.analysis_results['average_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['average_infilled']
        std_infilled = self.analysis_results['std_infilled']

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original RV values
        plt.plot(indices, original_values, 'ro', label='Original RV', markersize = POINT_DIM)

        # Plot infilled RV values
        plt.plot(indices, infilled_values, 'bo', label='Infilled RV', markersize = POINT_DIM)

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
        plt.title('Rhythm Variations (RV) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('RV Value (Unique Durations / Total Notes)')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
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
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            file.write("Filename\tRV Original\tRV Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['rv_original']:.4f}\t"
                           f"{stats['rv_infilled']:.4f}\n")

class QR(Metric):
    """
        Qualified Rhythm frequency class

        QR measures how many distinct note durations the model plays within a sequence,
        considering only qualified note durations (from 1/32 up to 4 bar long notes).
        As in https://musicalmetacreation.org/mume2018/proceedings/Trieu.pdf,
        it is computed as the average ratio across all sequences of
        the number of distinct qualified note durations to the total number of notes in the sequence.
        In https://arxiv.org/pdf/1709.06298 every note above 1/32 is considered
        qualified (definition too weak imo).
    """

    def __init__(self):
        super().__init__()
        self.compare_with_original = True
        self.file_statistics = []
        self.qr_original = None
        self.qr_infilled = None
        self.analysis_results = {
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        infilling_start_ticks = window_bars_ticks[generation_config.context_size]
        infilling_end_ticks = window_bars_ticks[-generation_config.context_size - 1]

        track = score.tracks[generation_config.infilled_track_idx]
        durations = np.array([note.duration for note in track.notes
                              if infilling_start_ticks <= note.time < infilling_end_ticks])

        # Define qualified durations 1/32 1/16 1/8 1/4 1/2 1 2 4 8
        max_dur = score.tpq * 2**4 #4 bar long note
        min_dur = score.tpq * 2**(-3) # 1/32 long note

        # Calculate qualified rhythm variations ratio
        if len(durations) == 0:
            qr = np.nan
        else:
            qualified_durations = len([d for d in durations if d >= min_dur and d <= max_dur])
            total_notes = len(durations)
            qr = qualified_durations / total_notes

        if kwargs.get("is_original", False):
            self.qr_original = qr
            self.file_statistics.append({
                'filename': generation_config.filename,
                'qr_original': self.qr_original,
                'qr_infilled': self.qr_infilled
            })
        else:
            self.qr_infilled = qr

    @override
    def analysis(self):
        """Compute statistics of original and infilled QR values."""
        original_values = [stats['qr_original'] for stats in self.file_statistics
                           if not np.isnan(stats['qr_original'])]
        infilled_values = [stats['qr_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['qr_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0

        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "QR"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = abs(self.analysis_results['average_original'] - self.analysis_results['average_infilled'])
        std_dev = ((self.analysis_results['std_original'] ** 2) / len(self.file_statistics) +
                   (self.analysis_results['std_infilled'] ** 2) / len(self.file_statistics)) ** 0.5

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"QR: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")

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

        # Retrieve analysis results
        avg_original = self.analysis_results['average_original']
        std_original = self.analysis_results['std_original']
        avg_infilled = self.analysis_results['average_infilled']
        std_infilled = self.analysis_results['std_infilled']

        # Create plot
        plt.figure(figsize=(10, 6))

        # Plot original QR values
        plt.plot(indices, original_values, 'ro', label='Original QR', markersize = POINT_DIM)

        # Plot infilled QR values
        plt.plot(indices, infilled_values, 'bo', label='Infilled QR', markersize = POINT_DIM)

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
        plt.title('Qualified Rhythm Variations (QR) of Original and Infilled MIDI Files')
        plt.xlabel('MIDI File Index')
        plt.ylabel('QR Value (Qualified Unique Durations / Total Notes)')
        selected_indices = indices[::STEP]  # Select every 100th index
        selected_labels = [f"File number {i}" for i in selected_indices]  # Create labels

        plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)

        # Update legend with mean and std. deviation
        plt.legend()

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(output_folder / "qr_original_vs_infilled.png")
        plt.close()

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the QR values and statistics to a text file.
        """
        output_file = Path(output_folder) / "qr_results.txt"

        with output_file.open(mode='w') as file:
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            file.write("Filename\tQR Original\tQR Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['qr_original']:.4f}\t"
                           f"{stats['qr_infilled']:.4f}\n")

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
        self.compare_with_original = True
        self.file_statistics = []
        self.groove_original = None
        self.groove_infilled = None
        self.analysis_results = {
            'average_original': None,
            'std_original': None,
            'average_infilled': None,
            'std_infilled': None
        }

    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
        window_bars_ticks = kwargs.get('window_bars_ticks', None)
        context_size = generation_config.context_size

        infilled_bars = generation_config.infilled_bars[1] - generation_config.infilled_bars[0]
        track = score.tracks[generation_config.infilled_track_idx]
        times = np.array([note.time for note in track.notes])
        """
        #ticks_per_bar = window_bars_ticks[-1] - window_bars_ticks[-2]
        # Need to compute the maximum between the differences of bar ticks
        # in window_bars_ticks, otherwise line 843 gives IndexOutOfBounds
        ticks_per_bar = max(
            window_bars_ticks[i + 1] - window_bars_ticks[i]
            for i in range(len(window_bars_ticks) - 1)
        )
        # Initialize the grooving pattern matrix
        grooving_pattern_matrix = np.zeros((infilled_bars, ticks_per_bar), dtype=bool)

        # Fill in the grooving pattern matrix for each bar
        for i in range(infilled_bars):
            bar_start = window_bars_ticks[i + context_size]
            bar_end = window_bars_ticks[i + context_size + 1]
            bar_times = times[(times >= bar_start) & (times < bar_end)] - bar_start
            grooving_pattern_matrix[i, bar_times] = 1
        """
        ticks_per_subdivision = score.tpq // 8  # Define the coarser granularity
        ticks_per_bar = max(
            window_bars_ticks[i + 1] - window_bars_ticks[i]
            for i in range(len(window_bars_ticks) - 1)
        )
        subdivisions_per_bar = ticks_per_bar // ticks_per_subdivision  # New matrix width

        # Initialize the grooving pattern matrix with coarser granularity
        grooving_pattern_matrix = np.zeros((infilled_bars, subdivisions_per_bar), dtype=bool)

        # Fill in the grooving pattern matrix for each bar
        for i in range(infilled_bars):
            bar_start = window_bars_ticks[i + context_size]
            bar_end = window_bars_ticks[i + context_size + 1]
            bar_times = times[(times >= bar_start) & (times < bar_end)] - bar_start

            # Convert tick times to coarse subdivisions
            bar_subdivisions = np.unique(bar_times // ticks_per_subdivision)
            grooving_pattern_matrix[i, bar_subdivisions] = 1

        # Compute pairwise grooving pattern similarities between adjacent bars
        hamming_distance = np.count_nonzero(
            grooving_pattern_matrix[:-1] != grooving_pattern_matrix[1:]
        )
        groove_consistency = 1 - hamming_distance / (subdivisions_per_bar * infilled_bars)

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
    def analysis(self, comparison="compare"):
        """
        Compute statistics of original and infilled Groove Consistency values.
        Perform paired statistical analysis between base and comparison models.
        """
        # Basic statistics as before
        original_values = [stats['groove_original'] for stats in self.file_statistics
                           if not np.isnan(stats['groove_original'])]
        infilled_values = [stats['groove_infilled'] for stats in self.file_statistics
                           if not np.isnan(stats['groove_infilled'])]

        self.analysis_results['average_original'] = np.mean(original_values) if original_values else 0
        self.analysis_results['std_original'] = np.std(original_values) if original_values else 0
        self.analysis_results['average_infilled'] = np.mean(infilled_values) if infilled_values else 0
        self.analysis_results['std_infilled'] = np.std(infilled_values) if infilled_values else 0
        
        # Organize scores by model and prompt for paired analysis
        base_scores = {}
        comparison_scores = {}
        
        import re
        
        for stats in self.file_statistics:
            filename = stats['filename']
            
            # Extract the prompt identifier using regex to handle varying generationtime values
            # Format: 277_track0_infill_bars24_26_context_8_generationtime_0.971_comparison.mid
            match = re.match(r'^(.+?)_generationtime_.+?(base|epoch\d+)\.mid$', filename)
            
            if match:
                prompt_id = match.group(1)  # The part before "_generationtime_"
                model_type = match.group(2)  # "base" or "epochX"
                
                # Skip if groove values are NaN
                if np.isnan(stats['groove_original']) or np.isnan(stats['groove_infilled']):
                    continue
                
                # We want to compare infilled values between models
                if model_type == comparison:
                    comparison_scores[prompt_id] = stats['groove_infilled']
                else:
                    base_scores[prompt_id] = stats['groove_infilled']
        
        # Find common prompts for paired analysis
        common_prompts = sorted(set(base_scores.keys()) & set(comparison_scores.keys()))
        
        if not common_prompts:
            self.analysis_results['paired_analysis'] = {
                "paired_test_performed": False,
                "reason": "No common prompts found between base and epoch models"
            }
            return self.analysis_results
        
        # Create paired arrays
        base_paired = [base_scores[prompt] for prompt in common_prompts]
        comparison_paired = [comparison_scores[prompt] for prompt in common_prompts]
        
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
        
        # Add paired analysis to results
        self.analysis_results['paired_analysis'] = paired_analysis
        
        return self.analysis_results

    @override
    def output_results(self, output_folder: Path | str):
        """Output results to files."""
        output_folder = Path(output_folder) / "GrooveConsistency"
        output_folder.mkdir(parents=True, exist_ok=True)

        global_output_file = output_folder.parent / "summary.txt"

        mean_diff = self.analysis_results['average_infilled']
        std_dev = self.analysis_results['std_infilled']
        
        # Add paired analysis results if available
        paired_analysis = self.analysis_results.get('paired_analysis', {})
        paired_test_performed = paired_analysis.get('paired_test_performed', False)

        with global_output_file.open(mode='a', encoding='utf-8') as f:
            f.write(f"GS: mean: {mean_diff:.5f}, std.dev: {std_dev:.5f}\n")
            
            if paired_test_performed:
                f.write(f"GS Paired Analysis: base_mean: {paired_analysis['base_mean']:.5f} comparison_mean: {paired_analysis['comparison_mean']:.5f}\n")
                f.write(f"GS Effect Size: Cohen's d: {paired_analysis['cohens_d']:.5f} ({effect_size_interp})\n")
            if paired_analysis.get("wilcoxon_p_value", False):
                f.write(f"GS Wilcoxon P: {paired_analysis['wilcoxon_p_value']}\n")

        self.plot(output_folder)
        self.output_to_txt(output_folder)

    def output_to_txt(self, output_folder: Path | str):
        """
        Write the Groove Consistency values and statistics to a text file.
        Include paired statistical analysis between base and comparison models.
        """
        output_file = Path(output_folder) / "groove_consistency_results.txt"

        with output_file.open(mode='w') as file:
            # Write statistics
            file.write(f"Original: Average={self.analysis_results['average_original']:.4f}, "
                       f"Std={self.analysis_results['std_original']:.4f}\n")
            file.write(f"Infilled: Average={self.analysis_results['average_infilled']:.4f}, "
                       f"Std={self.analysis_results['std_infilled']:.4f}\n\n")
            
            # Add paired analysis results if performed
            paired_analysis = self.analysis_results.get('paired_analysis', {})
            paired_test_performed = paired_analysis.get('paired_test_performed', False)
            
            if paired_test_performed:
                file.write("\nPaired Statistical Analysis (Base vs comparison):\n")
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
                
            # Write individual file results
            file.write("Individual File Results:\n")
            file.write("Filename\tGroove Original\tGroove Infilled\n")

            for stats in self.file_statistics:
                file.write(f"{stats['filename']}\t"
                           f"{stats['groove_original']:.4f}\t"
                           f"{stats['groove_infilled']:.4f}\n")
            
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
    
    def plot(self, output_folder: Path | str):
        """
        Plot Groove Consistency values for original and infilled MIDI files.
        Add additional plots for paired model comparison.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Extract data, filtering out NaN values
            valid_stats = [(stats['groove_original'], stats['groove_infilled'], stats['filename'])
                          for stats in self.file_statistics
                          if not (np.isnan(stats['groove_original']) or np.isnan(stats['groove_infilled']))]

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

            # Create plot for original vs infilled
            plt.figure(figsize=(10, 6))

            # Plot original Groove Consistency values
            plt.plot(indices, original_values, 'ro', label='Original Groove Consistency', markersize=5)

            # Plot infilled Groove Consistency values
            plt.plot(indices, infilled_values, 'bo', label='Infilled Groove Consistency', markersize=5)

            # Add connecting lines between original and infilled points
            for i in range(len(indices)):
                plt.plot([indices[i], indices[i]],
                        [original_values[i], infilled_values[i]],
                        'k--', lw=0.2)

            # Plot mean and standard deviation for original values
            plt.axhline(avg_original, color='r', linestyle='-', linewidth=1, label=f'Mean Original ({avg_original:.4f})')
            plt.fill_between(indices, avg_original - std_original, avg_original + std_original, color='r', alpha=0.2, label='Std Dev Original')

            # Plot mean and standard deviation for infilled values
            plt.axhline(avg_infilled, color='b', linestyle='-', linewidth=1, label=f'Mean Infilled ({avg_infilled:.4f})')
            plt.fill_between(indices, avg_infilled - std_infilled, avg_infilled + std_infilled, color='b', alpha=0.2, label='Std Dev Infilled')

            # Annotate plot
            plt.title('Groove Consistency of Original and Infilled MIDI Files')
            plt.xlabel('MIDI File Index')
            plt.ylabel('Groove Consistency')
            
            # Handle axis ticks
            if len(indices) > 20:
                step = max(1, len(indices) // 20)
                selected_indices = indices[::step]
                selected_labels = [f"{i}" for i in selected_indices]
                plt.xticks(selected_indices, selected_labels, rotation=45, ha='right', fontsize=8)
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plt.savefig(Path(output_folder) / "groove_consistency_original_vs_infilled.png")
            plt.close()
            
            # Additional plots for paired analysis
            paired_analysis = self.analysis_results.get('paired_analysis', {})
            paired_test_performed = paired_analysis.get('paired_test_performed', False)
            
            if paired_test_performed and paired_analysis.get("n_pairs", 0) > 0:
                # Extract paired data
                common_prompts = paired_analysis["common_prompts"]
                paired_data = paired_analysis["paired_data"]
                
                base_scores = [paired_data[prompt]["base"] for prompt in common_prompts]
                comparison_scores = [paired_data[prompt]["comparison"] for prompt in common_prompts]
                
                # Plot 1: Comparison of base vs comparison scores
                plt.figure(figsize=(10, 6))
                
                # Box plot comparison
                data = {
                    'Base': base_scores,
                    'comparison': comparison_scores
                }
                sns.boxplot(data=data)
                
                # Add individual data points
                sns.stripplot(data=data, color='black', alpha=0.5, jitter=True)
                
                plt.title("Groove Consistency Comparison Between Models")
                plt.ylabel("Groove Consistency")
                
                # Add statistical annotation
                p_value = paired_analysis["t_p_value"]
                sig_text = f"p = {p_value:.6f}" + (" *" if p_value < 0.05 else "")
                plt.annotate(sig_text, xy=(0.5, 0.95), xycoords='axes fraction', 
                            ha='center', va='center', fontsize=12, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(Path(output_folder) / "groove_base_vs_comparison_comparison.png")
                plt.close()
                
                # Plot 2: Paired differences
                differences = [comparison_scores[i] - base_scores[i] for i in range(len(base_scores))]
                
                plt.figure(figsize=(10, 6))
                sns.histplot(differences, kde=True)
                plt.axvline(x=0, color='k', linestyle='--', label="No difference")
                plt.axvline(x=np.mean(differences), color='r', linestyle='-', 
                            label=f"Mean difference: {np.mean(differences):.3f}")
                plt.title("Distribution of Groove Consistency Differences (comparison - Base)")
                plt.xlabel("Groove Consistency Difference")
                plt.ylabel("Frequency")
                plt.legend()
                plt.tight_layout()
                plt.savefig(Path(output_folder) / "groove_differences_histogram.png")
                plt.close()
                
                # Plot 3: Scatter plot with diagonal line
                plt.figure(figsize=(8, 8))
                plt.scatter(base_scores, comparison_scores, alpha=0.7)
                
                # Add diagonal line (y=x)
                min_val = min(min(base_scores), min(comparison_scores))
                max_val = max(max(base_scores), max(comparison_scores))
                padding = (max_val - min_val) * 0.05
                plot_min = max(0, min_val - padding)  # Groove consistency should be non-negative
                plot_max = max_val + padding
                plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', alpha=0.5)
                
                plt.title("Base vs. comparison Groove Consistency")
                plt.xlabel("Base Model Groove Consistency")
                plt.ylabel("comparison Model Groove Consistency")
                
                # Count points above/below diagonal
                above_diagonal = sum(comparison > base for comparison, base in zip(comparison_scores, base_scores))
                below_diagonal = sum(comparison < base for comparison, base in zip(comparison_scores, base_scores))
                equal = sum(comparison == base for comparison, base in zip(comparison_scores, base_scores))
                
                # Add annotation about points above/below diagonal
                plt.annotate(f"comparison > Base: {above_diagonal} samples\ncomparison < Base: {below_diagonal} samples\nEqual: {equal} samples", 
                            xy=(0.05, 0.95), xycoords='axes fraction', 
                            ha='left', va='top', fontsize=10, 
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(Path(output_folder) / "groove_scatter_comparison.png")
                plt.close()
                
        except Exception as e:
            print(f"Error during plotting: {e}")
            # Continue execution even if plotting fails
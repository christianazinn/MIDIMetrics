import os
import time
from pathlib import Path
import re
import miditok
import numpy as np
from miditok.utils import get_bars_ticks
from numpy.polynomial.polynomial import Polynomial
import traceback
from classes.constants import OUTPUT_DIR, ORIGINAL_MIDIFILES_DIR
from classes.metric import Metric
from classes.metric_config import MetricConfig
from metrics.pattern_matching_metrics import ContentPreservationMetric, F1Onsets
from metrics.pitch_metrics import UPC, PR, Polyphony, PV, \
    PitchClassHistogramEntropy

from symusic import Score

from metrics.rythm_metrics import RV, QR, GrooveConsistency

from classes.generation_config import parse_filename, GenerationConfig

class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig
    output_dir : str | Path

    def __init__(self, metric_config:MetricConfig, output_dir: str | Path):
        self.metric_config = metric_config
        self.output_dir = output_dir

        self.metrics=[]
        if metric_config.content_preservation:
            self.metrics.append(ContentPreservationMetric())
        if metric_config.upc:
            self.metrics.append(UPC())
        if metric_config.pr:
            self.metrics.append(PR())
        if metric_config.polyphony:
            self.metrics.append(Polyphony())
        if metric_config.pv:
            self.metrics.append(PV())
        if metric_config.rv:
            self.metrics.append(RV())
        if metric_config.qr:
            self.metrics.append(QR())
        if metric_config.groove_consistency:
            self.metrics.append(GrooveConsistency())
        if metric_config.pitch_class_histogram_entropy:
            self.metrics.append(PitchClassHistogramEntropy())
        if metric_config.f1onsets:
            self.metrics.append(F1Onsets())

    def compute_metrics(self, midi_files: list[str | Path], MUSIAC_ORIGINAL_MIDIFILES_DIR=None, musiac = False):

        start_time = time.time()

        error_number = 0

        for midi_file in midi_files:

            infilled_score = Score(midi_file)

            _generation_config = parse_filename(midi_file)

            song_name = f"{str(midi_file.stem).split('_')[0]}.mid"


            if musiac:

                pattern = re.compile(r"(miditest\d+)_track\d+_infill_bars\d+_\d+_orig_bars(\d+)_(\d+)")
                match = pattern.match(midi_file.stem)
                if match:
                    test_id = match.group(1)
                    orig_bars_1, orig_bars_2 = match.group(2), match.group(3)

                    # Construct the new filename
                    matched_file = f"{test_id}_orig_bars{orig_bars_1}_{orig_bars_2}.mid"
                    original_score = Score(MUSIAC_ORIGINAL_MIDIFILES_DIR / matched_file)
                else:
                    error_number+=1
                    return

            else:

                original_score = Score(ORIGINAL_MIDIFILES_DIR / song_name)

            # Removing empty tracks
            for idx in range(len(original_score.tracks) - 1, -1, -1):
                if original_score.tracks[idx].note_num() == 0:
                    del original_score.tracks[idx]


            # Notice that _window_bars_ticks contains also the tick of the
            # bar right after the end of the context. This is donce for
            # indices purposes
            _window_bars_ticks_infilled = self._get_window_bars_ticks(_generation_config, infilled_score)

            if _window_bars_ticks_infilled is None:
                msg = ("[ERROR] MetricsProcessor::compute_metrics Couldn't compute"
                       f" bars ticks values for midi file: {midi_file}")
                raise ValueError(msg)


            for metric in self.metrics:
                try:
                    metric.compute_metric(
                        generation_config=_generation_config,
                        score = infilled_score,
                        window_bars_ticks = _window_bars_ticks_infilled,
                        is_original = False)

                    if metric.compare_with_original:
                        _window_bars_ticks_original = self._get_window_bars_ticks(
                            _generation_config,
                            original_score
                        )
                        metric.compute_metric(
                            generation_config = _generation_config,
                            score = original_score,
                            window_bars_ticks = _window_bars_ticks_original,
                            is_original = True)
                except:
                    error_number += 1
                    #traceback.print_exc()
                    continue


        end_time = time.time()

        print(f"Time to compute metrics: {end_time - start_time} seconds")
        print(f"errors_{error_number}")
        print(f"error rate{error_number/len(midi_files)}")
        # print(f"Time in re: {total_time_re} seconds")

        for metric in self.metrics:
            metric.analysis()
            metric.output_results(OUTPUT_DIR / self.output_dir)

    def _get_window_bars_ticks(self, generation_config: GenerationConfig, score: Score):
        bars_ticks = np.array(get_bars_ticks(score))

        infilling_start_idx = generation_config.infilled_bars[0]
        infilling_end_idx = generation_config.infilled_bars[1]
        
        if infilling_end_idx >= len(bars_ticks):
            intermediate = bars_ticks[infilling_start_idx - generation_config.context_size
                       :infilling_end_idx + generation_config.context_size]
            return np.append(intermediate, bars_ticks[-1] + bars_ticks[-1] - bars_ticks[-2])
        else:
            return bars_ticks[infilling_start_idx - generation_config.context_size
                              :infilling_end_idx + generation_config.context_size + 1]





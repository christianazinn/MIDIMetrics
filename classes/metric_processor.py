import time
from pathlib import Path

import miditok
import numpy as np
from miditok.utils import get_bars_ticks
from numpy.polynomial.polynomial import Polynomial

from classes.constants import OUTPUT_DIR, ORIGINAL_MIDIFILES_DIR
from classes.metric import Metric
from classes.metric_config import MetricConfig
from metrics.pattern_matching_metrics import NGramsRepetitions, ContentPreservationMetric
from metrics.pitch_metrics import BarPitchVarietyMetric, BarAbsolutePitchesMetric, UPC, PR, Polyphony, PV, \
    PitchClassHistogramEntropy

from symusic import Score

from metrics.rythm_metrics import BarNoteDensityMetric, NoteDurationsFrequencyMetric, \
    NoteDurationsSetMetric, PolyphonyMinMaxMetric, RV, QR, GrooveConsistency

from classes.generation_config import parse_filename, GenerationConfig

class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig

    def __init__(self, metric_config:MetricConfig):
        self.metric_config = metric_config

        self.metrics=[]
        if metric_config.bar_absolute_pitches:
            self.metrics.append(BarAbsolutePitchesMetric())
        if metric_config.bar_pitch_variety:
            self.metrics.append(BarPitchVarietyMetric())
        if metric_config.bar_note_density:
            self.metrics.append(BarNoteDensityMetric())
        if metric_config.note_durations_set:
            self.metrics.append(NoteDurationsSetMetric())
        if metric_config.note_durations_frequency:
            self.metrics.append(NoteDurationsFrequencyMetric())
        if metric_config.ngrams_repetitions:
            self.metrics.append(NGramsRepetitions())
        if metric_config.polyphony_min_max:
            self.metrics.append(PolyphonyMinMaxMetric())
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

    def compute_metrics(self, midi_files: list[str | Path]):
        #total_time_re = 0
        start_time = time.time()

        for midi_file in midi_files:
            infilled_score = Score(midi_file)

            _generation_config = parse_filename(midi_file)


            song_name = f"{str(midi_file.stem).split('_')[0]}.mid"
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
                    pass

        end_time = time.time()

        print(f"Time to compute metrics: {end_time - start_time} seconds")
        # print(f"Time in re: {total_time_re} seconds")

        for metric in self.metrics:
            metric.analysis()
            metric.output_results(OUTPUT_DIR)

    def _get_window_bars_ticks(self, generation_config: GenerationConfig, score: Score):
        bars_ticks = np.array(get_bars_ticks(score))

        infilling_start_idx = generation_config.infilled_bars[0]
        infilling_end_idx = generation_config.infilled_bars[1]

        return bars_ticks[infilling_start_idx - generation_config.context_size
                          :infilling_end_idx + generation_config.context_size + 1]





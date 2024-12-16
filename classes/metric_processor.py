import time
from abc import abstractmethod
from pathlib import Path

import numpy as np
from miditok.utils import get_bars_ticks
from numpy.f2py.crackfortran import analyzeargs

from classes.analyzers.metric_dictionaries_analyzer import MetricDictionariesAnalyzer
from classes.analyzers.metric_distributions_analyzer import MetricDistributionsAnalyzer
from classes.analyzers.metric_sets_analyzer import MetricSetsAnalyzer
from classes.constants import OUTPUT_DIR
from classes.metric import Metric, MetricScopeType, MetricAnalysisType
from classes.metric_analyzer import MetricAnalyzer
from classes.metric_config import MetricConfig
from metrics.pitch_metrics import BarPitchVarietyMetric, BarAbsolutePitchesMetric

from symusic import Score

from metrics.rythm_metrics import BarNoteDensityMetric, NoteDurationsMetric

from classes.generation_config import parse_filename, GenerationConfig

class MetricsProcessor:

    metrics: list[Metric]
    metric_config: MetricConfig

    def __init__(self, metric_config:MetricConfig):
        self.metric_config = metric_config

        self.metrics=[]
        if metric_config.bar_absolute_pitches:
            self.metrics.append(BarAbsolutePitchesMetric(MetricScopeType.BAR, MetricAnalysisType.SETS))
        if metric_config.bar_pitch_variety:
            self.metrics.append(BarPitchVarietyMetric(MetricScopeType.BAR, MetricAnalysisType.DIST))
        if metric_config.bar_note_density:
            self.metrics.append(BarNoteDensityMetric(MetricScopeType.BAR, MetricAnalysisType.DIST))
        if metric_config.note_durations:
            self.metrics.append(NoteDurationsMetric(MetricScopeType.GLOBAL, MetricAnalysisType.DICT))

    def compute_metrics(self, midi_files: list[str | Path], csv_output = False):
        #total_time_re = 0
        start_time = time.time()

        for midi_file in midi_files:
            self.score = Score(midi_file)

            _generation_config = parse_filename(midi_file)

            # Notice that _window_bars_ticks contains also the tick of the
            # bar right after the end of the context. This is donce for
            # indices purposes
            _window_bars_ticks = self._get_window_bars_ticks(_generation_config)

            if _window_bars_ticks is None:
                msg = ("[ERROR] MetricsProcessor::compute_metrics Couldn't compute"
                       f" bars ticks values for midi file: {midi_file}")
                raise ValueError(msg)




            for metric in self.metrics:

                # Get the distributions of the context (within the track)
                # and the infilling. The third output is the whole context
                track_context, infilling, _ = metric.compute_metric(
                    generation_config=_generation_config,
                    score = self.score,
                    window_bars_ticks = _window_bars_ticks)

                if _ is None:
                    metric.update_dataframe(midi_file,
                                            _generation_config.infilled_track_idx,
                                            track_context,
                                            infilling)
                else:
                    metric.update_dataframe(midi_file,
                                            _generation_config.infilled_track_idx,
                                            track_context,
                                            infilling,
                                            whole_context= _)

        end_time = time.time()

        print(f"Time to compute metrics: {end_time - start_time} seconds")
        #print(f"Time in re: {total_time_re} seconds")


        if csv_output:
            for metric in self.metrics:
                file_path = OUTPUT_DIR / f"{type(metric).__name__}_computation.csv"
                metric.raw_df.to_csv(file_path, index=False)

        self.analyze_metrics(csv_output)



    def analyze_metrics(self, csv_output):
        for metric in self.metrics:
            if metric.analysis_type == MetricAnalysisType.DIST:
                analyzer = MetricDistributionsAnalyzer(metric.raw_df)
            if metric.analysis_type == MetricAnalysisType.SETS:
                analyzer = MetricSetsAnalyzer(metric.raw_df)
            if metric.analysis_type == MetricAnalysisType.DICT:
                analyzer = MetricDictionariesAnalyzer(metric.raw_df)

            analyzer.analyze()

            if csv_output:
                file_path = OUTPUT_DIR / f"{type(metric).__name__}_analysis.csv"
                analyzer.analysis_df.to_csv(file_path, index=False)

                if metric.analysis_type == MetricAnalysisType.DIST:
                    file_path = OUTPUT_DIR / f"{type(metric).__name__}_agglomerated_analysis.csv"
                    analyzer.agglomerated_analysis_df.to_csv(file_path, index=False)


    def _get_window_bars_ticks(self, generation_config: GenerationConfig):
        bars_ticks = np.array(get_bars_ticks(self.score))

        infilling_start_idx = generation_config.infilled_bars[0]
        infilling_end_idx = generation_config.infilled_bars[1]

        return bars_ticks[infilling_start_idx - generation_config.context_size
                                       :infilling_end_idx + generation_config.context_size + 1]





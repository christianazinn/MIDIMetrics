from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class MetricScopeType(Enum):
    """
    Enum to define the scope type of the metric.
    """

    # Computed at bar level
    BAR = "bar"
    # Computed without considering bar granularity
    GLOBAL = "global"

class MetricAnalysisType(Enum):
    """
    Enum to define the type of the metric.
    """
    DIST = "distributions"
    SETS = "sets"
    DICT = "dictionaries"

class Metric(ABC):
    """
    Abstract base class for a flexible metric.

    This class allows for the computation of metrics at either the bar or global level,
    and accepts flexible input arguments to support diverse use cases.

    Attributes:
        type (MetricScopeType): Specifies whether the metric operates at the bar level
                           or the global level.
    """

    def __init__(self, scope_type: MetricScopeType, analysis_type: MetricAnalysisType):
        """
        Initializes a Metric with the specified type.

        :param type: The type of the metric (MetricType.BAR or MetricType.GLOBAL).
        """
        self._scope_type = scope_type
        self._analysis_type = analysis_type

        # Contains, for each row, the distributions computed over that MIDI file.
        self.raw_df = pd.DataFrame(columns=["File ID", "Track ID", "Track Context", "Infilling"])

    def update_dataframe(self, file_id, track_idx, track_context, infilling, whole_context = None):
        if whole_context == None:
            self.raw_df = pd.concat([
                self.raw_df,
                pd.DataFrame({
                    "File ID": file_id,  # Optional file_id from kwargs
                    "Track ID": track_idx,
                    "Track Context": [track_context],
                    "Infilling": [infilling]
                })
            ], ignore_index=True)
        else:
            self.raw_df = pd.concat([
                self.raw_df,
                pd.DataFrame({
                    "File ID": file_id,  # Optional file_id from kwargs
                    "Track ID": track_idx,
                    "Track Context": [track_context],
                    "Infilling": [infilling],
                    "Whole Context": [whole_context]
                })
            ], ignore_index=True)


    @property
    def scope_type(self) -> MetricScopeType:
        """
        Returns the scope type of the metric.

        :return: MetricType.BAR or MetricType.GLOBAL.
        """
        return self._scope_type

    @property
    def analysis_type(self) -> MetricAnalysisType:
        """
        Returns the analysis type of the metric.

        :return: MetricType.BAR or MetricType.GLOBAL.
        """
        return self._analysis_type

    @abstractmethod
    def compute_metric(self, metric_config, score, *args, **kwargs):
        """
        Computes the metric based on the provided arguments.

        This method is designed to accept flexible inputs to support various
        types of metrics. Derived classes must define the specific behavior.

        :param metric_config: Configuration object for the metric, containing
                              parameters and options for computation.
        :param score: The musical score or data object to analyze.
        :param args: Additional positional arguments for metric computation.
        :param kwargs: Additional keyword arguments for metric computation.
        :return: The computed metric data (type depends on the specific implementation).
        """
        raise NotImplementedError

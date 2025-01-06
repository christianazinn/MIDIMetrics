from abc import ABC, abstractmethod
from pathlib import Path

from symusic import Score

from classes.generation_config import GenerationConfig


class Metric(ABC):
    """
    Abstract base class for a flexible metric.

    This class allows for the computation of metrics at either the bar or global level,
    and accepts flexible input arguments to support diverse use cases.

    Attributes:
        type (MetricScopeType): Specifies whether the metric operates at the bar level
                           or the global level.
    """

    @abstractmethod
    def compute_metric(self, generation_config: GenerationConfig, score: Score, *args, **kwargs):
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

    @abstractmethod
    def analysis(self):
        raise NotImplementedError

    @abstractmethod
    def output_results(self, output_folder: Path | str):
        raise NotImplementedError

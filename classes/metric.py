from abc import ABC, abstractmethod
from enum import Enum

class MetricType(Enum):
    """
    Enum to define the type of the metric.
    """
    BAR = "bar"
    GLOBAL = "global"

class Metric(ABC):
    """
    Abstract base class for a flexible metric.

    This class allows for the computation of metrics at either the bar or global level,
    and accepts flexible input arguments to support diverse use cases.

    Attributes:
        type (MetricType): Specifies whether the metric operates at the bar level
                           or the global level.
    """

    def __init__(self, type: MetricType):
        """
        Initializes a Metric with the specified type.

        :param type: The type of the metric (MetricType.BAR or MetricType.GLOBAL).
        """
        self._type = type

    @property
    def type(self) -> MetricType:
        """
        Returns the type of the metric.

        :return: MetricType.BAR or MetricType.GLOBAL.
        """
        return self._type

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

    @abstractmethod
    def write_to_json(self):
        """
        Converts the computed metric data into a JSON-compatible format.

        Derived classes must implement this method to specify how their
        results are serialized for external usage.

        :return: A dictionary or JSON-serializable representation of the metric data.
        """
        raise NotImplementedError

from abc import ABC, abstractmethod

from symusic import Score

from classes.metric_config import MetricConfig


class Metric(ABC):

    @abstractmethod
    def compute_metric(self, metric_config: MetricConfig, score: Score):
        """

        :return:
        """
        raise NotImplementedError
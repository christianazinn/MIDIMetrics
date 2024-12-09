from abc import ABC, abstractmethod

import numpy as np
from symusic import Score

from classes.metric_config import MetricConfig


class Metric(ABC):


    @abstractmethod
    def compute_metric(self, metric_config: MetricConfig, score: Score, window_bars_ticks: np.array):
        """

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def write_to_json(self):

        raise NotImplementedError
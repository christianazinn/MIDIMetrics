from abc import abstractmethod
import pandas as pd


class MetricAnalyzer:
    """
    Base class for analyzing metrics within a DataFrame.

    This class is meant to be inherited by specific analyzers (e.g., for distribution, set, or dictionary-based analysis).
    It provides the basic functionality for initializing the analysis DataFrame and defining the `analyze` method,
    which must be implemented by subclasses.

    Attributes:
        analysis_df (pd.DataFrame): The DataFrame containing raw data to be analyzed.
    """

    def __init__(self, raw_df: pd.DataFrame):
        """
        Initializes the MetricAnalyzer with a DataFrame containing raw data.

        Args:
            raw_df (pd.DataFrame): The raw DataFrame that contains the data to analyze.
            The DataFrame should include common columns like 'File ID' and 'Track ID',
            along with any other specific columns required for analysis.
        """
        # Store the raw data in the 'analysis_df' attribute for analysis.
        self.analysis_df = raw_df

    @abstractmethod
    def analyze(self):
        """
        Abstract method that should be overridden by specific analyzers.

        This method is meant to perform the analysis of the raw data stored in 'analysis_df'.
        Each subclass should implement its own version of this method to handle the specific analysis logic.

        Raises:
            NotImplementedError: If called directly from this base class without being overridden.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
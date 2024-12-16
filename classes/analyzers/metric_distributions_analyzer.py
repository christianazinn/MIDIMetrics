from typing_extensions import override
from classes.metric_analyzer import MetricAnalyzer
import pandas as pd
import numpy as np


class MetricDistributionsAnalyzer(MetricAnalyzer):
    """
    This class inherits from `MetricAnalyzer` and is responsible for analyzing the
    distributions of 'Track Context' and 'Infilling' columns in the raw DataFrame.
    It computes various statistics (mean, median, standard deviation, min, max) for
    each row's 'Track Context' and 'Infilling' arrays and also calculates comprehensive metrics
    across all rows in the DataFrame.

    Attributes:
        analysis_df (pd.DataFrame): The DataFrame containing the data to analyze.
    """

    def __init__(self, raw_df: pd.DataFrame):
        """
        Initializes the MetricDistributionsAnalyzer with a DataFrame containing raw data.

        Args:
            raw_df (pd.DataFrame): The raw DataFrame that contains 'Track Context' and 'Infilling' columns,
            which are expected to be numpy arrays.
        """
        super().__init__(raw_df)  # Initialize the parent class with the raw DataFrame

    @override
    def analyze(self):
        """
        Analyze the distributions of 'Track Context' and 'Infilling' columns, and add computed statistics
        to the DataFrame.

        This method orchestrates the analysis by computing the row-wise statistics
        and then computing additional comprehensive metrics across all rows of the DataFrame.
        """
        self.compute_statistics()  # Compute the statistics for each row
        self.compute_comprehensive_metrics()  # Compute additional comprehensive metrics

    def compute_statistics(self):
        """
        Compute statistics (mean, median, standard deviation, min, and max) for each row's 'Track Context'
        and 'Infilling' arrays and store them as new columns in the DataFrame.

        The following statistics are computed:
            - Mean
            - Median
            - Standard Deviation
            - Minimum
            - Maximum

        These statistics are added to the DataFrame as new columns:
            - 'Context Mean', 'Context Median', 'Context Std', 'Context Min', 'Context Max'
            - 'Infilling Mean', 'Infilling Median', 'Infilling Std', 'Infilling Min', 'Infilling Max'
        """
        # Compute statistics for 'Track Context' column and assign to new columns in the DataFrame
        self.analysis_df['Track Context Mean'] = self.analysis_df['Track Context'].apply(np.mean)
        self.analysis_df['Track Context Median'] = self.analysis_df['Track Context'].apply(np.median)
        self.analysis_df['Track Context Std'] = self.analysis_df['Track Context'].apply(np.std)
        self.analysis_df['Track Context Min'] = self.analysis_df['Track Context'].apply(np.min)
        self.analysis_df['Track Context Max'] = self.analysis_df['Track Context'].apply(np.max)

        # Compute statistics for 'Infilling' column and assign to new columns in the DataFrame
        self.analysis_df['Infilling Mean'] = self.analysis_df['Infilling'].apply(np.mean)
        self.analysis_df['Infilling Median'] = self.analysis_df['Infilling'].apply(np.median)
        self.analysis_df['Infilling Std'] = self.analysis_df['Infilling'].apply(np.std)
        self.analysis_df['Infilling Min'] = self.analysis_df['Infilling'].apply(np.min)
        self.analysis_df['Infilling Max'] = self.analysis_df['Infilling'].apply(np.max)

    def compute_comprehensive_metrics(self):
        """
        Compute additional metrics across all rows of the DataFrame, such as the mean of the means
        and other aggregated statistics for the 'Track Context' and 'Infilling' columns.

        The following comprehensive metrics are calculated:
            - Mean of 'Context Mean'
            - Mean of 'Infilling Mean'
            - Mean of 'Context Std'
            - Mean of 'Infilling Std'
            - Mean of 'Context Min'
            - Mean of 'Infilling Min'
            - Mean of 'Context Max'
            - Mean of 'Infilling Max'

        These metrics can be printed or further processed as needed.
        """
        # Compute the mean of the statistics for the entire DataFrame
        agglomerated_analysis = {
            "Track Context Mean Mean": self.analysis_df["Track Context Mean"].mean(),
            "Infilling Mean Mean": self.analysis_df["Infilling Mean"].mean(),
            "Track Context Std Mean": self.analysis_df["Track Context Std"].mean(),
            "Infilling Std Mean": self.analysis_df["Infilling Std"].mean(),
            "Track Context Min Mean": self.analysis_df["Track Context Min"].mean(),
            "Infilling Min Mean": self.analysis_df["Infilling Min"].mean(),
            "Track Context Max Mean": self.analysis_df["Track Context Max"].mean(),
            "Infilling Max Mean": self.analysis_df["Infilling Max"].mean(),
        }

        # Convert the dictionary into a DataFrame
        self.agglomerated_analysis_df = pd.DataFrame(
            list(agglomerated_analysis.items()),
            columns=["Metric", "Value"]
        )
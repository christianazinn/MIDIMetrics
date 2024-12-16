from typing_extensions import override

import pandas as pd

from classes.metric_analyzer import MetricAnalyzer


class MetricSetsAnalyzer(MetricAnalyzer):
    """
    This class inherits from `MetricAnalyzer` and is responsible for analyzing the
    set relationships between the 'Infilling' and 'Track Context' columns in the DataFrame.

    Specifically, it calculates how many elements in the 'Infilling' list are not present
    in the 'Track Context' list for each row.

    Attributes:
        analysis_df (pd.DataFrame): The DataFrame containing the data to analyze.
    """

    def __init__(self, raw_df: pd.DataFrame):
        """
        Initializes the MetricSetsAnalyzer with a DataFrame containing raw data.

        Args:
            raw_df (pd.DataFrame): The raw DataFrame that contains 'Track Context' and 'Infilling' columns,
            which are expected to be lists or numpy arrays.
        """
        super().__init__(raw_df)  # Initialize the parent class with the raw DataFrame

    @override
    def analyze(self):
        """
        Analyze the set relationships between the 'Infilling' and 'Track Context' columns.

        This method computes the number of elements in the 'Infilling' list that are not
        present in the corresponding 'Track Context' list for each row, and adds this
        information as a new column 'Infilling Unique Count' in the DataFrame.
        """
        self.compute_uniqueness()  # Perform the uniqueness computation

    def compute_uniqueness(self):
        """
        Compute the number of unique elements in the 'Infilling' list that are not present
        in the 'Track Context' list for each row.

        This is calculated for each row in the DataFrame and added as a new column:
            - 'Infilling Unique Count': Number of elements in 'Infilling' that are not in 'Track Context'
        """

        def unique_count(row):
            # Convert to sets for efficient comparison
            context_set = set(row['Track Context'])
            infilling_set = set(row['Infilling'])
            # Count elements in 'Infilling' that are not in 'Track Context'
            unique_elements = infilling_set - context_set
            return len(unique_elements)

        # Apply the unique_count function to each row in the DataFrame
        self.analysis_df['Infilling Unique Count'] = self.analysis_df.apply(unique_count, axis=1)
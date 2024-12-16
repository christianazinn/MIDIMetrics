
from typing_extensions import override
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from classes.metric_analyzer import MetricAnalyzer


class MetricDictionariesAnalyzer(MetricAnalyzer):
    """
    This class inherits from `MetricAnalyzer` and analyzes the differences
    between the distributions in the 'Track Context', 'Whole Context', and 'Infilling' columns.

    The analysis includes metrics such as Cosine Similarity, KL Divergence, and
    Jensen-Shannon Divergence to evaluate the similarity between distributions.
    """

    def __init__(self, raw_df: pd.DataFrame):
        """
        Initializes the MetricDictionariesAnalyzer with a DataFrame containing raw data.

        Args:
            raw_df (pd.DataFrame): The raw DataFrame that contains the columns:
            'Track Context', 'Whole Context', and 'Infilling', which are expected to be dictionaries.
        """
        super().__init__(raw_df)  # Initialize the parent class with the raw DataFrame

    @override
    def analyze(self):
        """
        Analyze the distribution differences between 'Track Context', 'Whole Context', and 'Infilling'.

        This method computes metrics such as Cosine Similarity, KL Divergence, and Jensen-Shannon Divergence
        for each row, comparing the distributions in the specified columns, and adds the results as new columns
        to the DataFrame.
        """
        self.compute_distribution_metrics()  # Perform the distribution analysis

    def compute_distribution_metrics(self):
        """
        Compute distribution similarity metrics for each row in the DataFrame.

        The metrics calculated are:
            - Cosine Similarity between 'Infilling' and 'Track Context'
            - Cosine Similarity between 'Infilling' and 'Whole Context'
            - KL Divergence from 'Infilling' to 'Track Context'
            - KL Divergence from 'Infilling' to 'Whole Context'
            - Jensen-Shannon Divergence between 'Infilling' and 'Track Context'
            - Jensen-Shannon Divergence between 'Infilling' and 'Whole Context'
        """

        def normalize_distribution(distribution):
            """
            Normalize a dictionary representing a distribution so values sum to 1.

            Args:
                distribution (dict): A dictionary where keys are durations and values are counts.

            Returns:
                dict: A normalized dictionary where values are probabilities.
            """
            total = sum(distribution.values())
            if total == 0:
                return {key: 0 for key in distribution}
            return {key: value / total for key, value in distribution.items()}

        def align_distributions(dict1, dict2):
            """
            Align two dictionaries by ensuring they share the same keys, filling missing keys with zeros.

            Args:
                dict1 (dict): First dictionary.
                dict2 (dict): Second dictionary.

            Returns:
                tuple: Two aligned dictionaries with the same keys.
            """
            all_keys = set(dict1.keys()).union(set(dict2.keys()))
            aligned_dict1 = {key: dict1.get(key, 0) for key in all_keys}
            aligned_dict2 = {key: dict2.get(key, 0) for key in all_keys}
            return aligned_dict1, aligned_dict2

        def cosine_similarity(dict1, dict2):
            """
            Compute cosine similarity between two distributions.

            Args:
                dict1 (dict): First distribution.
                dict2 (dict): Second distribution.

            Returns:
                float: Cosine similarity between the two distributions.
            """
            vec1 = np.array(list(dict1.values()))
            vec2 = np.array(list(dict2.values()))
            return 1 - cosine(vec1, vec2)  # scipy.spatial.distance.cosine gives distance, so subtract from 1

        def kl_divergence(dict1, dict2):
            """
            Compute KL Divergence between two distributions.

            Args:
                dict1 (dict): First distribution (P).
                dict2 (dict): Second distribution (Q).

            Returns:
                float: KL Divergence D(P || Q).
            """
            vec1 = np.array(list(dict1.values()))
            vec2 = np.array(list(dict2.values()))
            vec2[vec2 == 0] = 1e-10  # Avoid division by zero
            return entropy(vec1, vec2)

        def js_divergence(dict1, dict2):
            """
            Compute Jensen-Shannon Divergence between two distributions.

            Args:
                dict1 (dict): First distribution (P).
                dict2 (dict): Second distribution (Q).

            Returns:
                float: Jensen-Shannon Divergence between the two distributions.
            """
            vec1 = np.array(list(dict1.values()))
            vec2 = np.array(list(dict2.values()))
            m = 0.5 * (vec1 + vec2)
            return 0.5 * entropy(vec1, m) + 0.5 * entropy(vec2, m)

        # Apply the computations to each row
        def compute_metrics(row):
            # Normalize and align the distributions
            track_context = normalize_distribution(row['Track Context'])
            whole_context = normalize_distribution(row['Whole Context'])
            infilling = normalize_distribution(row['Infilling'])

            track_context, infilling_tc = align_distributions(track_context, infilling)
            whole_context, infilling_wc = align_distributions(whole_context, infilling)

            # Compute metrics for Track Context
            cosine_sim_tc = cosine_similarity(infilling_tc, track_context)
            kl_div_tc = kl_divergence(infilling_tc, track_context)
            js_div_tc = js_divergence(infilling_tc, track_context)

            # Compute metrics for Whole Context
            cosine_sim_wc = cosine_similarity(infilling_wc, whole_context)
            kl_div_wc = kl_divergence(infilling_wc, whole_context)
            js_div_wc = js_divergence(infilling_wc, whole_context)

            return pd.Series({
                'Cosine Similarity (Track Context)': cosine_sim_tc,
                'KL Divergence (Track Context)': kl_div_tc,
                'Jensen-Shannon Divergence (Track Context)': js_div_tc,
                'Cosine Similarity (Whole Context)': cosine_sim_wc,
                'KL Divergence (Whole Context)': kl_div_wc,
                'Jensen-Shannon Divergence (Whole Context)': js_div_wc
            })

        # Apply the compute_metrics function to each row in the DataFrame
        self.analysis_df[[
            'Cosine Similarity (Track Context)',
            'KL Divergence (Track Context)',
            'Jensen-Shannon Divergence (Track Context)',
            'Cosine Similarity (Whole Context)',
            'KL Divergence (Whole Context)',
            'Jensen-Shannon Divergence (Whole Context)'
        ]] = self.analysis_df.apply(compute_metrics, axis=1)

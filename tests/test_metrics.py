from pathlib import Path, WindowsPath

from classes.generation_config import GenerationConfig
from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor

from tests_utils import HERE

MIDI_PATHS = list((HERE / "midis").glob("**/*.midi"))

metric_config = MetricConfig(
    #bar_absolute_pitches= True,
    bar_pitch_variety=True,
    #bar_note_density=True,
    #note_durations_set=True
    #note_durations_frequency=True #TODO! (maybe remove)
    #ngrams_repetitions=True,
    polyphony_min_max=True
)

if __name__ == "__main__":

    #path = HERE/"midis"/"track0_infill_bars47_51_generationtime_13.701295137405396.midi"

    metrics_processor = MetricsProcessor(
        metric_config
    )

    metrics_processor.compute_metrics(MIDI_PATHS, csv_output = True)

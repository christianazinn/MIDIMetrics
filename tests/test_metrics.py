from pathlib import Path, WindowsPath

from classes.generation_config import GenerationConfig
from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor

from tests_utils import HERE

MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))

metric_config = MetricConfig(
    bar_absolute_pitches= True,
    bar_pitch_variety=True,
    bar_note_density=True,
    note_durations=True
)

if __name__ == "__main__":

    path = HERE/"midis"/"track4_infill_bars150_154_generationtime_4.830227613449097.midi.mid"

    metrics_processor = MetricsProcessor(
        metric_config
    )

    metrics_processor.compute_metrics([path], csv_output = True)

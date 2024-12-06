from pathlib import Path

from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor

HERE = Path(__file__).parent

metric_config = MetricConfig(
    generation_type="infilling",
    infilled_bars=(52,56),
    context_size= 4,
    infilled_track_idx=0,
    bar_pitch_variety=True
)

if __name__ == "__main__":
    MIDI_file_path = HERE/"midis"/"track0_infill_bars51_55_generationtime_4.01728081703186.midi.mid"

    metrics_processor = MetricsProcessor(
        metric_config
    )

    metrics_processor.compute_metrics(MIDI_file_path)

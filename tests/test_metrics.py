from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor
from pathlib import Path

HERE = Path(__file__).parent

MIDI_PATHS = list((HERE / "midis").glob("**/*.mid"))

metric_config = MetricConfig(
    #bar_absolute_pitches=True,
    #bar_pitch_variety=True,
    #bar_note_density=True,
    #note_durations_set=True,
    #note_durations_frequency=True #TODO! (maybe remove)
    #ngrams_repetitions=True,
    #polyphony_min_max=True,

    content_preservation=True, # OK
    upc = True, # OK
    pr = True, #OK
    polyphony = True, # OK
    pv = True, #OK
    rv = True, #OK
    qr = True, #OK
    groove_consistency=True, #OK
    pitch_class_histogram_entropy = True #OK
)

if __name__ == "__main__":

    metrics_processor = MetricsProcessor(
        metric_config
    )

    metrics_processor.compute_metrics(MIDI_PATHS)

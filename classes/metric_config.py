from dataclasses import dataclass

@dataclass
class MetricConfig:
    """
        List of metrics to use
    """
    bar_absolute_pitches: bool = False
    bar_pitch_variety: bool = False
    bar_note_density: bool = False
    note_durations_set: bool = False
    note_durations_frequency: bool = False
    ngrams_repetitions: bool = False
    polyphony_min_max: bool = False
    content_preservation: bool = False
    upc : bool = False
    pr : bool = False
    polyphony : bool = False
    pv: bool = False
    rv: bool = False
    qr: bool = False
    groove_consistency: bool = False
    pitch_class_histogram_entropy: bool = False




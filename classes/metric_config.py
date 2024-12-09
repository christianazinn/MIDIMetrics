from dataclasses import dataclass
from typing import Tuple

@dataclass
class MetricConfig:
    """
        Configuration for the computation of the metrics.

        Specifies which type of generation has been performed on the MIDI file and
        which controls have to be computed.

        :param generationType: type of generation: either infilling or track.
        :param infilledBars: tuple of indeces of the infilled bars
        """

    generation_type: str = None
    infilled_track_idx: int  = None
    infilled_bars: Tuple[int, int] = None
    context_size: int = None
    bar_absolute_pitches: bool = False
    bar_pitch_variety: bool = False
    bar_note_density: bool = False
    note_durations: bool = False

    def __post_init__(self):
        if self.generation_type is None:
            msg = (
                "Please provide either 'infilling' or 'track' for the generation_type parameter"
            )
            raise ValueError(msg)

        if self.generation_type == "infilling" and self.infilled_bars is None:
            msg = "generation_type is set to 'infilling' but infilled_bars is not defined"
            raise ValueError(msg)

        if self.generation_type == "infilling" and self.context_size is None:
            msg = "generation_type is set to 'infilling' but context_size is not defined"
            raise ValueError(msg)






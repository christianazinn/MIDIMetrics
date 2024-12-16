from dataclasses import dataclass
import re
from pathlib import Path
from typing import Tuple


@dataclass
class GenerationConfig:
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


def parse_filename(file_path: Path) -> GenerationConfig:
    # Convert the Path object to string
    filename = str(file_path.name)

    # Remove any extra file extensions (if needed)
    filename = filename.rstrip('.midi')  # Removes a trailing .midi if it's there

    # Use regular expression to parse the filename
    match = re.match(r'track(\d+)_infill_bars(\d+)_([\d]+)_generationtime_[\d.]+', filename)

    if not match:
        raise ValueError(f"Filename format is invalid: {filename}")

    # Extract components from the regex match
    infilled_track_idx = int(match.group(1))  # Extract track ID
    bar_start = int(match.group(2)) + 1  # Extract starting bar
    bar_end = int(match.group(3)) + 1  # Extract ending bar

    # Build the GenerationConfig object
    generation_config = GenerationConfig(
        generation_type="infilling",  # The generation type is "infilling" by default
        infilled_bars=(bar_start, bar_end),
        context_size=4,  # Assuming a fixed context size for now
        infilled_track_idx=infilled_track_idx
    )

    return generation_config



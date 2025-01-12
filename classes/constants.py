from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"
ORIGINAL_MIDIFILES_DIR = Path(__file__).parent.parent / "tests" / "original_midis"

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 24,
    "special_tokens": [
        "PAD",
        "BOS",
        "EOS",
        "Infill_Bar",  # Indicates a bar to be filled in a seq
        "Infill_Track",  # Used in seq2seq to instruct the decoder to gen a new track
        "FillBar_Start",  # Start of the portion to infill (containing n bars)
        "FillBar_End",  # Ends the portion to infill
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": False,  # cannot be used as extracting tokens in data loading
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 127)),
    "base_tokenizer": "REMI",
}
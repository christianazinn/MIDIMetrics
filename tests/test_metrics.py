from pathlib import Path, WindowsPath

import os

from classes.generation_config import GenerationConfig
from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor

HERE = Path(__file__).parent

MODEL = os.getenv("MODEL", "mrwkv") # "midimistral"
N_BARS = int(os.getenv("N_BARS", 2))

CONTEXT = int(os.getenv("CTX", 2))

DRUMS = os.getenv("DRUMS") == 1
END_INFILLING = os.getenv("END_INFILLING") == 1
if DRUMS:
    following = f"context{CONTEXT}_drums"
elif END_INFILLING:
    following = "endinfilling"
else:
    following = f"context{CONTEXT}"
if os.getenv("pop909"):
    following += "_pop909"

MIDI_PATHS = list((HERE / f"FINALTEST/{MODEL}/bars_infill{N_BARS}_{following}").glob("**/*"))
print((HERE / f"FINALTEST/{MODEL}/bars_infill{N_BARS}_{following}"))
print(len(MIDI_PATHS))

if DRUMS:
    metric_config = MetricConfig(
        content_preservation=True,  # OK
        groove_consistency=True,  # OK
        f1onsets=True
    )
else:
    metric_config = MetricConfig(
        content_preservation=True, # OK
        groove_consistency=True, #OK
        pitch_class_histogram_entropy = True, #OK
        f1onsets = True
    )

if __name__ == "__main__":
    metrics_processor = MetricsProcessor(
        metric_config,
        output_dir = Path(f"{MODEL}/{N_BARS}bar_{following}")
    )

    metrics_processor.compute_metrics(MIDI_PATHS, None, musiac=False)

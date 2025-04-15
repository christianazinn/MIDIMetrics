from pathlib import Path, WindowsPath

from mmm_refactored import NUM_BARS

from classes.generation_config import GenerationConfig
from classes.metric_config import MetricConfig
from classes.metric_processor import MetricsProcessor

from tests_utils import HERE

MODEL = "musiac"
N_BARS = 8

CONTEXT=0

DRUMS = False
END_INFILLING = True
if DRUMS:
    following = f"context{CONTEXT}_drums"
elif END_INFILLING:
    following = "endinfilling"
else:
    if MODEL == "ours":
        following = f"context{CONTEXT}"
    else:
        following = f"context_{CONTEXT}"


#MIDI_PATHS = list((HERE / f"FINALTEST/{MODEL}/bars_infill{N_BARS}_context_{CONTEXT}").glob("**/*"))
MIDI_PATHS = list((HERE / f"FINALTEST/{MODEL}/bars_infill{N_BARS}_{following}").glob("**/*"))

if DRUMS:
    metric_config = MetricConfig(

        content_preservation=True,  # OK
        rv=True,  # OK
        qr=True,  # OK
        groove_consistency=True,  # OK
        f1onsets=True
    )
else:
    metric_config = MetricConfig(

        content_preservation=True, # OK
        upc = True, # OK

        pr = True, #OK
        polyphony = True, # OK
        pv = True, #OK
        rv = True, #OK
        qr = True, #OK
        groove_consistency=True, #OK
        pitch_class_histogram_entropy = True, #OK
        f1onsets = True
    )

if __name__ == "__main__":

    if MODEL == "musiac":
        musiac_b = True
        MUSIAC_ORIGINAL_MIDIFILES_DIR = Path(__file__).parent.parent / "tests" / "musiac_original_midis"
    else:
        musiac_b = False
        MUSIAC_ORIGINAL_MIDIFILES_DIR = None

    #path = HERE/"midis"/"track0_infill_bars47_51_generationtime_13.701295137405396.midi"

    metrics_processor = MetricsProcessor(
        metric_config,
        output_dir = Path(f"{MODEL}/{N_BARS}bar_{following}")
    )

    metrics_processor.compute_metrics(MIDI_PATHS, MUSIAC_ORIGINAL_MIDIFILES_DIR, musiac=musiac_b)

from symusic import Score, Synthesizer, dump_wav
from miditok.utils import get_bars_ticks
import os, random
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

path = "FINALTEST/mrwkv/bars_infill8_context16/"
outfolder = "evals/"
synth = Synthesizer()
CTX = 4

def make_outname(path):
    x = path.split("_")
    return f"{outfolder}{x[0]}_{x[-1]}.wav"

def trim_start_silence(input_path, output_path, silence_thresh=-50, min_silence_len=500):
    audio = AudioSegment.from_wav(input_path)

    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if nonsilent_ranges:
        start_trim = nonsilent_ranges[0][0]  # first non-silent part
        trimmed_audio = audio[start_trim:]
    else:
        print("Didn't find anything silent...?")
        trimmed_audio = audio  # no non-silent audio found

    trimmed_audio.export(output_path, format="wav")
    print(f"Trimmed audio saved to {output_path}")

def render(path_in):
    test_file = random.choice(os.listdir(path_in))
    original = test_file.split("_track")[0]

    others_stem = test_file.split("_generationtime")[0]
    bar_start = int(test_file.split("infill_bars")[1].split("_")[0])
    bar_end = int(test_file.split("infill_bars")[1].split("_")[1])

    alls = [path_in + file for file in os.listdir(path_in) if others_stem in file and "state2" not in file and "state3" not in file and "lora32" not in file]

    adjusted_start = bar_start - CTX
    adjusted_end = bar_end + CTX

    scores = [(path.split("/")[-1], Score(path)) for path in alls]
    scores.append((f"{original}_original", Score(f"original_midis/{original}.mid")))

    scores = [(make_outname(path), score.clip(get_bars_ticks(score)[adjusted_start], get_bars_ticks(score)[adjusted_end])) for path, score in scores]

    for path, score in scores:
        print(path)
        outwav = synth.render(score, stereo=True)
        tmp = path + "_tmp.wav"
        dump_wav(tmp, outwav, sample_rate=44100, use_int16=True)
        trim_start_silence(tmp, path)
        os.remove(tmp)

for _ in range(6):
    render(path)

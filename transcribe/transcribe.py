lang = 'ja'
fs = 16000
tag = 'Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave'

import os
import time
import torch
import string
import soundfile
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

d = ModelDownloader()
# It may takes a while to download and build models
speech2text = Speech2Text(
    **d.download_and_unpack(tag),
    device="cuda",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

def transcribe(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            print("transcribing " + filename)
            speech, rate = soundfile.read(directory + "/" + filename)
            nbests = speech2text(speech)
            text, *_ = nbests[0]
            print(f"ASR hypothesis: {text}")
            output = os.path.join(directory, filename[:-4] + ".lab")
            with open(output, "wb") as f:
                f.write(text.encode("utf-8"))
        else:
            continue

if __name__ == "__main__":
    input_directory = "original/output"
    transcribe(input_directory)
lang = 'ja'
fs = 16000
tag = 'Shinji Watanabe/laborotv_asr_train_asr_conformer2_latest33_raw_char_sp_valid.acc.ave'

import os
import time
import torch
import string
import soundfile

import whisper

# It may takes a while to download and build models
speech2text = whisper.load_model("large")

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

def transcribe(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            print("transcribing " + filename)
            speech, rate = soundfile.read(directory + "/" + filename)
            speech = whisper.pad_or_trim(speech)
            speech = speech.astype("float32")
            speech = speech2text.transcribe(speech)
            text = speech["text"]
            print(f"ASR hypothesis: {text}")
            output = os.path.join(directory, filename[:-4] + ".lab")
            with open(output, "wb") as f:
                f.write(text.encode("utf-8"))
        else:
            continue

if __name__ == "__main__":
    input_directory = "original/output"
    transcribe(input_directory)
input_directory = "original/bakemonogatari"
output_directory = "raw/bakemonogatari"

# dataset filelist
out_file = f"filelists/monogatari_out.txt"
dataset_name = [
    ('bakemonogatari','JP'),
]

##########
# 1. mp3 to wav 16000
# mp3towav.py
# 2. slice wav into sentences
# slice.py
# 3. transcribe using whisper or espnet
# transcribe.py
# 4. generate filelist
# make_filelist.py
##########
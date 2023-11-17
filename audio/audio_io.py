import torch
import torchaudio

import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")

print(SAMPLE_WAV)

metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)

gsm_metadata = torchaudio.info(SAMPLE_GSM)
print(gsm_metadata)

wav_8000_metadata = torchaudio.info(SAMPLE_WAV_8000)
print(wav_8000_metadata)

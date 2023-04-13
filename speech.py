import deepspeech
import wave
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from os import path
from pydub import AudioSegment

# # files                                                                         
src = "test.mp3"
dst = "test1.wav"

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")



model_file_path = 'deepspeech-0.9.3-models.pbmm'
scorer_file_path = 'deepspeech-0.9.3-models.scorer'
audio_file_path = dst

model = deepspeech.Model(model_file_path)
model.enableExternalScorer(scorer_file_path)

# Load audio file
with wave.open(audio_file_path, 'rb') as audio_file:
    audio_data = audio_file.readframes(audio_file.getnframes())
    audio_data = np.frombuffer(audio_data, dtype=np.int16)

    # Run speech recognition
text = model.stt(audio_data)
print(text)

# import numpy as np
# import sounddevice as sd
# import time
#
# # Samples per second
# sps = 44100
#
# # Frequency / pitch
# freq_hz = 440.0
#
# # Duration
# duration_s = 5.0
#
# # Attenuation so the sound is reasonable
# atten = 0.3
#
# # NumpPy magic to calculate the waveform
# each_sample_number = np.arange(duration_s * sps)
# waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
# waveform_quiet = waveform * atten
# #
# # devices = sd.query_devices()
# # print(devices)
# sd.default.device = 12
# devices = sd.query_devices()
# print(devices)
#
# # Play the waveform out the speakers
# sd.play(waveform_quiet, sps)
# time.sleep(duration_s)
# sd.stop()

from __future__ import print_function
import sounddevice as sd
from numpy import pi, sin, arange
import threading
import time

def streamer(stream, data):
    stream.start()
    stream.write(data)
    stream.close()

ddd = sd.query_devices()
# print(ddd)
sd.default.device = 6

for i in range(1,7):
    x = 0.1*sin(2*pi*i*440*arange(96e3)/48e3, dtype='float32')
    # stream = sd.OutputStream(device='default', samplerate=48000, channels=1, dtype='float32')
    # stream = sd.OutputStream(device=ddd[0]['name'], samplerate=48000, channels=1, dtype='float32')
    stream = sd.OutputStream(samplerate=48000, channels=1, dtype='float32')
    thread = threading.Thread(target=streamer, args=(stream, x))
    thread.start()
    print('Sin %dHz' % (i*440))
    time.sleep(1.5)
time.sleep(5)
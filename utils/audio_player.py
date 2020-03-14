
import time
import threading

import sounddevice as sd
import numpy as np

from utils.config import load_yml


class AudioPlayer(object):
    def __init__(self):
        self.sound_device = None
        self._get_audio_device()

    def _get_audio_device(self):
        self.sound_device = load_yml()["sound_device"]

    @staticmethod
    def audio_streamer(stream, data):
        stream.start()
        stream.write(data)
        stream.close()

    def play(self, audio_data):
        sampling_rate = 44100
        audio_time = 1.*(audio_data.shape[0] + 10) / sampling_rate
        sd.default.device = self.sound_device
        stream = sd.OutputStream(samplerate=sampling_rate, channels=1, dtype='float32')
        thread = threading.Thread(target=self.audio_streamer, args=(stream, audio_data))
        thread.start()
        time.sleep(audio_time)


if __name__ == '__main__':
    pass

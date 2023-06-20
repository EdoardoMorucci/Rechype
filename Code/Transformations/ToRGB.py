class ToRGB(object):
    def __init__(self):
        pass

    def __call__(self, waveform, sample_rate):
        waveform.unsqueeze(0)

        waveform = waveform.repeat(3, 1, 1)

        return waveform, sample_rate

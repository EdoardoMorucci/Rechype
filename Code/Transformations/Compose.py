class Compose(object):

    """Custom class for composing transformation on audio files"""

    def __init__(self, transforms: list) -> None:
        # Store the transformations
        self.transforms = transforms

    def __call__(self, waveform, sample_rate):
        """Apply each transformation to a tuple given by the waveform
        and its sample rate"""

        # Loop over all transformations
        for t in self.transforms:
            # Apply transformation to waveform and sample rate
            waveform, sample_rate = t(waveform, sample_rate)

        return waveform, sample_rate

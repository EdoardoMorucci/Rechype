import torch


class CropOrPad(object):

    """Give a maximum length for a sample in milliseconds,
    crop the sample if length is exceeding or add zero-padding
    to the right if shorter than maximum length"""

    def __init__(self, maximum_length: int):
        """
        maximum_length: max allowed length for an audio sample in milliseconds
        """

        # Store maximum length
        self.maximum_length = maximum_length

    def __call__(self, waveform: torch.tensor, sample_rate: int):
        """Apply cropping or padding to the sample"""

        # Get the waveform length
        n_audio_channels, waveform_length = waveform.shape

        # Convert the maximal length from milliseconds
        maximum_length = self.maximum_length // 1000 * sample_rate

        # If waveform longer than allowed
        if waveform_length > maximum_length:
            # Crop it
            waveform = waveform[..., :maximum_length]

        # If shorter, add zero padding to the right
        else:
            # Create the padding tensor
            padding = torch.zeros((n_audio_channels, maximum_length - waveform_length))

            # Concatenate it to the waveform
            waveform = torch.cat([waveform, padding], dim=1)

        return waveform, sample_rate

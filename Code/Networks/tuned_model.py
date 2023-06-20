import torch

from Code.Networks.sound_model import Model as SoundModel
from Code.Networks.wavegram_logmel_cnn14 import Model as Backbone


class Model(torch.nn.Module):
    def __init__(self, **parameters):
        # Init the model
        super(Model, self).__init__()

        # Initiate the backbone model
        backbone = Backbone(**parameters)

        # Initiate the head-model
        self.model = SoundModel(backbone=backbone, **parameters)

    def forward(self, x):
        return torch.sigmoid(self.model(x))

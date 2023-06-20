import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Siamese network for audio similarity estimation
    """

    def __init__(self, backbone, freeze_backbone: bool = True, latent_size: int = 256):
        # Call standard constructor
        super(Model, self).__init__()

        # Store backbone
        self.feature_extractor = backbone

        # Freeze backbone if required
        if freeze_backbone:
            # Freeze pretrained layers
            for layer in backbone.children():
                for param in layer.parameters():
                    param.requires_grad = False

        # Store the number of features extracted by the backbone architecture
        self.n_features = list(backbone.children())[-2].out_features

        # Add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(self.n_features * 2, latent_size),
            nn.ReLU(inplace=True),
            nn.Linear(latent_size, 1),
        )

        # Add sigmoid at the end
        self.sigmoid = nn.Sigmoid()

        # Initialize the weights
        self.fc.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """Initialise weight of the fully connected layer with
        Xavier initialisation"""

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass on one of the two audio files"""

        # Extract features
        _, features = self.feature_extractor(waveform)

        # Rearrange features
        features = features.view(features.size()[0], -1)

        return features

    def forward(self, waveform_1, waveform_2):
        """Forward pass on the two audio files"""

        #  Get the features for the two waveforms
        features_1 = self.forward_once(waveform_1)
        features_2 = self.forward_once(waveform_2)

        # Concatenate features
        features = torch.cat((features_1, features_2), 1)

        # Pass the concatenation to the linear layers
        output = self.fc(features)

        # Pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)

        return output

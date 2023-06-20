import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(
        self, backbone, n_labels, dropout_rate=0.9, freeze_backbone: int = 1, **kwargs
    ):
        """
        Build a model for classifying audio based on a backbone
        """

        # Init the model
        super(Model, self).__init__()

        # Store the backbone model
        self.backbone = backbone

        # Read the layers from last to first, neglecting the very last one
        for i, layer in enumerate(list(self.backbone.children())[-2::-1]):
            for param in layer.parameters():
                # In this case, the user wants to unfreeze the
                # entire network
                if freeze_backbone == -1:
                    param.requires_grad = True

                # If this is not the case, then it depends
                # on the layer number
                else:
                    # If the layer number exceeds the required number
                    if i > freeze_backbone:
                        # Freeze these inner layers
                        param.requires_grad = False

                    else:
                        # Else, make them trainable
                        param.requires_grad = True

        # Add dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Add fully connected layer on top
        self.fc_out = torch.nn.Linear(2048, n_labels)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # Extract features from backbone
        # h_transfer = self.backbone(x)["embedding"]
        _, h_transfer = self.backbone(x)

        # Pass through dropout
        h_transfer = self.dropout(h_transfer)

        # Pass them into the fully connected
        # You don't need to use a sigmoid activation, as that is part
        # of the CrossEntropy loss we are using for finetuning purposes
        y_pred = self.fc_out(h_transfer)

        return y_pred

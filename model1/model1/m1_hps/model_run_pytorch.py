"""PyTorch version of Model 1."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Model object."""

    def __init__(self, input_shape, samples, batchSamples, classCount):
        """Initialize the model object."""
        super(Net, self).__init__()

        self.batchSamples = batchSamples

        # Need batch norm and shuffle.  Shuffle is done with DataLoader.
        # Change BatchNorm to LayerNorm on SambaNova.
        self.ln1   = nn.LayerNorm(input_shape)

        in_features = input_shape[0] * input_shape[1]
        out_features = samples * 16
        self.fc1 = nn.Linear(in_features, out_features)
        self.dropout1 = nn.Dropout2d(0.20)

        in_features = out_features
        out_features = samples * 12
        self.fc2 = nn.Linear(in_features, out_features)
        self.dropout2 = nn.Dropout2d(0.10)



        # TODO: Insert more layers here.



        in_features = out_features
        out_features = 5
        self.fcN_1 = nn.Linear(in_features, out_features)

        in_features = out_features
        out_features = classCount
        self.fcN = nn.Linear(in_features, out_features)

    def forward(self, x):
        # sourcery skip: inline-immediately-returned-variable
        """
        Do forward pass.

        Args:
            x: represents our data.
        """
        # Pass data through ln1
        x = self.ln1(x)

        x = self.fc1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.dropout2(x)


        # TODO: Insert more layers here.


        x = self.fcN_1(x)
        x = F.relu(x)

        x = self.fcN(x)


        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output
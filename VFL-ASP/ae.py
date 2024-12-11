import torch.nn as nn
from torch.nn import functional as F


# convert tensor to numpy
def to_np(X):
    return X.data.cpu().numpy()


class AutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim, activation_name: str = 'leaky_relu') -> None:
        super().__init__()

        self.activation_name = activation_name
        self.activation_functions = {
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU
        }

        self.activation_forward = {
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'relu': F.relu,
            'leaky_relu': F.leaky_relu
        }

        # Define the units for the encoder layers
        encoder_units = [in_dim, 32, 64, 128, latent_dim]

        # Mirror the encoder structure for the decoder
        decoder_units = encoder_units[-1::-1]
        # print(encoder_units, decoder_units)

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        # Construct the encoder layers
        for i in range(len(encoder_units) - 1):
            self.encoder.add_module(f'Layer {i + 1}: net', nn.Linear(encoder_units[i], encoder_units[i + 1]))
            if i < len(encoder_units) - 2:  # Avoid activation after the latent layer
                self.encoder.add_module(f'Layer {i + 1}: activation', self.activation_functions[activation_name]())

        # Construct the decoder layers
        for i in range(len(decoder_units) - 1):
            self.decoder.add_module(f'Layer {i + 1}: net', nn.Linear(decoder_units[i], decoder_units[i + 1]))
            # Assuming you want activation functions for all decoder layers except the final output layer
            if i < len(decoder_units) - 2:
                self.decoder.add_module(f'Layer {i + 1}: activation', self.activation_functions[activation_name]())

    def forward(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def generate(self, X):
        return to_np(self.forward(X)[0])

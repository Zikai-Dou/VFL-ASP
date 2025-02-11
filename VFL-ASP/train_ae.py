import numpy as np
import torch
from torch import nn
from torch import optim


class StepTwo:
    def __init__(self, s2_model, params, device) -> None:
        self.model = s2_model
        self.params = params
        self.device = device

    def training(self, X_passive_1, Xs_emb):
        # Ensure data is a PyTorch tensor and move to the designated device
        if isinstance(X_passive_1, np.ndarray):
            self.X_passive_1 = torch.from_numpy(X_passive_1).to(self.device).to(torch.float32)  # old version
        else:
            self.X_passive_1 = X_passive_1.to(self.device).to(torch.float32)

        if isinstance(Xs_emb, np.ndarray):
            Xs_emb = torch.from_numpy(Xs_emb).to(self.device).to(torch.float32)  # old version
        else:
            Xs_emb = Xs_emb.to(self.device).to(torch.float32)

        # Move the model to the same device
        self.model = self.model.to(self.device)

        # Calculate the dividing line for the data
        divide_line = Xs_emb.shape[0]
        # print('Dividing line: ', divide_line)

        # Create a DataLoader for the passive 1 data
        data_batches = torch.utils.data.DataLoader(
            dataset=self.X_passive_1,
            batch_size=self.params['batch_size'],
            shuffle=False
        )

        # Define loss functions and optimizer
        mse_loss = nn.MSELoss()  # default: reduction='mean'
        # mae_loss = nn.L1Loss() optional for loss_emb
        optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])

        # Convert Xs_emb from torch.float64 to torch.float32
        Xs_emb = Xs_emb.to(torch.float32)

        # Training loop
        for epoch in range(self.params['num_epochs']):
            num_samples = 0
            for _, X in enumerate(data_batches):
                num_samples += X.shape[0]
                encoded, decoded = self.model(X)

                # Calculate the reconstruction loss
                loss_rec = mse_loss(X, decoded)

                # Calculate the embedding loss (only for the overlapping samples) and the reconstruction loss
                if 0 < num_samples - divide_line < X.shape[0]:
                    loss_emb = mse_loss(encoded[:X.shape[0] - (num_samples - divide_line), :],
                                        Xs_emb[num_samples - X.shape[0]:, :])
                    loss = loss_emb * self.params['lmd'] + loss_rec * (1 - self.params['lmd'])
                elif num_samples - divide_line <= 0:
                    loss_emb = mse_loss(encoded,
                                        Xs_emb[num_samples - X.shape[0]:num_samples, :])
                    loss = loss_emb * self.params['lmd'] + loss_rec * (1 - self.params['lmd'])
                else:
                    loss = loss_rec

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def embedding_approximation(self):
        approximation = self.model.generate(self.X_passive_1)
        combo = np.concatenate([self.X_passive_1.detach().cpu().numpy(), approximation], axis=1)

        return approximation, combo

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.autograd.set_detect_anomaly(True)


class LocalModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LocalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First linear layer
        self.fc2 = nn.Linear(128, 64)  # Second linear layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function after first layer
        # x = F.dropout(x, p=0.5)  # Dropout for regularization
        x = F.relu(self.fc2(x))  # Activation function after second layer
        x = self.fc3(x)
        return x  # Output raw scores (logits)


class GlobalModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(GlobalModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Client:
    def __init__(self, client_id, model, data, labels=None):
        self.client_id = client_id
        self.model = model
        self.data = data
        self.labels = labels
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def get_data_loader(self, batch_size):
        # Ensure the data is a torch Tensor
        data_tensor = torch.tensor(self.data, dtype=torch.float32)

        # If labels are not provided, create dummy labels
        if self.labels is None:
            labels_tensor = torch.zeros(data_tensor.size(0), 1, dtype=torch.float32)
        else:
            labels_tensor = torch.tensor(self.labels, dtype=torch.long)

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(data_tensor, labels_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def generate_output(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)

    def compute_output(self, X):
        self.model.train()  # Ensure the model is in train mode
        return self.model(X)

    def update_model(self, local_output, global_grad):
        # Ensure we calculate gradients for all parameters in the local model
        local_params = [self.model.fc1.weight, self.model.fc1.bias,
                        self.model.fc2.weight, self.model.fc2.bias,
                        self.model.fc3.weight, self.model.fc3.bias]

        # Computing the gradients of the local model's weights given the 'global_grad'
        local_grads = torch.autograd.grad(local_output, local_params, grad_outputs=global_grad, retain_graph=True)

        self.optimizer.zero_grad()

        # Manually setting the gradients
        for param, grad in zip(local_params, local_grads):
            param.grad = grad

        self.optimizer.step()


class VFL:
    def __init__(self, passive_client, active_client, global_model):
        self.passive_client = passive_client
        self.active_client = active_client
        self.global_model = global_model
        self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=0.01)

    def fit(self, num_epochs, batch_size, num_classes=None, patience=100, min_delta=0.0001):
        global_train_loss = 0
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            passive_loader = self.passive_client.get_data_loader(batch_size)
            active_loader = self.active_client.get_data_loader(batch_size)

            total_loss = 0
            num_batches = 0

            for (passive_batch, _), (active_batch, active_labels) in zip(passive_loader, active_loader):
                passive_output = self.passive_client.compute_output(passive_batch)
                active_output = self.active_client.compute_output(active_batch)

                combined_output = torch.cat((passive_output, active_output), dim=1)

                global_predictions = self.global_model(combined_output)

                one_hot_labels = torch.zeros(active_labels.size(0), num_classes)
                one_hot_labels.scatter_(1, active_labels, 1)

                global_loss = torch.nn.functional.cross_entropy(global_predictions, one_hot_labels)
                total_loss += global_loss.item()
                num_batches += 1

                passive_grads = torch.autograd.grad(global_loss, passive_output, retain_graph=True)[0]
                active_grads = torch.autograd.grad(global_loss, active_output, retain_graph=True)[0]

                self.global_optimizer.zero_grad()
                global_loss.backward(retain_graph=True)
                self.global_optimizer.step()

                self.passive_client.update_model(passive_output, passive_grads)
                self.active_client.update_model(active_output, active_grads)

            average_loss = total_loss / num_batches
            # global_train_loss = average_loss

            # if (epoch + 1) % 100 == 0:
            #     print(f'Epoch [{epoch + 1}], Global Training Loss: {average_loss}')

            # Early Stopping
            if average_loss < best_loss - min_delta:
                best_loss = average_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # print(f'Final Global Training Loss: {global_train_loss}')

    def generate_soft_labels(self, passive_data, active_data):
        self.passive_client.model.eval()
        self.active_client.model.eval()
        self.global_model.eval()

        passive_data_tensor = torch.tensor(passive_data, dtype=torch.float32)
        active_data_tensor = torch.tensor(active_data, dtype=torch.float32)
        with torch.no_grad():
            passive_output = self.passive_client.generate_output(passive_data_tensor)
            active_output = self.active_client.generate_output(active_data_tensor)
            combined_output = torch.cat((passive_output, active_output), dim=1)
            logits = self.global_model(combined_output)
            return torch.softmax(logits, dim=1)

    def generate_predictions(self, passive_data, active_data):
        # Obtain the softmax probabilities using the existing method
        probabilities = self.generate_soft_labels(passive_data, active_data)

        # Convert probabilities to actual class predictions
        _, hard_labels = torch.max(probabilities, dim=1)

        return hard_labels

# GPU version
class VFL_GPU:
    def __init__(self, passive_client, active_client, global_model, device):
        self.passive_client = passive_client
        self.active_client = active_client
        self.global_model = global_model
        self.global_optimizer = optim.Adam(self.global_model.parameters(), lr=0.01)
        self.device = device

    def fit(self, num_epochs, batch_size, num_classes=None, patience=100, min_delta=0.0001):
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            passive_loader = self.passive_client.get_data_loader(batch_size)
            active_loader = self.active_client.get_data_loader(batch_size)

            total_loss = 0
            num_batches = 0

            for (passive_batch, _), (active_batch, active_labels) in zip(passive_loader, active_loader):
                passive_batch = passive_batch.to(self.device)
                active_batch = active_batch.to(self.device)
                active_labels = active_labels.to(self.device)

                passive_output = self.passive_client.compute_output(passive_batch)
                active_output = self.active_client.compute_output(active_batch)

                combined_output = torch.cat((passive_output, active_output), dim=1).to(self.device)

                global_predictions = self.global_model(combined_output)

                one_hot_labels = torch.zeros(active_labels.size(0), num_classes, device=self.device)
                one_hot_labels.scatter_(1, active_labels, 1)

                global_loss = torch.nn.functional.cross_entropy(global_predictions, one_hot_labels)
                total_loss += global_loss.item()
                num_batches += 1

                passive_grads = torch.autograd.grad(global_loss, passive_output, retain_graph=True)[0]
                active_grads = torch.autograd.grad(global_loss, active_output, retain_graph=True)[0]

                self.global_optimizer.zero_grad()
                global_loss.backward(retain_graph=True)
                self.global_optimizer.step()

                self.passive_client.update_model(passive_output, passive_grads)
                self.active_client.update_model(active_output, active_grads)

            average_loss = total_loss / num_batches

            # if (epoch + 1) % 100 == 0:
            #     print(f'Epoch [{epoch + 1}], Global Training Loss: {average_loss}')

            # Early Stopping
            if average_loss < best_loss - min_delta:
                best_loss = average_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                # print(f'Early stopping at epoch {epoch + 1}')
                break

        # print(f'Final Global Training Loss: {global_train_loss}')

    def generate_soft_labels(self, passive_data, active_data):
        self.passive_client.model.eval()
        self.active_client.model.eval()
        self.global_model.eval()

        passive_data_tensor = torch.tensor(passive_data, dtype=torch.float32).to(self.device)
        active_data_tensor = torch.tensor(active_data, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            passive_output = self.passive_client.generate_output(passive_data_tensor)
            active_output = self.active_client.generate_output(active_data_tensor)
            combined_output = torch.cat((passive_output, active_output), dim=1)
            logits = self.global_model(combined_output)
            return torch.softmax(logits, dim=1)

    def generate_predictions(self, passive_data, active_data):
        # Obtain the softmax probabilities using the existing method
        probabilities = self.generate_soft_labels(passive_data, active_data)

        # Convert probabilities to actual class predictions
        _, hard_labels = torch.max(probabilities, dim=1)

        return hard_labels


class StudentModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Apply log_softmax to convert scores into log-probabilities, kl_div_loss expects log-probabilities
        return torch.log_softmax(x, dim=1)


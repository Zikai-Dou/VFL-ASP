import argparse
import warnings
import torch
import numpy as np
import random
import time

start_time = time.time()

seed = 100
print('Global Seed:', seed)

# Set the seed for generating random numbers
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# If using CUDA, set the random seed for all GPUs to ensure reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore")

from fsvd import *
from train_fsvd import *
from ae import *
from train_ae import *
from performance import *
from utils import *
from vfl_kd import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a single task!')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='Choose your GPU resource.')

    args = parser.parse_args()
    device = get_gpu(args.gpu)

    task_config = load_task_config()
    setting = load_dataset_config(dataset='Breast', type=task_config['split'])  # task_config['dataset']

    p2_num_sample = setting['p2_num_sample']
    p1_num_sample = setting['p1_num_sample']
    active_num_sample = setting['active_num_sample']
    p2_num_feature = setting['p2_num_feature']
    p1_num_feature = setting['p1_num_feature']
    active_num_feature = setting['active_num_feature']
    shared_sample_2 = setting['shared_sample_2']
    shared_sample_1 = setting['shared_sample_1']
    shared_feature_2 = setting['shared_feature_2']
    shared_feature_1 = setting['shared_feature_1']

    print('Dataset: ', 'Breast')  # task_config['dataset']
    print('Step 1: FSVD')
    print('Step 2: AE')
    print('Step 3-4: VFL_KD')

    print('Run on ', device)

    # Split data
    (X_passive_2, y_passive_2, X_passive_1, y_passive_1, X_active, y_active,
     X_shared_2, y_shared_2, Xs, X_shared_1, y_shared_1,
     p2_shared, p1_shared_2, p1_shared_1, a_shared) = eval(task_config['split'])('Breast')

    print('Passive 2: ', X_passive_2.shape)
    print('Passive 1: ', X_passive_1.shape)
    print('Active: ', X_active.shape)
    print('2nd overlapping: ', X_shared_2.shape)
    print('1st overlapping: ', X_shared_1.shape)

    # Step 1
    s1_model_config = load_model_config("FSVD")  # yaml file
    s1_model = FSVD()
    s1 = StepOne(s1_model, s1_model_config['model_params'])
    Emb_U, Emb_US = s1.training(X_shared=X_shared_2, Xs=Xs)
    print('Embeddings: ', Emb_US.shape)

    # Step 2
    s2_model_config = load_model_config("AE")  # yaml file
    s2_model = AutoEncoder(X_passive_1.shape[1], Emb_US.shape[1], **s2_model_config['model_params'])
    s2 = StepTwo(s2_model, s2_model_config['exp_params'], device)
    s2.training(X_passive_1, Emb_US)
    Emb_a, combo = s2.embedding_approximation()
    print('Embedding approximation:', Emb_a.shape)


    print("-------Step 1 Result---------")
    test = Test()
    print('p2')
    test.run(p2_shared, y_shared_2)
    print('p1')
    test.run(p1_shared_2, y_shared_2)
    print('Emb_US')
    test.run(Emb_US, y_shared_2)
    print('X_shared')
    test.run(X_shared_2, y_shared_2)

    # Step 3
    print("--------Step 3 Result---------")
    start_time_2 = time.time()

    accuracies_2 = []
    for i in range(100):
        print(f'Iteration {i + 1}')
        # Initialize Clients with local models
        client_p1 = Client(
            client_id=3,
            model=LocalModel(X_passive_1.shape[1], 32),
            data=p1_shared_1[:int(300 - shared_sample_2), :]
        )
        active_2 = Client(
            client_id=4,
            model=LocalModel(X_active.shape[1], 32),
            data=a_shared[:int(300-shared_sample_2), :],
            labels=y_shared_1[:int(300-shared_sample_2), :]
        )

        # Initialize Global Model (same as before)
        global_model_2 = GlobalModel(input_size=64, output_size=2)  # Assuming input size and output size remain suitable

        # Initialize VFL with new client_p1
        vfl_2 = VFL(passive_client=client_p1, active_client=active_2, global_model=global_model_2)

        # Train VFL with new setup
        vfl_2.fit(num_epochs=1000, batch_size=50, num_classes=2, patience=100)

        # Generate global predictions with new setup
        teacher_prediction_2 = vfl_2.generate_predictions(
            p1_shared_1[int(300 - shared_sample_2):, :],
            a_shared[int(300-shared_sample_2):, :]
        )

        # Calculate accuracy for the new setup
        y_active_test_te_2 = torch.tensor(y_shared_1[int(300-shared_sample_2):, :], dtype=torch.long)
        y_active_test_te_2 = torch.tensor([label.item() for sublist in y_active_test_te_2 for label in sublist],
                                          dtype=torch.long)

        total_test_2 = y_active_test_te_2.size(0)
        correct_test_2 = (teacher_prediction_2 == y_active_test_te_2).sum().item()
        teacher_accuracy_2 = correct_test_2 / total_test_2

        # Add accuracy to list
        accuracies_2.append(teacher_accuracy_2)

    # Calculate average accuracy
    # print(accuracies_2)
    average_accuracy_2, confidence_interval_2 = confidence_interval(accuracies_2)
    print(f'Teacher Test Accuracy 2: {average_accuracy_2}, {confidence_interval_2}')

    end_time_2 = time.time()
    elapsed_time_2 = end_time_2 - start_time_2
    print(f"Elapsed time: {elapsed_time_2:.5f} seconds")

    ############################################################################

    # VFL-ASP
    accuracies_1 = []
    sum_soft_labels = None
    for i in range(100):
        print(f'Iteration {i + 1}')
        # Initialize Clients with local models
        embedding_a = Client(
            client_id=1,
            model=LocalModel(Emb_a.shape[1], 32),
            data=Emb_a[shared_sample_2:300, :]
        )
        active = Client(
            client_id=2,
            model=LocalModel(X_active.shape[1], 32),
            data=a_shared[:int(300-shared_sample_2), :],
            labels=y_shared_1[:int(300-shared_sample_2), :]
        )

        # Initialize Global Model
        global_model = GlobalModel(input_size=64, output_size=2)  # for concatenated output from both clients

        # Initialize VFL
        vfl = VFL(passive_client=embedding_a, active_client=active, global_model=global_model)

        # Train VFL
        vfl.fit(num_epochs=1000, batch_size=50, num_classes=2, patience=100)  # 4 classes for MIMIC, 2 classes for Breast

        # Generate Soft Labels. Make sure this is a probability distribution
        soft_labels = vfl.generate_soft_labels(
            Emb_a[shared_sample_2:300, :],
            a_shared[:int(300-shared_sample_2), :]
        )  # with both emb_a and active
        # print('soft_labels:', soft_labels.shape)

        if sum_soft_labels is None:
            sum_soft_labels = torch.zeros_like(soft_labels)

        sum_soft_labels += soft_labels

        # Generate global predictions
        teacher_prediction = vfl.generate_predictions(
            Emb_a[300:, :],
            a_shared[int(300-shared_sample_2):, :]
        )
        # print('Teacher Prediction:', teacher_prediction)

        y_active_test_te = torch.tensor(y_shared_1[int(300-shared_sample_2):, :], dtype=torch.long)
        y_active_test_te = torch.tensor([label.item() for sublist in y_active_test_te for label in sublist],
                                        dtype=torch.long)

        total_test = y_active_test_te.size(0)
        correct_test = (teacher_prediction == y_active_test_te).sum().item()
        teacher_accuracy = correct_test / total_test

        # Add accuracy to list
        accuracies_1.append(teacher_accuracy)

    # Calculate average accuracy
    # print(accuracies_1)
    average_accuracy_1, confidence_interval_1 = confidence_interval(accuracies_1)
    print(f'Teacher Test Accuracy: {average_accuracy_1}, {confidence_interval_1}')

    end_time_1 = time.time()
    elapsed_time_1 = end_time_1 - end_time_2
    print(f"Elapsed time: {elapsed_time_1:.5f} seconds")

    # Average soft labels
    average_soft_labels = sum_soft_labels / len(accuracies_1)

    ##################################################################################

    # Instantiate the Active local Model
    accuracy_lo_v = []
    accuracy_lo_s = []

    # Early stopping parameters
    patience = 100  # Number of epochs to wait for improvement before stopping
    min_delta = 0.0001  # Minimum change in the monitored quantity to qualify as an improvement

    for _ in range(100):
        active_local_model = LocalModel(input_size=X_active.shape[1], output_size=2)

        # Convert data to PyTorch tensors
        a_shared_tensor = torch.tensor(a_shared[:int(300-shared_sample_2), :], dtype=torch.float32)
        y_shared_1_tensor = torch.from_numpy(y_shared_1[:int(300-shared_sample_2), :]).to(torch.long)  # Convert to tensor and ensure it's an integer type

        # Perform One-Hot Encoding
        one_hot_labels = torch.zeros(y_shared_1_tensor.size(0), 2)  # Initialize the one-hot tensor
        one_hot_labels.scatter_(1, y_shared_1_tensor, 1)
        # print(one_hot_labels)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(active_local_model.parameters(), lr=0.01)

        loss = 0
        best_loss = float('inf')
        epochs_no_improve = 0

        # Training the model
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = active_local_model(a_shared_tensor)
            loss = criterion(outputs, one_hot_labels)

            loss.backward()
            optimizer.step()

            # Check early stopping condition
            current_loss = loss.item()
            if current_loss < best_loss - min_delta:  # Improvement condition
                best_loss = current_loss
                epochs_no_improve = 0  # Reset the counter
            else:
                epochs_no_improve += 1  # Increment the counter

            if epochs_no_improve >= patience:  # Early stopping condition
                # print(f"Early stopping at epoch {epoch + 1}")
                break  # Exit the training loop

        # Display final loss
        active_local_loss = loss.item()
        # print(f'Active Local Training Loss: {active_local_loss}')

        # Test using the active local model against the VFL
        X_active_test_lo_v = torch.tensor(a_shared[int(300-shared_sample_2):, :], dtype=torch.float32)
        y_active_test_lo_v = torch.tensor(y_shared_1[int(300-shared_sample_2):, :], dtype=torch.long)
        y_active_test_lo_v = torch.tensor([label.item() for sublist in y_active_test_lo_v for label in sublist],
                                        dtype=torch.long)

        # Test using the active local model against the Student
        X_active_test_lo_s = torch.tensor(X_active[int(450-shared_sample_2):, :], dtype=torch.float32)
        y_active_test_lo_s = torch.tensor(y_active[int(450-shared_sample_2):, :], dtype=torch.long)
        y_active_test_lo_s = torch.tensor([label.item() for sublist in y_active_test_lo_s for label in sublist],
                                        dtype=torch.long)

        # Set the model to evaluation mode
        active_local_model.eval()

        # Make predictions against the VFL
        with torch.no_grad():
            predictions_lo_v = active_local_model(X_active_test_lo_v)

        # Apply softmax to obtain probabilities
        probabilities_v = F.softmax(predictions_lo_v, dim=1)

        # Get predicted labels (class indices with the highest probability)
        predicted_labels_lo_v = torch.argmax(probabilities_v, dim=1)

        # Calculate accuracy
        total_predictions_lo_v = y_active_test_lo_v.size(0)
        correct_predictions_lo_v = (predicted_labels_lo_v == y_active_test_lo_v).sum().item()
        active_local_accuracy_v = correct_predictions_lo_v / total_predictions_lo_v

        accuracy_lo_v.append(active_local_accuracy_v)

        # Make predictions against the Student
        with torch.no_grad():
            predictions_lo_s = active_local_model(X_active_test_lo_s)

        # Apply softmax to obtain probabilities
        probabilities_s = F.softmax(predictions_lo_s, dim=1)

        # Get predicted labels (class indices with the highest probability)
        predicted_labels_lo_s = torch.argmax(probabilities_s, dim=1)

        # Calculate accuracy
        total_predictions_lo_s = y_active_test_lo_s.size(0)
        correct_predictions_lo_s = (predicted_labels_lo_s == y_active_test_lo_s).sum().item()
        active_local_accuracy_s = correct_predictions_lo_s / total_predictions_lo_s

        accuracy_lo_s.append(active_local_accuracy_s)

    # Calculate average accuracy
    average_accuracy_lo_v, confidence_interval_lo_v = confidence_interval(accuracy_lo_v)
    average_accuracy_lo_s, confidence_interval_lo_s = confidence_interval(accuracy_lo_s)

    print(f'Active Local Test Accuracy against vfl: {average_accuracy_lo_v}, {confidence_interval_lo_v}')

    ##################################################################################

    # Step 4
    print("--------Step 4 Result---------")
    total_accuracy_0 = []

    # Early stopping parameters
    patience = 100  # Number of epochs to wait for improvement before stopping
    min_delta = 0.0001  # Minimum change in the monitored quantity to qualify as an improvement

    for _ in range(100):
        # Initialize the student model
        student_model = StudentModel(input_size=a_shared.shape[1], num_classes=2)

        # Convert data to PyTorch tensors
        a_shared_tensor = torch.tensor(a_shared[0:int(300-shared_sample_2), :], dtype=torch.float32)

        # Define loss function and optimizer
        criterion = nn.KLDivLoss(reduction='batchmean')  # KL Divergence Loss
        optimizer = optim.Adam(student_model.parameters(), lr=0.01)

        loss = 0
        best_loss = float('inf')
        epochs_no_improve = 0

        # Training the student model
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = student_model(a_shared_tensor)
            loss = criterion(outputs, average_soft_labels)

            loss.backward()
            optimizer.step()

            # Check early stopping condition
            current_loss = loss.item()
            if current_loss < best_loss - min_delta:  # Improvement condition
                best_loss = current_loss
                epochs_no_improve = 0  # Reset the counter
            else:
                epochs_no_improve += 1  # Increment the counter

            if epochs_no_improve >= patience:  # Early stopping condition
                # print(f"Early stopping at epoch {epoch + 1}")
                break  # Exit the training loop

        # Display final loss
        # student_loss = loss.item()
        # print(f'Student Training Loss: {student_loss}')

        # Test using the student model
        X_active_test_st = torch.tensor(X_active[int(450-shared_sample_2):, :], dtype=torch.float32)
        y_active_test_st = torch.tensor(y_active[int(450-shared_sample_2):, :], dtype=torch.long)
        y_active_test_st = torch.tensor([label.item() for sublist in y_active_test_st for label in sublist],
                                        dtype=torch.long)

        # Set the model to evaluation mode
        student_model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = student_model(X_active_test_st)

        # Convert probabilities to class labels
        predicted_labels = torch.argmax(predictions, dim=1)
        predicted_labels_list = [[label.item()] for label in predicted_labels]

        # Calculate accuracy
        total_predictions = y_active_test_st.size(0)
        correct_predictions = (predicted_labels == y_active_test_st).sum().item()
        student_accuracy = correct_predictions / total_predictions

        # Add accuracy to total
        total_accuracy_0.append(student_accuracy)

    # Calculate average accuracy
    average_accuracy_0, confidence_interval_s = confidence_interval(total_accuracy_0)
    print(f'Student Test Accuracy: {average_accuracy_0}, {confidence_interval_s}')

    print(f'Active Local Test Accuracy against student: {average_accuracy_lo_s}, {confidence_interval_lo_s}')

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.5f} seconds")
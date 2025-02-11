import numpy as np
import pandas as pd
import json
import yaml
import torch
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


seed = 100


def load_data(dataset: str, shuffle_columns=False, shuffle_samples=True, random_seed: int = seed):
    np.random.seed(random_seed)  # Set the seed for reproducibility
    print('Load data seed:', random_seed)

    if shuffle_columns:
        print('Shuffling columns...')

    if shuffle_samples:
        print('Shuffling samples...')

    if dataset == 'MIMIC':
        data = pd.read_csv('dataset/MIMIC/mimic3d.csv')
        drop_cols = ['hadm_id', 'AdmitDiagnosis', 'AdmitProcedure', 'religion', 'insurance', 'ethnicity',
                     'marital_status', 'ExpiredHospital', 'LOSdays', 'gender', 'admit_type', 'admit_location']
        data.drop(drop_cols, inplace=True, axis=1)

        feature_order = ['LOSgroupNum', 'NumTransfers', 'NumDiagnosis', 'NumCallouts', 'NumChartEvents', 'TotalNumInteract', 'NumProcEvents', 'age', 'NumInput', 'NumNotes', 'NumRx', 'NumMicroLabs', 'NumProcs', 'NumLabs', 'NumCPTevents', 'NumOutput']

        data = data[feature_order]

        # Shuffle column order if required
        if shuffle_columns:
            shuffled_cols = np.random.permutation(data.columns)
            data = data[shuffled_cols]
            print('Shuffled columns:', shuffled_cols)

        # Shuffle samples if required
        if shuffle_samples:
            data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        X = data.drop(['LOSgroupNum'], axis=1).to_numpy()
        y = data['LOSgroupNum'].to_numpy().reshape((len(X), 1))
        # y = (y > 1).astype(int)

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'Breast':
        data = pd.read_csv('dataset/Breast/data.csv')
        data.drop(["Unnamed: 32", "id"], inplace=True, axis=1)
        # Encode diagnosis column (M = 1, B = 0)
        data["diagnosis"] = [1 if i.strip() == "M" else 0 for i in data.diagnosis]

        # Define the order of the features
        feature_order = ['diagnosis', 'area_worst', 'area_mean', 'area_se', 'radius_mean', 'compactness_se', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'symmetry_worst', 'fractal_dimension_worst', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'smoothness_se', 'perimeter_se', 'texture_worst', 'texture_mean', 'radius_worst', 'perimeter_worst', 'perimeter_mean']

        # Ensure the columns are ordered correctly
        data = data[feature_order]

        # Shuffle column order if required
        if shuffle_columns:
            shuffled_cols = np.random.permutation(data.columns)
            data = data[shuffled_cols]
            print('Shuffled columns:', shuffled_cols)

        # Shuffle samples if required
        if shuffle_samples:
            data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        X = data.drop(["diagnosis"], axis=1).to_numpy()
        y = data['diagnosis'].to_numpy().reshape((len(X), 1))

        # # Standardize the data
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

    elif dataset == 'Credit':
        data = pd.read_csv('dataset/Credit/credit.csv')
        data.drop('ID', inplace=True, axis=1)

        # feature_order = []
        # data = data[feature_order]

        # Shuffle column order if required
        if shuffle_columns:
            shuffled_cols = np.random.permutation(data.columns)
            data = data[shuffled_cols]
            print('Shuffled columns:', shuffled_cols)

        # Shuffle samples if required
        if shuffle_samples:
            data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        X = data.drop(['default payment next month'], axis=1).to_numpy()
        y = data['default payment next month'].to_numpy().reshape((len(X), 1))

        # # Standardize the data
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

    elif dataset == 'HAPT':
        X_train = np.loadtxt('dataset/HAPT/X_train.txt')
        y_train = np.loadtxt('dataset/HAPT/y_train.txt') - 1
        X_test = np.loadtxt('dataset/HAPT/X_test.txt')
        y_test = np.loadtxt('dataset/HAPT/y_test.txt') - 1
        X = np.concatenate([X_train, X_test], axis=0).astype(float)
        y = np.concatenate([y_train, y_test], axis=0).astype(float).reshape([X.shape[0], 1])

        if shuffle_columns:
            col_indices = np.random.permutation(X.shape[1])
            X = X[:, col_indices]
            print('Shuffled columns:', col_indices)

            # Shuffle samples if required
        if shuffle_samples:
            sample_indices = np.random.permutation(X.shape[0])
            X = X[sample_indices, :]
            y = y[sample_indices, :]
            print('Shuffled samples')
        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    else:
        print('Wrong dataset!')
        return None

    return X, y


def load_dataset_config(dataset: str, type: str):
    with open('configs/configuration.json') as load_config:
        config = json.load(load_config)

    return config[dataset][type]


def load_task_config():
    filename = './configs/task_config.yaml'
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def load_model_config(model_name: str):
    filename = './configs/' + model_name.lower() + '.yaml'
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def get_gpu(gpu_idx='0'):
    return torch.device('cuda:' + gpu_idx if torch.cuda.is_available() else "cpu")


def confidence_interval(data, confidence=0.95):
    # Convert the data list to a numpy array
    data = np.array(data)

    # Calculate the mean of the data
    mean = np.mean(data)

    # Calculate the standard error of the mean
    sem = stats.sem(data)

    # Calculate the confidence interval
    h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)

    return mean, h


# For 3 parties
def unequal_split(dataset='MIMIC', start_sample=0, start_feature=0, random_seed=seed):
    X, y = load_data(dataset, random_seed=random_seed)
    # For 'MIMIC' dataset, (58976, 15) 4
    # For 'Breast' dataset, (569, 30) 2
    # For 'Credit' dataset, (30000, 23) 2

    num_unique_labels = len(torch.unique(torch.tensor(y)))
    print('Number of unique labels:', num_unique_labels)
    print('Max label index:', torch.max(torch.tensor(y)))
    print('Min label index:', torch.min(torch.tensor(y)))

    setting = load_dataset_config(dataset, 'unequal_split')

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

    # Passive party 2
    X_passive_2 = X[start_sample:start_sample + p2_num_sample, start_feature:start_feature + p2_num_feature]
    y_passive_2 = y[start_sample:start_sample + p2_num_sample, :]

    # Passive party 1
    X_passive_1 = X[start_sample + p2_num_sample - shared_sample_2:start_sample + p2_num_sample + p1_num_sample - shared_sample_2,
                  start_feature + p2_num_feature - shared_feature_2:start_feature + p2_num_feature + p1_num_feature - shared_feature_2]
    y_passive_1 = y[start_sample + p2_num_sample - shared_sample_2:start_sample + p2_num_sample + p1_num_sample - shared_sample_2, :]

    # Active party
    X_active = X[start_sample + p2_num_sample + p1_num_sample - shared_sample_2 - shared_sample_1:start_sample + p2_num_sample + p1_num_sample + active_num_sample - shared_sample_2 - shared_sample_1,
               start_feature + p2_num_feature + p1_num_feature - shared_feature_2 - shared_feature_1:start_feature + p2_num_feature + p1_num_feature + active_num_feature - shared_feature_2 - shared_feature_1]
    y_active = y[start_sample + p2_num_sample + p1_num_sample - shared_sample_2 - shared_sample_1:start_sample + p2_num_sample + p1_num_sample + active_num_sample - shared_sample_2 - shared_sample_1, :]

    # Second overlapping samples
    p2_shared = X_passive_2[p2_num_sample - shared_sample_2:p2_num_sample, :]
    p1_shared_2 = X_passive_1[:shared_sample_2, :p1_num_feature]
    X_shared_2 = np.concatenate([p2_shared, p1_shared_2], axis=1)
    y_shared_2 = y_passive_2[p2_num_sample - shared_sample_2:p2_num_sample, :]
    Xs = [p2_shared, p1_shared_2]

    # First overlapping samples
    p1_shared_1 = X_passive_1[p1_num_sample - shared_sample_1:p1_num_sample, :]
    a_shared = X_active[:shared_sample_1, :active_num_feature]
    X_shared_1 = np.concatenate([p1_shared_1, a_shared], axis=1)
    y_shared_1 = y_active[:shared_sample_1, :]

    return (X_passive_2, y_passive_2, X_passive_1, y_passive_1, X_active, y_active,
            X_shared_2, y_shared_2, Xs, X_shared_1, y_shared_1,
            p2_shared, p1_shared_2, p1_shared_1, a_shared)


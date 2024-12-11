import numpy as np
import pandas as pd
# from sklearn.feature_selection.tests.test_base import feature_names
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

seed = 100

def load_data(dataset: str, shuffle_samples=True, seed: int = seed):
    np.random.seed(seed)

    if dataset == 'MIMIC':
        data = pd.read_csv('dataset/MIMIC/mimic3d.csv')
        drop_cols = ['hadm_id', 'AdmitDiagnosis', 'AdmitProcedure', 'religion', 'insurance', 'ethnicity',
                     'marital_status', 'ExpiredHospital', 'LOSdays', 'gender', 'admit_type', 'admit_location']
        data.drop(drop_cols, inplace=True, axis=1)

        if shuffle_samples:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = data.drop(['LOSgroupNum'], axis=1).to_numpy()
        y = data['LOSgroupNum'].to_numpy()
        # y = (y > 1).astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif dataset == 'Breast':
        data = pd.read_csv('dataset/Breast/data.csv')
        data.drop(["Unnamed: 32", "id"], inplace=True, axis=1)
        data["diagnosis"] = [1 if i.strip() == "M" else 0 for i in data.diagnosis]

        if shuffle_samples:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = data.drop(["diagnosis"], axis=1).to_numpy()
        y = data['diagnosis'].to_numpy()

        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

    elif dataset == 'Credit':
        data = pd.read_csv('dataset/Credit/credit.csv')
        data.drop('ID', inplace=True, axis=1)

        # Shuffle samples if required
        if shuffle_samples:
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

        X = data.drop(['default payment next month'], axis=1).to_numpy()
        y = data['default payment next month'].to_numpy()

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    else:
        print('Wrong dataset!')
        return None, None

    return X, y


# Load the data
X, y = load_data('Credit')  # 'MIMIC' or 'Breast' or 'Credit'
########################################################################################################################
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=None)

# Initialize and train the MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='tanh', alpha=0.1, max_iter=1000)

model.fit(X_train, y_train)

# Calculate permutation importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=1000, random_state=None)

# Get feature names
# feature_names = pd.read_csv('dataset/Breast/data.csv').drop(["Unnamed: 32", "id", "diagnosis"], axis=1).columns
# feature_names = pd.read_csv('dataset/MIMIC/mimic3d.csv').drop(['hadm_id', 'AdmitDiagnosis', 'AdmitProcedure', 'religion', 'insurance', 'ethnicity',
#                      'marital_status', 'ExpiredHospital', 'LOSdays', 'gender', 'admit_type', 'admit_location'], axis=1).columns
feature_names = pd.read_csv('dataset/Credit/credit.csv').drop('ID', axis=1).columns

# Sort features by their importance
importance_sorted_idx = perm_importance.importances_mean.argsort()[::-1]
important_features = [(feature_names[idx], perm_importance.importances_mean[idx]) for idx in importance_sorted_idx]

# Display the feature importance
print("Feature importance, sorted from most to least important:")
print(important_features)

# the feature names in order, without importance values
feature_names_sorted = [feature_names[idx] for idx in importance_sorted_idx]
print("Sorted feature names from most to least important:")
print(feature_names_sorted)

# Breast dataset
# ['texture_mean', 'concavity_mean', 'concave points_worst', 'texture_worst', 'radius_se', 'area_se', 'symmetry_worst', 'perimeter_se', 'radius_worst', 'concave points_mean', 'fractal_dimension_mean', 'smoothness_se', 'concavity_worst', 'symmetry_se', 'fractal_dimension_worst', 'area_worst', 'compactness_mean', 'concave points_se', 'compactness_se', 'area_mean', 'texture_se', 'fractal_dimension_se', 'smoothness_worst', 'perimeter_worst', 'compactness_worst', 'radius_mean', 'perimeter_mean', 'smoothness_mean', 'symmetry_mean', 'concavity_se']
# ['area_worst', 'area_mean', 'area_se', 'radius_mean', 'compactness_se', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'symmetry_worst', 'fractal_dimension_worst', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'smoothness_se', 'perimeter_se', 'texture_worst', 'texture_mean', 'radius_worst', 'perimeter_worst', 'perimeter_mean']

# MIMIC dataset
# ['TotalNumInteract', 'NumChartEvents', 'NumInput', 'NumLabs', 'NumTransfers', 'NumNotes', 'NumOutput', 'NumDiagnosis', 'NumRx', 'NumProcEvents', 'NumCPTevents', 'age', 'NumMicroLabs', 'NumCallouts', 'NumProcs']
# ['NumTransfers', 'NumDiagnosis', 'NumCallouts', 'NumChartEvents', 'TotalNumInteract', 'NumProcEvents', 'age', 'NumInput', 'NumNotes', 'NumRx', 'NumMicroLabs', 'NumProcs', 'NumLabs', 'NumCPTevents', 'NumOutput']

# Credit dataset
# ['LIMIT_BAL', 'BILL_AMT4', 'BILL_AMT6', 'BILL_AMT5', 'BILL_AMT3', 'PAY_4', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'BILL_AMT1', 'PAY_5', 'PAY_6', 'PAY_AMT5', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT6']
# ['PAY_0', 'PAY_6', 'PAY_2', 'LIMIT_BAL', 'PAY_5', 'PAY_4', 'BILL_AMT4', 'EDUCATION', 'BILL_AMT3', 'AGE', 'BILL_AMT6', 'BILL_AMT2', 'BILL_AMT5', 'BILL_AMT1', 'PAY_AMT5', 'PAY_AMT4', 'PAY_AMT6', 'MARRIAGE', 'PAY_3', 'PAY_AMT3', 'PAY_AMT1', 'SEX', 'PAY_AMT2']
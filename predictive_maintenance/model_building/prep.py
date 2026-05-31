# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))
user_info = api.whoami()
HF_USER = user_info["name"]
print("Authenticated as:", HF_USER)


DATA_DIR = "predictive_maintenance/data"

# Repo identifiers used throughout the notebook
DATASET_REPO_ID = f"{HF_USER}/engine-predictive-maintenance"

DATASET_PATH = f"hf://datasets/{DATASET_REPO_ID}/engine_data.csv"
engine_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'engine_condition'

# List of features in the dataset (all engine sensor readings)
features = [
    'engine_rpm', 'lub_oil_pressure', 'fuel_pressure',
    'coolant_pressure', 'lub_oil_temp', 'coolant_temp'
]

# Define predictor matrix (X) using selected numeric and categorical features
X = engine_dataset[features]

# Define target variable
y = engine_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv(os.path.join(DATA_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(DATA_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(DATA_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(DATA_DIR, "ytest.csv"), index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=DATA_DIR / file_path,
        path_in_repo=file_path,  # just the filename
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
    )

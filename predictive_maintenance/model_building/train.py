import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, recall_score
import xgboost as xgb
import joblib
import mlflow
from huggingface_hub import login, HfApi
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("predictive-maintenance-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))
user_info = api.whoami()
HF_USER = user_info["name"]
print("Authenticated as:", HF_USER)

# Repo identifiers used throughout the notebook
DATASET_REPO_ID = f"{HF_USER}/engine-predictive-maintenance"
MODEL_REPO_ID = f"{HF_USER}/engine-predictive-maintenance-model"

# Load preprocessed data from Hugging Face
Xtrain_path = f"hf://datasets/{DATASET_REPO_ID}/Xtrain.csv"
Xtest_path = f"hf://datasets/{DATASET_REPO_ID}/Xtest.csv"
ytrain_path = f"hf://datasets/{DATASET_REPO_ID}/ytrain.csv"
ytest_path = f"hf://datasets/{DATASET_REPO_ID}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("Preprocessed data loaded successfully from Hugging Face.")

# All predictors are numeric engine-sensor readings — no categorical features
# (column names match the snake_case headers stored in data/Xtrain.csv)
numeric_features = [
    'engine_rpm', 'lub_oil_pressure', 'fuel_pressure',
    'coolant_pressure', 'lub_oil_temp', 'coolant_temp'
]

# Class-weight to handle the (mild) imbalance between healthy / faulty engines
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
print(f"Class weight (scale_pos_weight): {class_weight:.2f}")

# Preprocessing: scale all numeric sensor features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features)
)

# Base XGBoost classifier
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, verbosity=0,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators':      [50, 75, 100],
    'xgbclassifier__max_depth':         [2, 3, 4],
    'xgbclassifier__colsample_bytree':  [0.4, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.6],
    'xgbclassifier__learning_rate':     [0.01, 0.1],
    'xgbclassifier__reg_lambda':        [0.4, 0.6],
}

# Full model pipeline (preprocessor + classifier)
model_pipeline = make_pipeline(preprocessor, xgb_model)

print("Starting model training with GridSearchCV...")

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # Log the best run in the parent
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Evaluate the best model with a custom classification threshold
    best_model = grid_search.best_estimator_
    print(f"Best CV Score : {grid_search.best_score_:.4f}")
    print(f"Best Params   : {grid_search.best_params_}")

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True)

    mlflow.log_metrics({
        "train_accuracy":  train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall":    train_report['1']['recall'],
        "train_f1-score":  train_report['1']['f1-score'],
        "test_accuracy":   test_report['accuracy'],
        "test_precision":  test_report['1']['precision'],
        "test_recall":     test_report['1']['recall'],
        "test_f1-score":   test_report['1']['f1-score'],
    })

    # Save the model locally
    model_path = "engine-predictive-maintenance-model.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = MODEL_REPO_ID
    repo_type = "model"

    # Step 1: Check if the model repo exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repository '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id}' not found. Creating new repository...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repository '{repo_id}' created.")

    # Upload the model file to Hugging Face
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model successfully uploaded to Hugging Face!")

print("Model training and registration completed!")

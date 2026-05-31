from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))
user_info = api.whoami()
HF_USER = user_info["name"]
print("Authenticated as:", HF_USER)

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "predictive_maintenance" / "data"

# Repo identifiers used throughout the notebook
DATASET_REPO_ID = f"{HF_USER}/engine-predictive-maintenance"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset")
    print(f"Space '{DATASET_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{DATASET_REPO_ID}' not found. Creating new space...")
    create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", private=False)
    print(f"Space '{DATASET_REPO_ID}' created.")

api.upload_folder(
    folder_path=DATA_DIR,
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
)

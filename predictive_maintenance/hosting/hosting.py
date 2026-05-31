from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
user_info = api.whoami()
HF_USER = user_info["name"]
print("Authenticated as:", HF_USER)

# Repo identifiers used throughout the notebook
APP_REPO_ID = f"{HF_USER}/engine-predictive-maintenance"
 # Step 1: Check if the model repo exists
try:
    api.repo_info(repo_id=APP_REPO_ID, repo_type="space")
    print(f"Model repository '{APP_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repository '{APP_REPO_ID}' not found. Creating new repository...")
    create_repo(repo_id=APP_REPO_ID, repo_type="space", private=False)
    print(f"Model repository '{APP_REPO_ID}' created.")

api.upload_folder(
    folder_path="predictive_maintenance/deployment",     # the local folder containing your files
    repo_id=APP_REPO_ID,          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

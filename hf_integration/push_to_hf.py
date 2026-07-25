"""
Script to push your trained Anomaly Detection model to Hugging Face Model Hub.

Pre-requisites:
1. Install huggingface_hub: pip install huggingface_hub
2. Authenticate via terminal: huggingface-cli login
   (Paste your write-access API token from https://huggingface.co/settings/tokens)
"""

import os
# pyrefly: ignore [missing-import]
from huggingface_hub import HfApi, create_repo

def push_model_to_hf(model_path: str, repo_id: str):
    """
    Pushes the trained model weights file to Hugging Face Model Hub.
    
    Args:
        model_path (str): Local path to your saved model (e.g. 'best_model.keras')
        repo_id (str): Your Hugging Face repo ID (e.g. 'your-username/ai-anomaly-detection')
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return

    api = HfApi()
    
    print(f"Creating / verifying Hugging Face repository '{repo_id}'...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("Repository ready!")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print(f"Uploading '{model_path}' to Hugging Face repository '{repo_id}'...")
    try:
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            repo_type="model"
        )
        print("\nSuccess! Model uploaded to Hugging Face Hub.")
        print(f"URL: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading model: {e}")

if __name__ == "__main__":
    MODEL_FILE = "best_model.keras"  # Or "improved_best_model.keras"
    HF_REPO_ID = "SantoshDN/ai-enable-anomaly-detection"
    
    print("=== Hugging Face Model Push Utility ===")
    print(f"Target Repo: {HF_REPO_ID}")
    print(f"Target File: {MODEL_FILE}")
    print("---------------------------------------")
    push_model_to_hf(MODEL_FILE, HF_REPO_ID)

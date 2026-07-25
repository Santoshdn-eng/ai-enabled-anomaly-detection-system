"""
Script to pull/download the trained model weights from Hugging Face Hub in Python / Backend API.
"""

import os
# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

def pull_model_from_hf(repo_id: str, filename: str = "best_model.keras", save_dir: str = "./models") -> str:
    """
    Downloads the trained model weights from Hugging Face Hub and saves locally.
    
    Returns:
        str: Absolute local filepath of the downloaded model.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading '{filename}' from Hugging Face repository '{repo_id}'...")
    
    try:
        # pyrefly: ignore [no-matching-overload]
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded model to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"Failed to pull model from Hugging Face Hub: {e}")
        # pyrefly: ignore [bad-return]
        return None

if __name__ == "__main__":
    HF_REPO_ID = "SantoshDN/ai-enable-anomaly-detection"
    local_file = pull_model_from_hf(HF_REPO_ID)
    print(f"Model path ready for inference: {local_file}")

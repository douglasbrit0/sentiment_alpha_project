from huggingface_hub import login, HfApi, create_repo, upload_folder
import os

# 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

def upload_to_huggingface(local_model_path: str, repo_name: str, hf_token: str, private: bool = False):
    """
    Uploads a local model directory to the Hugging Face Model Hub.

    Args:
        local_model_path (str): Path to the local directory
        repo_name (str): Your HF username and repo name 
        hf_token (str): Hugging Face access token
        private (bool): Whether the repo should be private

    Returns:
        model_url (str): Full URL to the uploaded model
    """

    login(token = hf_token)

    create_repo(repo_id = repo_name,
                token = hf_token,
                repo_type = "model",
                private = private,
                exist_ok = True
        )
    
    upload_folder(
        folder_path=local_model_path,
        repo_id=repo_name,
        token=hf_token
    )

    return f"https://huggingface.co/{repo_name}"
    
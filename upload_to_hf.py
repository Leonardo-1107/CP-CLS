from huggingface_hub import upload_folder

upload_folder(
    folder_path="./models",
    repo_id="CVPR-SMILE/SMILE_mini",
    repo_type="model",
    path_in_repo="classifier",
    commit_message="upload monai classifier",
    revision="main"  # or "v0.1" for a tagged version
)
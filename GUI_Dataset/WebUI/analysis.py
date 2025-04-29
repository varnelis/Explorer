from huggingface_hub import snapshot_download
from pathlib import Path


while True:
    try:
        dir = Path(__file__).parent
        local_dir = dir / Path("data-7k")
        snapshot_download(
            repo_id="biglab/webui-7k",
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        break
    except ConnectionError:
        pass

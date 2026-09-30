"""Download the two local models used by the console router."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E5_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
QWEN_REVISION = "73e3e38d981303bc594367cd910ea6eb48349da8"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download router models / Скачать модели роутера")
    parser.add_argument("--only", choices=("all", "e5", "qwen"), default="all")
    parser.add_argument("--lang", choices=("ru", "en"), default="ru")
    args = parser.parse_args(argv)
    if args.only in ("all", "e5"):
        from sentence_transformers import SentenceTransformer
        target = ROOT / "models" / "multilingual-e5-small"
        print("Downloading E5..." if args.lang == "en" else "Скачиваю E5...")
        SentenceTransformer("intfloat/multilingual-e5-small", revision=E5_REVISION).save(str(target))
        print(target)
    if args.only in ("all", "qwen"):
        from huggingface_hub import snapshot_download
        target = ROOT / "models" / "qwen3-0.6b-4bit"
        print("Downloading Qwen..." if args.lang == "en" else "Скачиваю Qwen...")
        snapshot_download(repo_id="mlx-community/Qwen3-0.6B-4bit",
                          revision=QWEN_REVISION, local_dir=target)
        print(target)


if __name__ == "__main__":
    main()

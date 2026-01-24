import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "indices"
MODEL_DIR = DATA_DIR / "models"
IMAGE_DIR = DATA_DIR / "images"
UPLOAD_DIR = IMAGE_DIR / "uploads"

for directory in [DATA_DIR, INDEX_DIR, MODEL_DIR, IMAGE_DIR, UPLOAD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


CLIP_MODEL_PATH = MODEL_DIR / "clipViTb32"
DINO_MODEL_PATH = MODEL_DIR / "dinoV2b14reg"


CLIP_ONLINE_ID = "openai/clip-vit-base-patch32"
DINO_ONLINE_ID = "facebook/dinov2-with-registers-base"


PHASH_BITS = 64
WHASH_BITS = 64
CLIP_DIM = 768
DINO_DIM = 768


PHASH_THRESHOLD = 4
WHASH_THRESHOLD = 4
CLIP_THRESHOLD = 0.59
DINO_THRESHOLD = 0.55
HIST_THRESHOLD = 0.80
STRUCTURE_CHECK_THRESHOLD = 3

# download_and_place.py
from pathlib import Path
import shutil
import kagglehub

# HANDLE = "orvile/crema-d-emotional-multimodal-dataset"
HANDLE = "reganw/cmu-mosi"
# DEST   = Path("/home/kiyoya/research/robust-mer/ROBUST-MER/data/CREMA-D/raw")
DEST   = Path("/home/kiyoya/research/robust-mer/ROBUST-MER/data/MOSI/raw")

# 1) まずKaggleHubのキャッシュにDL（成功すると展開先のパスが返る）
src = Path(kagglehub.dataset_download(HANDLE))
print("cached at:", src)

# 2) 目的地へコピー（上書き許可）
DEST.mkdir(parents=True, exist_ok=True)
for p in src.iterdir():
    dst = DEST / p.name
    if p.is_dir():
        shutil.copytree(p, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(p, dst)

print("✅ placed to:", DEST)

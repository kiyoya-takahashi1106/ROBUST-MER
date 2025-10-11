"""
CREMA-D を指定フォルダに自動ダウンロード & 解凍 (.env対応)
— 先に .env を読み、環境変数をセットしてから kaggle を import する —
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ================== 設定 ==================
REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = (REPO_ROOT / "data" / "CREMA-D" / "raw").resolve()
DATASET = "orvile/crema-d-emotional-multimodal-dataset"
SKIP_IF_EXISTS = True
# =========================================


def load_credentials():
    # .env を「スクリプト隣 → リポジトリ直下」の順で探す
    for env_path in [Path(__file__).parent / ".env", REPO_ROOT / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            print(f"✅ .env を読み込みました: {env_path}")
            break
    else:
        print("⚠️ .env が見つかりません。既存の環境変数を利用します。")

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        raise EnvironmentError(
            "❌ Kaggle 認証情報が見つかりません。\n"
            "  `.env`（スクリプト隣 or リポジトリ直下）に以下を追加してください:\n\n"
            "  KAGGLE_USERNAME=あなたのKaggleユーザー名\n"
            "  KAGGLE_KEY=あなたのAPIキー\n"
        )

    # ← ここで **確実に** 環境変数へ反映（kaggle import 前に！）
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    print(f"🔐 認証ユーザー: {username}")


def already_extracted(path: Path) -> bool:
    # 代表的なフォルダがあれば解凍済みとみなす（必要に応じ調整）
    for p in [
        path / "AudioWAV",
        path / "AudioMP4",
        path / "VideoFlash",
        path / "VideoMP4",
        path / "processedResults",
    ]:
        if p.exists():
            return True
    return False

def main():
    # 1) 認証情報をロードして環境変数にセット
    load_credentials()

    # 2) ここで初めて kaggle を import（認証情報セット後）
    from kaggle.api.kaggle_api_extended import KaggleApi

    # 3) 保存先
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 ダウンロード先: {SAVE_DIR}")

    # 4) 認証 & ダウンロード
    api = KaggleApi()
    api.authenticate()

    print(f"📦 Kaggle データセット: {DATASET}")
    print("⬇️ ダウンロード（自動解凍）開始...")
    api.dataset_download_files(
        dataset=DATASET,
        path=str(SAVE_DIR),
        unzip=True,
        quiet=False
    )
    print("✅ CREMA-D のダウンロードと解凍が完了しました！")


if __name__ == "__main__":
    main()

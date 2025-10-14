#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOSI raw mp4 を segment/{train,valid,test}/{audio,text,video} に整形するスクリプト
- 入力:  data/MOSI/raw/<video_id>/<clip_id>.mp4
- ラベル: label.csv (columns: video_id,clip_id,text,label,label_T,label_A,label_V,annotation,mode)
- 出力:  data/MOSI/segment/<mode>/{audio,text,video}/<video_id>-<clip_id>.(wav|txt|flv)

必要:
  - Python: pandas
  - ffmpeg（コマンドが使える状態）
"""

import subprocess
from pathlib import Path
import pandas as pd


# ==== 設定 ====
ROOT = Path("../data/MOSI")
RAW_DIR = ROOT / "raw"          # raw/<video_id>/<clip_id>.mp4
LABEL_CSV = ROOT / "raw" / "label.csv"   # 提示の label.txt を CSV として保存したものを想定
OUT_BASE = ROOT / "segment"     # segment/{train,valid,test}/{audio,text,video}



# ffmpeg のコマンド雛形
def run(cmd):
    subprocess.run(cmd, check=True)



def ensure_dirs(base_mode_dir: Path):
    (base_mode_dir / "audio").mkdir(parents=True, exist_ok=True)
    (base_mode_dir / "text").mkdir(parents=True, exist_ok=True)
    (base_mode_dir / "video").mkdir(parents=True, exist_ok=True)



def main():
    assert LABEL_CSV.exists(), f"ラベルファイルがありません: {LABEL_CSV.resolve()}"
    df = pd.read_csv(LABEL_CSV)

    # 必須列チェック
    required_cols = {"video_id", "clip_id", "text", "mode"}
    missing = required_cols - set(df.columns)
    assert not missing, f"label.csv に必須列が不足: {missing}"

    # clip_id は整数/文字どちらでも OK、文字→数字も許容
    # ここでは文字列化してそのまま使う（ゼロ詰め等は要求なし）
    for i, row in df.iterrows():
        video_id = str(row["video_id"])
        clip_id  = str(row["clip_id"])
        text     = "" if pd.isna(row["text"]) else str(row["text"])
        mode     = str(row["mode"]).strip().lower()  # train/valid/test

        # 出力先ディレクトリ作成
        mode_dir = OUT_BASE / mode
        ensure_dirs(mode_dir)

        # 入力 mp4
        in_mp4 = RAW_DIR / video_id / f"{clip_id}.mp4"
        if not in_mp4.exists():
            print(f"[WARN] 入力が見つかりません: {in_mp4}")
            continue

        # 出力パス
        stem = f"{video_id}-{clip_id}"
        out_wav = mode_dir / "audio" / f"{stem}.wav"
        out_txt = mode_dir / "text"  / f"{stem}.txt"
        out_flv = mode_dir / "video" / f"{stem}.flv"

        # --- audio: mp4 -> wav (mono, 16kHz) ---
        if not out_wav.exists():
            cmd_audio = [
                "ffmpeg", "-y",
                "-i", str(in_mp4),
                "-ac", "1",           # mono
                "-ar", "16000",       # 16kHz
                "-vn",                # no video
                str(out_wav),
            ]
            try:
                run(cmd_audio)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] 音声変換失敗: {in_mp4} -> {out_wav} ({e})")
                continue

        # --- video: mp4 -> flv（映像のみ；音声削除）---
        if not out_flv.exists():
            cmd_video = [
                "ffmpeg", "-y",
                "-i", str(in_mp4),
                "-an",                # drop audio track
                "-c:v", "libx264",    # H.264 で再エンコード（互換性重視）
                "-preset", "veryfast",
                "-pix_fmt", "yuv420p",
                "-f", "flv",
                str(out_flv),
            ]
            try:
                run(cmd_video)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] 動画変換失敗: {in_mp4} -> {out_flv} ({e})")
                continue

        # --- text: label.csv -> txt ---
        if not out_txt.exists():
            try:
                out_txt.write_text(text.strip() + "\n", encoding="utf-8")
            except Exception as e:
                print(f"[ERROR] テキスト書き込み失敗: {out_txt} ({e})")
                continue

        print(f"[OK] {mode}: {stem}")

    print("\n完了！出力ルート:", OUT_BASE.resolve())



if __name__ == "__main__":
    main()

import os
import shutil
from pathlib import Path
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm
from datasets import load_dataset
import gdown
import subprocess
import requests


RAW_DIR = Path("./datasets/raw")
DATASET_LINKS = [r"/mnt/d/picture",
                "https://huggingface.co/datasets/PedroSampaio/fruits-360",
                "https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection"]

def download_http(url, save_dir):
    filename = os.path.basename(url)
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_dir / filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        return True
    return False

from datasets import load_dataset, concatenate_datasets
from PIL import Image

def download_huggingface(url, save_dir):
    """
    Tải dataset từ Hugging Face bằng datasets.load_dataset.
    - url có thể là "https://huggingface.co/datasets/owner/repo" hoặc "owner/repo".
    - Lưu tất cả ảnh vào save_dir/<label>/...
    - Trả về True/False.
    """
    try:
        # 1) Chuẩn hóa dataset_id từ URL hoặc trả nguyên nếu đã là "owner/repo"
        if "huggingface.co" in url:
            if "/datasets/" in url:
                dataset_id = url.split("/datasets/")[-1].strip("/")
            else:
                # dự phòng: lấy phần cuối làm id
                dataset_id = url.rstrip("/").split("/")[-1]
        else:
            dataset_id = url

        print(f"🔽 Đang tải dataset Hugging Face: {dataset_id}")

        # 2) Load dataset (tải tất cả splits)
        dataset = load_dataset(dataset_id)
        # Nếu trả về DatasetDict (multiple splits), gộp lại
        if isinstance(dataset, dict) or hasattr(dataset, "keys"):
            # dataset có thể là DatasetDict — gộp tất cả split thành 1 Dataset
            try:
                dataset = concatenate_datasets([dataset[s] for s in dataset.keys()])
            except Exception as e:
                print("⚠️ Không thể concatenate splits trực tiếp:", e)
                # fallback: nếu dataset là DatasetDict, lặp từng split lưu ảnh từng split
                # (nhưng ở đây ta cố gắng gộp; chỉ proceed nếu gộp được)
                raise

        # 3) đảm bảo thư mục đích tồn tại
        save_dir.mkdir(parents=True, exist_ok=True)

        # 4) lưu ảnh: xử lý nhiều dạng field 'image'
        saved = 0
        for i, sample in enumerate(dataset):
            try:
                # Có thể image có ở key 'image' hoặc 'img' — thử nhiều khả năng
                img_field = None
                for key in ("image", "img", "image_path", "file"):
                    if key in sample:
                        img_field = sample[key]
                        break

                if img_field is None:
                    # Nếu không có key image, bỏ qua
                    print(f"⚠️ Mẫu {i} không có trường ảnh, bỏ qua.")
                    continue

                # Nếu là PIL Image
                if isinstance(img_field, Image.Image):
                    img = img_field
                else:
                    # Có thể là datasets Image (PILWrapper) -> tải bằng .to_pil()
                    try:
                        img = img_field.to_pil()  # một số version hỗ trợ
                    except Exception:
                        # Có thể là dict chứa 'path' hoặc bytes
                        if isinstance(img_field, dict) and "path" in img_field:
                            img = Image.open(img_field["path"])
                        elif isinstance(img_field, str) and os.path.exists(img_field):
                            img = Image.open(img_field)
                        else:
                            # cuối cùng, thử convert trực tiếp (nếu là numpy array)
                            import numpy as np
                            if isinstance(img_field, np.ndarray):
                                img = Image.fromarray(img_field)
                            else:
                                print(f"⚠️ Không xác định được dạng ảnh cho mẫu {i}, bỏ qua.")
                                continue

                # đảm bảo RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # label (nếu có)
                label = str(sample.get("label", "unknown"))
                label_dir = save_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                # tên file: dataset_id an toàn (thay / thành _)
                safe_dataset_id = dataset_id.replace("/", "_").replace(" ", "_")
                file_name = f"{safe_dataset_id}_{i:06d}.jpg"
                out_path = label_dir / file_name

                # lưu ảnh (đảm bảo thư mục cha đã tồn tại)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(out_path)
                saved += 1

            except Exception as e:
                print(f"❌ Lỗi lưu ảnh mẫu {i}: {e}")
                continue

        print(f"✅ Hoàn tất: đã lưu {saved} ảnh từ {dataset_id} vào {save_dir}")
        return True

    except Exception as e:
        print("❌ HF error:", e)
        return False

def download_kaggle(url, save_dir):
    try:
        parts = url.split("/datasets/")[-1]
        subprocess.run(["kaggle", "datasets", "download", "-d", parts, "-p", str(save_dir)], check=True)
        return True
    except Exception as e:
        print("❌ Kaggle error:", e)
        return False

def download_gdrive(url, save_dir):
    try:
        gdown.download(url, output=str(save_dir), quiet=False, fuzzy=True)
        return True
    except Exception as e:
        print("❌ Drive error:", e)
        return False

def copy_local(path, save_dir):
    p = Path(path)
    if p.is_dir():
        for f in p.rglob("*.*"):
            shutil.copy(f, save_dir / f.name)
    elif p.is_file():
        shutil.copy(p, save_dir / p.name)

def detect_source(link):
    if "huggingface.co" in link:
        return "huggingface"
    elif "kaggle.com" in link:
        return "kaggle"
    elif "drive.google.com" in link:
        return "gdrive"
    elif link.startswith("http"):
        return "http"
    else:
        return "local"

def remove_duplicates(folder):
    print("🧹 Đang kiểm tra ảnh trùng lặp...")
    hashes = {}
    removed = 0

    for img_path in tqdm(list(Path(folder).rglob("*.jpg")), desc="Đang quét ảnh"):
        try:
            img = Image.open(img_path)
            img_hash = imagehash.phash(img)

            if img_hash in hashes:
                # Nếu hash đã tồn tại → ảnh trùng → xoá đi
                os.remove(img_path)
                removed += 1
            else:
                hashes[img_hash] = img_path
        except Exception as e:
            print(f"Lỗi khi xử lý {img_path}: {e}")

    print(f"✅ Đã xoá {removed} ảnh trùng lặp")

def main():
    for link in tqdm(DATASET_LINKS, desc="Đang xử lý dataset"):
        source = detect_source(link)
        print(f"📦 Nguồn: {source} → {link}")

        if source == "huggingface":
            download_huggingface(link, RAW_DIR)
        elif source == "kaggle":
            download_kaggle(link, RAW_DIR)
        elif source == "gdrive":
            download_gdrive(link, RAW_DIR)
        elif source == "http":
            download_http(link, RAW_DIR)
        elif source == "local":
            copy_local(link, RAW_DIR)
    remove_duplicates(RAW_DIR)

if __name__ == "__main__":
        main()
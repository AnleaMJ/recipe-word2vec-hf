# data_loader.py
import csv
import zipfile
from pathlib import Path

import gdown

DATASET_NAME = "m3hrdadfi/recipe_nlg_lite"
GDRIVE_FILE_ID = "1PGH5H_oW7wUvMw_5xaXvbEN7DFll-wDX"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "recipe_nlg_lite.zip"
CSV_DIR = DATA_DIR / "recipe_nlg_lite"
SPLITS = ("train", "test")


def ensure_raw_data():
    if all((CSV_DIR / f"{s}.csv").exists() for s in SPLITS):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        print(f"Downloading {DATASET_NAME} archive ...")
        gdown.download(url, str(ZIP_PATH), quiet=False)
    if not CSV_DIR.exists():
        print("Extracting ...")
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(DATA_DIR)


def _split_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]


def _split_steps(value) -> list[str]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    parts = [p.strip() for p in text.split(". ") if p.strip()]
    if parts and not parts[-1].endswith("."):
        parts[-1] += "."
    return parts


def load_split(split: str = "train") -> list[dict]:
    ensure_raw_data()
    path = CSV_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, quotechar='"', delimiter="\t")
        for row in reader:
            rows.append({
                "uid": row.get("uid"),
                "name": (row.get("name") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "link": row.get("link"),
                "ner": _split_list(row.get("ner")),
                "ingredients": _split_list(row.get("ingredients")),
                "steps": _split_steps(row.get("steps")),
            })
    return rows


def load_all() -> dict[str, list[dict]]:
    return {s: load_split(s) for s in SPLITS}


if __name__ == "__main__":
    data = load_split("train")
    print(f"train recipes: {len(data)}")
    sample = data[0]
    print("name:", sample["name"])
    print("ner[:3]:", sample["ner"][:3])
    print("ingredients[:3]:", sample["ingredients"][:3])
    print("steps[:3]:", sample["steps"][:3])

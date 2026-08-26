# data_loader.py
import csv
import re
import zipfile
from pathlib import Path

import gdown

DATASET_NAME = "m3hrdadfi/recipe_nlg_lite"
INDIAN_CSV_URL = "https://raw.githubusercontent.com/nileshely/Indian-Food/main/IndianFoodDataset.csv"
GDRIVE_FILE_ID = "1PGH5H_oW7wUvMw_5xaXvbEN7DFll-wDX"
DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "recipe_nlg_lite.zip"
CSV_DIR = DATA_DIR / "recipe_nlg_lite"
INDIAN_CSV = DATA_DIR / "IndianFoodDataset.csv"
SPLITS = ("train", "test")

UNIT_RE = re.compile(
    r"^\s*[\d/\.\-\s]+(?:cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|"
    r"gram|grams|kilogram|kg|ml|litre|liter|pinch|inch|inches|cloves|"
    r"sprig|sprigs|small|medium|large|whole|sliced|chopped|finely|"
    r"roughly|roughly|to taste|as required|as needed)\s*",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"\s*\([^)]*\)")
PREP_RE = re.compile(r"\s*-\s+.*$")


def _clean_ingredient(raw: str) -> str:
    name = raw.strip()
    name = PREP_RE.sub("", name)
    name = PAREN_RE.sub("", name)
    name = re.sub(r"^\s*[\d/\.\-]+\s*", "", name)
    for unit in (
        "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
        "gram", "grams", "kilogram", "kg", "ml", "litre", "liter", "pinch",
        "inch", "inches", "cloves", "sprig", "sprigs",
    ):
        pattern = re.compile(rf"^\s*{unit}\s+", re.IGNORECASE)
        name = pattern.sub("", name)
    name = name.strip(" -,")
    return name.lower()


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


def ensure_indian_data():
    if INDIAN_CSV.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading Indian food dataset ...")
    gdown.download(INDIAN_CSV_URL, str(INDIAN_CSV), quiet=False)


def _split_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    items = [p.strip() for p in text.split(",") if p.strip()]
    return [item for item in items if item.lower() != "none"]


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


def load_indian() -> list[dict]:
    ensure_indian_data()
    rows = []
    with open(INDIAN_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_ing = row.get("TranslatedIngredients") or row.get("Ingredients") or ""
            ingredients = [_clean_ingredient(x) for x in raw_ing.split(",") if x.strip()]
            ingredients = [i for i in ingredients if len(i) > 2]

            raw_steps = row.get("TranslatedInstructions") or row.get("Instructions") or ""
            steps = _split_steps(raw_steps)

            rows.append({
                "uid": row.get("\ufeffSrno") or row.get("Srno"),
                "name": (row.get("TranslatedRecipeName") or row.get("RecipeName") or "").strip(),
                "description": "",
                "link": row.get("URL"),
                "ner": ingredients,
                "ingredients": ingredients,
                "steps": steps,
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

    print()
    indian = load_indian()
    print(f"indian recipes: {len(indian)}")
    s2 = indian[0]
    print("name:", s2["name"])
    print("ingredients[:5]:", s2["ingredients"][:5])
    print("steps[:3]:", s2["steps"][:3])

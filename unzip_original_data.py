import zipfile

from config import DATA

DATA.mkdir(parents=True, exist_ok=True)
path_to_zip_file = DATA / "original_data" / "data.zip"
directory_to_extract_to = DATA / "original_data"

with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

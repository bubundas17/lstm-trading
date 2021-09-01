import wget
from Config import *
from pathlib import Path
import os.path
import zipfile

START_YEAR = 2018
START_MONTH = 8

END_YEAR = 2021
END_MONTH = 7

files = []

# create Url list
for x in range(START_YEAR, END_YEAR + 1):
    start = 1
    end = 12
    if(x == START_YEAR):
        start = START_MONTH
    
    if(x == END_YEAR):
        end = END_MONTH
    
    for y in range(start, end+1):
        files.append({"url": f"https://data.binance.vision/data/spot/monthly/klines/{COIN_PEAR}/1m/{COIN_PEAR}-1m-{x}-{y:02}.zip", "name": f"{COIN_PEAR}-1m-{x}-{y:02}.zip"})

print("TOTAl FILES:", len(files))

raw_dataset_path = "raw_datasets/"
Path(raw_dataset_path).mkdir(parents=True, exist_ok=True) # Ensure Directory

# Download Files
for file in files:
    file_path = f"{raw_dataset_path}{file['name']}"
    if(os.path.isfile(file_path)):
        print(f"Already Downloaded: {file['name']}")
        continue
    
    print(f"\nDownloading: {file['name']}")
    wget.download(file["url"], file_path)

# Unzip CSV files
dataset_path = f"datasets/{COIN_PEAR}/"
Path(dataset_path).mkdir(parents=True, exist_ok=True) # Ensure Directory
for file in files:
    file_path = f"{raw_dataset_path}{file['name']}"
    print(f"Unpacking Files: {file['name']}")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
# Download The files
# wget.download('')

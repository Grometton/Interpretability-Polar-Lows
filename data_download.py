import os
from urllib import request
import zipfile 

'''
def download_and_extract_data(data_path="data/"):
    if not os.path.exists(data_path + "train/") and not os.path.exists(data_path + "test/"):
        os.makedirs(data_path, exist_ok=True)
        print("Downloading training set...")
        request.urlretrieve("https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/QHV7PJ", data_path + "train.zip")
        with zipfile.ZipFile(data_path + "/train.zip", 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Downloading test set...")
        request.urlretrieve("https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/Z7JPFT", data_path + "test.zip")
        with zipfile.ZipFile(data_path + "/test.zip", 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Done.")
    else:
        print("Data already downloaded and extracted.")
'''

#!/usr/bin/env python3
import os
import sys
from urllib import request
import zipfile

def download_and_extract_data(data_path:str):
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    
    if not os.path.exists(train_dir) and not os.path.exists(test_dir):
        os.makedirs(data_path, exist_ok=True)
        
        # Download and extract training set
        print("Downloading training set...")
        train_zip_path = os.path.join(data_path, "train.zip")
        request.urlretrieve(
            "https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/QHV7PJ",
            train_zip_path
        )
        with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # Download and extract test set
        print("Downloading test set...")
        test_zip_path = os.path.join(data_path, "test.zip")
        request.urlretrieve(
            "https://dataverse.no/api/access/datafile/:persistentId?persistentId=doi:10.18710/FV5T9U/Z7JPFT",
            test_zip_path
        )
        with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        print("Done.")
    else:
        print(f"Data already downloaded and extracted in {data_path}.")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 data_download.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    download_and_extract_data(data_path)

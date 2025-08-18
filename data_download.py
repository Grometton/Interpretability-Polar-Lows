import os
from urllib import request
import zipfile 


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
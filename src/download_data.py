from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import kaggle

print("Kaggle version")
print(kaggle.__version__)

api = KaggleApi()
api.authenticate()

#download Files

api.dataset_download_files(
    "snehaanbhawal/resume-dataset",
    path="./Files",
    unzip=True
)

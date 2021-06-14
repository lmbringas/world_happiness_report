import os
import pickle
import threading

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import MinMaxScaler

from processing.process_dataset import DATASET_DIRECTORY
from processing.run_pipeline import run_pipeline
from processing.utils import create_directory

app = FastAPI()

create_directory("../static")

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/static"), name="static")


@app.get("/datasets")
async def list_dataset():
    datasets_folders = os.listdir(DATASET_DIRECTORY)
    datasets = []
    for dataset in datasets_folders:
        dataset_splited = dataset.split("__")
        print(dataset_splited[-1] == "finished")
        if dataset_splited[-1] == "finished":
            datasets.append(dataset_splited[0])
    return {"datasets": datasets}


@app.get("/years/")
async def list_years(dataset_name: str):
    path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/{dataset_name}.csv"
    dataframe = pd.read_csv(path)
    years = [int(x) for x in dataframe["year"].unique()]
    years.sort()
    return {"years": years}


@app.get("/data/")
async def get_data(dataset_name: str, year: int):
    dataframe_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/{dataset_name}.csv"
    som_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/{year}__{dataset_name}.pickle"
    model_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/model.pickle"
    dataframe = pd.read_csv(dataframe_path)
    som = None
    model = None
    with open(som_path, "rb") as f:
        som = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    clusters = som.hdbscan()[0]
    df = model.experiment_df
    print(model.ignored_columns)
    scaler = MinMaxScaler()
    year_mask = df["year"] == year
    data_values = df[year_mask].drop(columns=model.ignored_columns).values
    experiment_values = scaler.fit_transform(data_values)
    print(som.variables)

    print(np.flip(np.unique(clusters)))
    return {"data": ""}


@app.post("/upload_report/")
async def image(dataset: UploadFile = File(...)):
    pipeline = threading.Thread(target=run_pipeline, args=(dataset,), daemon=True)
    pipeline.start()
    pipeline.join()
    return {"okey": "polilla"}

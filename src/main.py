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
    som_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/{year}__{dataset_name}.pickle"
    model_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/model.pickle"
    som = None
    model = None
    with open(som_path, "rb") as f:
        som = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    clusters = som.hdbscan()[0]
    df = model.experiment_df

    scaler = MinMaxScaler()
    year_mask = df["year"] == year
    data_values = df[year_mask].drop(columns=model.ignored_columns).values
    experiment_values = scaler.fit_transform(data_values)

    columns = df.drop(columns=model.ignored_columns).columns.to_list()
    columns.insert(0, "Life Ladder")

    return_data = []
    for cluster in np.flip(np.unique(clusters)):
        cluster_name = f"Cluster {cluster}" if cluster != -1 else "Outliers"
        print(cluster_name)

        data = {"name": cluster_name, "countries": []}

        year_mask = model.imputed_df["year"] == year

        countries = som.map_attachments(
            experiment_values,
            model.imputed_df[year_mask]["Country name"].tolist(),
        )[clusters == cluster]

        activations = som.map_attachments(
            experiment_values,
            model.imputed_df[year_mask].drop(columns=model.ignored_columns).values,
        )[clusters == cluster]

        for countriy_list, activation_list in zip(countries, activations):

            for country, activation in zip(countriy_list, activation_list):
                results = {perspective: act for perspective, act in zip(columns, activation)}
                results["code"] = df[df["Country name"] == country].country_name_alpha_3.to_list()[
                    0
                ]

                data["countries"].append(results)
        return_data.append(data)
    return {"data": return_data}


@app.post("/upload_report/")
async def upload_report(dataset: UploadFile = File(...)):
    pipeline = threading.Thread(target=run_pipeline, args=(dataset,), daemon=True)
    pipeline.start()
    return {"okey": "polilla"}


@app.get("/report/")
async def get_report(dataset_name: str, year: int):
    som_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/{year}__{dataset_name}.pickle"
    model_path = f"{DATASET_DIRECTORY}/{dataset_name}__finished/model.pickle"
    som = None
    model = None
    with open(som_path, "rb") as f:
        som = pickle.load(f)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    clusters = som.hdbscan()[0]
    df = model.experiment_df

    scaler = MinMaxScaler()
    year_mask = df["year"] == year
    data_values = df[year_mask].drop(columns=model.ignored_columns).values
    experiment_values = scaler.fit_transform(data_values)

    columns = df.drop(columns=model.ignored_columns).columns
    year_mask = model.imputed_df["year"] == year

    result = []
    for cluster in np.flip(np.unique(clusters)):
        cluster_name = f"Cluster {cluster}" if cluster != -1 else "Outliers"
        countries = np.concatenate(
            som.map_attachments(
                experiment_values, model.imputed_df[year_mask]["Country name"].tolist()
            )[clusters == cluster].ravel()
        )

        mapped_values = []
        for mv in som.map_attachments(experiment_values, experiment_values)[clusters == cluster]:
            arr = []
            for values in mv:
                results = {perspective: value for perspective, value in zip(columns, values)}
                arr.append(results)
            mapped_values += arr

        mapped_values = np.array(mapped_values)
        data = {
            "name": cluster_name,
            "countries": countries.tolist(),
            "values": mapped_values.tolist(),
        }
        print(data)
        result.append(data)
    return {"data": result}

import os

from fastapi import File, UploadFile
from models.neuronal_map_model import NeuronalMapModel

from processing.process_dataset import process_dataset
from processing.utils import create_directory


def save_dataset(dataset, filename):
    path = "../datasets/"
    create_directory(path)
    full_path = path + filename
    with open(full_path, "wb+") as f:
        f.write(dataset.file.read())
        f.close()
    return full_path


def run_pipeline(dataset: UploadFile = File(...)):
    filename = dataset.filename.replace(" ", "_")
    path = save_dataset(dataset, filename)
    # cleaning data
    dataframe = process_dataset(path, filename)

    # build model
    model = NeuronalMapModel(dataframe, filename)
    model.train()

import os
import re

import pycountry
from pandas import DataFrame
from sklearn.impute import KNNImputer


def clean_dataset(dataset: DataFrame):
    country_column_name = "Country name"
    copy_dataset = dataset.copy()

    regions = [
        "Somaliland region",
        "Congo (Kinshasa)",
        "Hong Kong S.A.R. of China",
        "North Cyprus",
    ]

    for region in regions:
        region_mask = copy_dataset[country_column_name] == region
        copy_dataset = copy_dataset[~region_mask]

    return copy_dataset


def clean_country_names(name: str) -> str:
    mappeded_names = {
        "Taiwan Province of China": "Taiwan",
        "Ivory Coast": "Côte d'Ivoire",
        "Laos": "LAO PEOPLE'S DEMOCRATIC REPUBLIC",
        "Palestinian Territories": "Palestine",
        "South Korea": "Korea, Republic of",
        "Swaziland": "Eswatini",
    }
    regex = re.compile(".*?\((.*?)\)")
    result = re.findall(regex, name)
    if len(result) > 0:
        return name.replace(f"({result[0]})", "")
    result_name = mappeded_names.get(name, name)
    return result_name


def country_search_fuzzy(country_name: str) -> str:
    names = pycountry.countries.search_fuzzy(country_name)
    if len(names) > 0:
        return names[0].alpha_3
    return country_name


def create_directory(path: str):
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)


def save_dataframe(df: DataFrame, folder: str, filename: str):
    create_directory(folder)
    path = f"{folder}/{filename}.csv"
    df.to_csv(path)

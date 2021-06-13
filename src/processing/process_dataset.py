import pandas as pd

from processing.utils import (
    clean_country_names,
    clean_dataset,
    country_search_fuzzy,
    save_dataframe,
)

DATASET_DIRECTORY = "../process_dataset"


def process_dataset(dataset_path: str, filename: str):
    country_column_name = "Country name"
    df = pd.read_excel(dataset_path)
    clean_df = clean_dataset(df)
    clean_df["clean_country_names"] = clean_df[country_column_name].apply(clean_country_names)
    clean_df["country_name_alpha_3"] = clean_df["clean_country_names"].apply(country_search_fuzzy)
    path = f"{DATASET_DIRECTORY}/{filename}"
    save_dataframe(clean_df, path, filename)
    return clean_df

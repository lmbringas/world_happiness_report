import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from neural_map import NeuralMap, _plot
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler


class NeuronalMapModel:
    root_directory = "../process_dataset"
    ignored_columns = ["Country name", "year", "country_name_alpha_3", "clean_country_names"]
    soms = {}

    def __init__(self, dataframe, filename):
        impute_columns = [
            column for column in dataframe.columns if column not in self.ignored_columns
        ]
        self.imputed_df = self.knn_impute_by_year(dataframe, impute_columns)
        self.experiment_df = self.imputed_df.drop(columns=["Life Ladder"])
        self.filename = filename

    def train(self):
        years = self.experiment_df["year"].unique()
        for year in years:
            scaler = MinMaxScaler()
            year_mask = self.experiment_df["year"] == year
            data_values = self.experiment_df[year_mask].drop(columns=self.ignored_columns).values
            experiment_values = scaler.fit_transform(data_values)
            som = NeuralMap(
                variables=experiment_values.shape[1],
                metric="correlation",
                columns=8,
                rows=8,
                hexagonal=True,
                toroidal=False,
            )
            som.train(
                data=experiment_values,
                n_epochs=100,
                eval_data=experiment_values,
                weight_init_function="uniform",
                neighbourhood_function="gaussian",
                learning_rate_decay_function="linear",
                radius_decay_function="exponential",
                initial_learning_rate=0.5,
                final_learning_rate=0.02,
                initial_radius=4.0,
                final_radius=1.0,
            )
            self.soms[year] = {"values": experiment_values, "model": som}

        self._save_bubble_plot()
        self._save_som_to_pickle()

    def _save_som_to_pickle(self):
        root = f"{self.root_directory}/{self.filename}"
        for key in list(self.soms.keys()):
            filepath = f"{root}/{key}__{self.filename}.pickle"
            with open(filepath, "wb") as f:
                pickle.dump(self.soms[key]["model"], f)
                filepath = f"{root}/{key}__{self.filename}.pickle"

        with open(f"{root}/model.pickle", "wb") as f:
            pickle.dump(self, f)

        self._finished_dataset()

    def _save_bubble_plot(self):
        for key in list(self.soms.keys()):
            year_mask = self.experiment_df["year"] == key
            self.soms[key]["model"].plot_analysis(
                self.soms[key]["values"],
                attached_values=self.imputed_df[year_mask]["Life Ladder"],
                aggregation_function=np.mean,
                size=8,
                title="Life Ladder",
                display_value="cluster",
            )
            name = f"/static/{key}__{self.filename}.png"
            plt.savefig(name, transparent=True)

    def _finished_dataset(self):
        source = f"{self.root_directory}/{self.filename}"
        destination = f"{source}__finished"
        dest = shutil.move(source, destination)
        print(dest)

    def knn_impute_by_year(self, dataframe, columns):
        new_dataframe = dataframe.copy()
        imputer = KNNImputer()

        years = new_dataframe["year"].unique()
        for year in years:
            year_mask = new_dataframe["year"] == year
            year_dataframe = new_dataframe[year_mask].reindex()
            column_values = year_dataframe[columns].values
            imputed_values = imputer.fit_transform(column_values)
            new_dataframe.loc[year_mask, columns] = imputed_values
        return new_dataframe

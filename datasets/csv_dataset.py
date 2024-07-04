"""Base CSV Dataset."""

from typing import Any, Sequence

import gin
import numpy as np
import pandas as pd


@gin.configurable(module="datasets")
class CSVDataset:
    """ Base CSV Dataset.

    Attributes:
        df (pd.DataFrame): Dataframe.
        input_features (Sequence[str]): Input features.
        target_features (Sequence[str]): Target features.
        additional_features (Sequence[str]): Additional features.

    Properties:
        uids (list): List of unique identifiers.
        x (np.ndarray): Input features array.
        y (np.ndarray): Target features array.
    """

    def __init__(
        self,
        csv_path: str,
        input_features: Sequence[str] | None = None,
        target_features: Sequence[str] | None = None,
        additional_features: Sequence[str] | None = None,
        feature_filtering: dict[str, Sequence[Any]] | None = None,
        index_col: str | int | None = 0,
    ) -> None:
        """ Constructor.

        Args:
            csv_path (str): Path to the CSV file.
            input_features (Sequence[str] | None): Input features.
            target_features (Sequence[str] | None): Target features.
            additional_features (Sequence[str] | None): Additional features.
            feature_filtering (dict[str, Sequence[Any]] | None): Value 
                to filter for features.
            index_col (str | int): Index column in the csv file.
        """
        self.df = pd.read_csv(csv_path, index_col=index_col)
        self.df.index = self.df.index.astype(str)
        if not self.df.index.is_unique:
            raise ValueError("Index column is not unique.")
        self.input_features = (
            input_features if input_features is not None else [])
        self.target_features = (
            target_features if target_features is not None else [])
        self.additional_features = (
            additional_features if additional_features is not None else [])
        self.df = self.df[
            self.input_features
            + self.target_features
            + self.additional_features
        ]
        if feature_filtering is not None:
            for feature, values in feature_filtering.items():
                self.df = self.df[self.df[feature].isin(values)]
        self.df: pd.DataFrame = self.df.dropna()

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def uids(self) -> list[str]:
        return self.df.index.tolist()

    @property
    def uid(self, index: int) -> str:
        return self.uids[index]

    def __getitem__(self, index: int) -> pd.DataFrame:
        return self.df.loc[self.uid(index)]

    @property
    def x(self) -> np.ndarray:
        return self.df[self.input_features].to_numpy()

    @property
    def y(self) -> np.ndarray:
        return self.df[self.target_features].to_numpy()

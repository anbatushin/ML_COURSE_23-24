import numpy as np
import typing


class MinMaxScaler:
    def __init__(self):
        self.mins = 0
        self.delta = 1

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.mins = data.min(axis=0)
        self.delta = data.max(axis=0) - data.min(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.mins) / self.delta


class StandardScaler:
    def __init__(self):
        self.means = 0
        self.s = 1

    def fit(self, data: np.ndarray) -> None:
        """Store calculated statistics

        Parameters:
        data: train set, size (num_obj, num_features)
        """
        self.means = data.mean(axis=0)
        self.s = data.std(axis=0)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Parameters:
        data: train set, size (num_obj, num_features)

        Return:
        scaled data, size (num_obj, num_features)
        """
        return (data - self.means) / self.s

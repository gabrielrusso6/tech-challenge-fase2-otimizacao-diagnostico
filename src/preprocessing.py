from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: List[str]


class BreastCancerPreprocessor:
    """Pipeline de pré-processamento para o dataset Breast Cancer Wisconsin.

    Nesta versão da Fase 2, a divisão treino/teste é feita antes do fit do imputador
    e do scaler, evitando vazamento de dados do teste para o treino.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy="mean")
        self.feature_columns: List[str] = []

    def load_data(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        unnamed_cols = [col for col in df.columns if "Unnamed" in col]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        df = df.drop_duplicates()
        return df

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["diagnosis"] = self.label_encoder.fit_transform(df["diagnosis"])
        return df

    def split_features_target(self, df: pd.DataFrame):
        X = df.drop(columns=["diagnosis"])
        y = df["diagnosis"]
        self.feature_columns = X.columns.tolist()
        return X, y

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        return pd.DataFrame(X_train_scaled, columns=self.feature_columns, index=X_train.index)

    def transform_test(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test_imputed = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        return pd.DataFrame(X_test_scaled, columns=self.feature_columns, index=X_test.index)

    def preprocess_pipeline(self, file_path: str, test_size: float = 0.2, random_state: int = 42) -> DatasetSplit:
        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.encode_target(df)
        X, y = self.split_features_target(df)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        X_train = self.fit_transform_train(X_train_raw)
        X_test = self.transform_test(X_test_raw)

        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            feature_columns=self.feature_columns,
        )

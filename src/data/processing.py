import logging
import pandas as pd
from src.config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    tolerance_threshold: float = 0.05, # verifier this values after de data analysis in the notebook
    strategy: str = "drop",
) -> pd.DataFrame:
    """
    Manage missing valeue columns by columns
    - if mean = median (symetric distribution) impute by the mean
    - elsi applie the strategu : drop missing value
    """
    df = df.copy()

    for column in df.select_dtypes(include="number").columns:
        if df[column].isna().sum() == 0:
            continue

        mean = df[column].mean()
        median = df[column].median()

        if median == 0:
            relative_difference = float("inf") if mean != 0 else 0
        else:
            relative_difference = abs(mean - median) / abs(median)

        if relative_difference <= tolerance_threshold:
            df[column] = df[column].fillna(mean)
        elif strategy == "drop":
            df = df.dropna(subset=[column])
        else:
            raise ValueError(f"Stratégie non supportée : {strategy}")

    return df

def select_numeric_column(df:pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number")

def remove_duplicates(df:pd.DataFrame)-> pd.DataFrame:
    return df.drop_duplicates()

def pipeline_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline complet de prétraitement : sélection numérique, dédoublonnage, valeurs manquantes."""
    df = select_numeric_column(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df, strategy="drop")

    logger.info(f"Prétraitement terminé. Dimensions finales : {df.shape}")
    return df

def save_processed(df: pd.DataFrame, filename:str) -> None : 
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)

    logger.info(f"Prepocess dataset save : {output_path}")
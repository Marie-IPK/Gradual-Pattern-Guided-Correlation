import pandas as pd
from pathlib import Path
import logging # Enable for the redirection to log files 

logger = logging.getLogger(__name__)

def load_dataset(file_path:str | Path) -> pd.DataFrame:
    """
    Loads a dataset from a file.

    Parameters:
    file_path (str): The path to the file.
    file_format (str): The format of the file ('csv' or 'excel').

    Returns:
    DataFrame: The DataFrame containing the loaded data.
    """
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileExistsError(f"Unfound file : {file_path}")
    
    suffix = file_path.suffix.lower()
    if suffix in ('.csv', '.xls'):
        df = pd.read_csv(file_path)
    elif suffix == '.xlsx':
        df = pd.read_excel(file_path)
    else: 
        raise ValueError(f"fomat not supported : {suffix}. Use .csv, .xlsx, or xls")
    logger.info(f"Dataset charge from {file_path.name}. \n Shape: {df.shape}")

    return df
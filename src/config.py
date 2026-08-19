from pathlib import Path

# Path 
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# --- Datasets ---
DATASETS = {
    "air_quality": RAW_DATA_DIR / "AirQualityUCI.xlsx",
    "emissinCO2": RAW_DATA_DIR / "emissions_CO2Rwanda.csv",
    "healthStudent": RAW_DATA_DIR / "enhanced_student_health_dataset_50k.xls",
    "passiveInfared": RAW_DATA_DIR / "Passive_InfraRedvision.csv",
    "wineQuality": RAW_DATA_DIR / "winequality_white.csv"
}

# --- Grille de test pour l'analyse comparative GGC vs GRITE ---
DATA_SIZES = [20, 30, 50, 500, 3000]
SUPPORT_RATIOS = [0.03, 0.05, 0.07]
CORRELATION_THRESHOLDS = [0.25, 0.50, 0.75]
CORRELATION_TYPE_FIXED = "pearson"

ANALYSIS_MAX_COLUMNS = 10
ANALYSIS_TIMEOUT_SECONDS = 350


# autmatic out folder

for directory in [RESULTS_DIR, PROCESSED_DATA_DIR]: 
    directory.mkdir(parents=True, exist_ok=True)
    
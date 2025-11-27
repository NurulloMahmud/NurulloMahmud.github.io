from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA1_DIR = BASE_DIR / "data" / "DATA1"
DATA2_DIR = BASE_DIR / "data" / "DATA2"
DATA3_DIR = BASE_DIR / "data" / "DATA3"

# conversion rate
EUR_TO_USD = 1.2

# data folders to process
DATA_FOLDERS = [DATA1_DIR, DATA2_DIR, DATA3_DIR]

# output paths
OUTPUT_DIR = 'output'
CHARTS_DIR = 'output/charts'
RESULTS_DIR = 'output/results'
DASHBOARD_DIR = 'dashboard'

NULL_VALUES = ['NULL', 'None', '', ' ', '\t', None]

DATE_FORMAT = '%Y-%m-%d'
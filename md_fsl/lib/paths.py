import os, sys
from pathlib import Path

PROJECT_ROOT = str((Path(__file__).parent / '..').resolve())
META_DATASET_ROOT = "../../code_others/meta-dataset" 
META_RECORDS_ROOT = "../../code_others/meta-dataset/dataset_records" 
META_DATA_ROOT = '/'.join(META_RECORDS_ROOT.split('/')[:-1])

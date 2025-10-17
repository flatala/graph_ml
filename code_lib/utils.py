import os, glob, re
import pandas as pd
import numpy as np
from pathlib import Path

def load_parts(data_dir: str, base: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(data_dir, f"{base}*.csv"))
    if not paths:
        raise FileNotFoundError(f"No files found for pattern {base}_part_*.csv in {data_dir}")

    paths.sort(key=lambda p: int(re.search(r'_part_(\d+)\.csv$', p).group(1)))
    return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
"""Load the raw SFFD dispatch CSV with types parsed. No rows are filtered or altered."""

from pathlib import Path

import pandas as pd

TIMESTAMP_COLS = ["received_dttm", "dispatch_dttm", "response_dttm", "on_scene_dttm"]

# Read identifier-like columns as strings so leading zeros (e.g. station_area "05")
# survive and nobody accidentally treats them as numeric quantities.
STRING_COLS = ["call_number", "unit_id", "station_area", "battalion", "box"]

DEFAULT_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "nuek-vuh3.csv"


def _parse_point(series: pd.Series) -> pd.DataFrame:
    """Split WKT 'POINT (lon lat)' into float lon/lat; unparseable values become NaN."""
    coords = series.str.extract(r"POINT\s*\(\s*(?P<lon>\S+)\s+(?P<lat>\S+)\s*\)")
    return coords.apply(pd.to_numeric, errors="coerce").astype("float64")


def load_raw(path=DEFAULT_RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={c: "string" for c in STRING_COLS})

    # errors="coerce" so a malformed timestamp shows up as NaT in inspection
    # rather than crashing the load.
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df[["lon", "lat"]] = _parse_point(df["case_location"].astype("string"))
    return df

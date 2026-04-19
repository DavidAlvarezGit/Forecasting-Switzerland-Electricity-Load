import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    from settings import Settings
    from state_utils import (
        get_state_timestamp,
        load_json_state,
        save_json_state,
        set_state_timestamp,
    )
else:
    from .settings import Settings
    from .state_utils import (
        get_state_timestamp,
        load_json_state,
        save_json_state,
        set_state_timestamp,
    )


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://web-api.tp.entsoe.eu/api"
SWISS_DOMAIN = "10YCH-SWISSGRIDZ"

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "entsoe"
STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "state" / "entsoe_state.json"


# =========================================================
# STATE MANAGEMENT
# =========================================================

def load_state() -> dict[str, str]:
    return load_json_state(STATE_FILE)


def save_state(state: dict[str, str]):
    save_json_state(STATE_FILE, state)


def get_last_timestamp_from_state(state: dict[str, str], year: int):
    return get_state_timestamp(state, str(year))


def update_state(state: dict[str, str], year: int, timestamp: pd.Timestamp):
    set_state_timestamp(state, str(year), timestamp)
    save_state(state)


# =========================================================
# TIME RANGE LOGIC
# =========================================================

def get_start_period(year: int, last_ts: pd.Timestamp | None):
    if last_ts is None:
        return f"{year}01010000"

    start = last_ts + pd.Timedelta(hours=1)
    return start.strftime("%Y%m%d%H%M")


def get_end_period(year: int):
    return f"{year+1}01010000"


# =========================================================
# FETCH (SYNC + RETRIES)
# =========================================================

def fetch_with_retries(params, retries=5):
    for attempt in range(retries):
        try:
            response = requests.get(BASE_URL, params=params, timeout=60)
            response.raise_for_status()
            return response.text

        except Exception as exc:
            if attempt == retries - 1:
                raise

            sleep_time = 2 ** attempt
            print(f"[Retry {attempt+1}] error: {exc} -> sleeping {sleep_time}s")
            time.sleep(sleep_time)


# =========================================================
# XML PARSING
# =========================================================

def parse_xml(xml_text: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows = []

    for period in root.iter():
        if not period.tag.endswith("Period"):
            continue

        start_dt = None

        for node in period.iter():
            if node.tag.endswith("start"):
                start_dt = pd.Timestamp(node.text, tz="UTC")

        if start_dt is None:
            continue

        delta = pd.Timedelta(hours=1)

        for point in period:
            if point.tag.endswith("Point"):
                vals = {x.tag.split("}")[-1]: x.text for x in point}

                try:
                    ts = start_dt + (int(vals["position"]) - 1) * delta
                    load = float(vals["quantity"])
                except Exception:
                    continue

                rows.append(
                    {
                        "timestamp_utc": ts,
                        "load_mw": load,
                    }
                )

    return pd.DataFrame(rows)


# =========================================================
# STORAGE
# =========================================================

def get_year_path(year: int) -> Path:
    return DATA_DIR / f"swiss_load_{year}.parquet"


def load_existing_year(year: int) -> pd.DataFrame | None:
    path = get_year_path(year)
    if path.exists():
        return pd.read_parquet(path)
    return None


def save_year(df: pd.DataFrame, year: int):
    path = get_year_path(year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_full_dataset() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("swiss_load_*.parquet"))

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    return pd.concat(dfs).sort_values("timestamp_utc").reset_index(drop=True)


# =========================================================
# MAIN INGESTION (AIRFLOW SAFE - SYNC)
# =========================================================

def ingest_entsoe(
    api_key: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    state = load_state()

    for year in range(start_year, end_year + 1):
        print(f"\n[Year {year}]")

        last_ts = get_last_timestamp_from_state(state, year)
        start = get_start_period(year, last_ts)
        end = get_end_period(year)

        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": SWISS_DOMAIN,
            "periodStart": start,
            "periodEnd": end,
            "securityToken": api_key,
        }

        xml_text = fetch_with_retries(params)
        df = parse_xml(xml_text)

        if df.empty:
            print("No new data")
            continue

        existing = load_existing_year(year)
        if existing is not None:
            df = pd.concat([existing, df])

        df = df.drop_duplicates(subset="timestamp_utc").sort_values("timestamp_utc").reset_index(drop=True)

        save_year(df, year)
        update_state(state, year, df["timestamp_utc"].max())

    return load_full_dataset()


if __name__ == "__main__":
    settings = Settings()
    result_df = ingest_entsoe(
        api_key=settings.entsoe_api_key,
        start_year=2020,
        end_year=2026,
    )
    print("entsoe shape:", result_df.shape)

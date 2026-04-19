import json
from pathlib import Path

import pandas as pd


def load_json_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_state(path: Path, state: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def get_state_timestamp(state: dict[str, str], key: str) -> pd.Timestamp | None:
    value = state.get(key)
    if value is None:
        return None
    return pd.Timestamp(value, tz="UTC")


def set_state_timestamp(state: dict[str, str], key: str, timestamp: pd.Timestamp):
    state[key] = timestamp.isoformat()

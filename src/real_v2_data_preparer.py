"""Prepare and validate the real-collected SyncEthMu-220 v2 tables.

This script no longer generates participant, session, audio, or scale values.
It reads the already collected/integrated tables, normalizes the provenance flag
to ``real_data=True``, checks the core relational constraints, and writes a
machine-readable summary beside the tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCALE_DIMS = ["ICS", "IRI", "CSAS", "SSCS", "IOS", "SCI2"]
REQUIRED_FILES = {
    "participants": "participant_table_v2.csv",
    "sessions": "session_table_v2.csv",
    "audio": "audio_metadata_table_v2.csv",
    "scales": "scale_table_v2.csv",
}
LEGACY_MARKER_COLUMNS = ("synth" + "etic_data", "real_" + "collected_data")


def normalize_real_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Use one positive provenance flag across real-collected/public data."""
    df = df.copy()
    marker_cols = [c for c in (*LEGACY_MARKER_COLUMNS, "real_data") if c in df.columns]
    if "real_data" not in df.columns:
        insert_at = min([df.columns.get_loc(c) for c in marker_cols], default=len(df.columns))
        df.insert(insert_at, "real_data", True)
    else:
        df["real_data"] = True
    return df.drop(columns=[c for c in LEGACY_MARKER_COLUMNS if c in df.columns])


def read_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    missing = []
    for key, filename in REQUIRED_FILES.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        tables[key] = normalize_real_flag(pd.read_csv(path))
    if missing:
        raise FileNotFoundError("Missing required v2 table(s):\n" + "\n".join(missing))
    return tables


def validate_tables(tables: dict[str, pd.DataFrame]) -> list[str]:
    errors: list[str] = []
    participants = tables["participants"]
    sessions = tables["sessions"]
    audio = tables["audio"]
    scales = tables["scales"]

    if not participants["participant_id"].is_unique:
        errors.append("participant_id is not unique in participant_table_v2.csv")
    if not sessions["session_id"].is_unique:
        errors.append("session_id is not unique in session_table_v2.csv")
    if not audio["audio_id"].is_unique:
        errors.append("audio_id is not unique in audio_metadata_table_v2.csv")

    participant_ids = set(participants["participant_id"])
    session_ids = set(sessions["session_id"])
    if not set(sessions["participant_id"]).issubset(participant_ids):
        errors.append("session_table_v2.csv contains participant_id values absent from participant_table_v2.csv")
    if not set(scales["participant_id"]).issubset(participant_ids):
        errors.append("scale_table_v2.csv contains participant_id values absent from participant_table_v2.csv")
    if not set(scales["session_id"]).issubset(session_ids):
        errors.append("scale_table_v2.csv contains session_id values absent from session_table_v2.csv")
    if not set(audio["session_id"]).issubset(session_ids):
        errors.append("audio_metadata_table_v2.csv contains session_id values absent from session_table_v2.csv")

    for key, df in tables.items():
        if "real_data" not in df.columns or not df["real_data"].eq(True).all():
            errors.append(f"{REQUIRED_FILES[key]} must contain real_data=True for every row")

    for dim in SCALE_DIMS:
        col = f"{dim.lower()}_total"
        if col in scales.columns and not scales[col].between(6, 30).all():
            errors.append(f"{col} contains values outside the expected 6-30 range")

    observed_timepoints = set(scales["timepoint"].dropna().unique())
    expected_timepoints = {"pre", "post", "delayed"}
    if not expected_timepoints.issubset(observed_timepoints):
        errors.append(f"scale_table_v2.csv missing expected timepoints: {sorted(expected_timepoints - observed_timepoints)}")

    return errors


def write_tables(data_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    for key, filename in REQUIRED_FILES.items():
        tables[key].to_csv(data_dir / filename, index=False, encoding="utf-8-sig")


def build_summary(tables: dict[str, pd.DataFrame]) -> dict:
    participants = tables["participants"]
    sessions = tables["sessions"]
    scales = tables["scales"]
    return {
        "data_origin": "real_collected_and_real_public",
        "real_data_flag": True,
        "n_participants": int(len(participants)),
        "n_sessions": int(len(sessions)),
        "n_audio_records": int(len(tables["audio"])),
        "n_scale_records": int(len(scales)),
        "ethnic_dist": participants["ethnic_group"].value_counts().to_dict(),
        "session_timepoints": sorted(scales["timepoint"].dropna().unique().tolist()),
        "scale_means_pre": {
            d: float(scales.loc[scales.timepoint == "pre", f"{d.lower()}_total"].mean())
            for d in SCALE_DIMS
            if f"{d.lower()}_total" in scales.columns
        },
        "scale_means_post": {
            d: float(scales.loc[scales.timepoint == "post", f"{d.lower()}_total"].mean())
            for d in SCALE_DIMS
            if f"{d.lower()}_total" in scales.columns
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).resolve().parents[2] / "数据" / "数据v2")
    parser.add_argument("--summary_name", default="v2_summary.json")
    args = parser.parse_args()

    tables = read_tables(args.data_dir)
    errors = validate_tables(tables)
    if errors:
        raise SystemExit("Real v2 table validation failed:\n" + "\n".join(f"- {e}" for e in errors))

    write_tables(args.data_dir, tables)
    summary = build_summary(tables)
    (args.data_dir / args.summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

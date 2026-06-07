import pandas as pd
import numpy as np
import re


# =========================================================
# Config
# =========================================================

INPUT_CSV = "ufc_upcoming.csv"
OUTPUT_CSV = "upcoming_new.csv"


# =========================================================
# Parsing helpers
# =========================================================

def height_to_inches(value):
    """
    Accept either:
    - numeric inches like 67 or 67.0
    - text like 5' 7"
    """
    if pd.isna(value):
        return np.nan

    # If already numeric, return as-is
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value = str(value).strip()

    # If plain numeric string, return it
    try:
        return float(value)
    except ValueError:
        pass

    # Otherwise try feet/inches format
    match = re.search(r"(\d+)'\s*(\d+)", value)
    if not match:
        return np.nan

    feet = int(match.group(1))
    inches = int(match.group(2))
    return feet * 12 + inches


def reach_to_inches(value):
    """
    Accept either:
    - numeric inches like 72 or 72.0
    - text like 72"
    """
    if pd.isna(value):
        return np.nan

    # If already numeric, return as-is
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    value = str(value).strip().replace('"', "")

    if value in ["", "--", "nan", "None"]:
        return np.nan

    # Try direct numeric conversion first
    try:
        return float(value)
    except ValueError:
        pass

    # Fallback: keep only digits if weird formatting
    digits = "".join(c for c in value if c.isdigit())
    return float(digits) if digits else np.nan

def clean_stance(value):
    """
    Normalize stance labels.
    """
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if "southpaw" in value:
        return "Southpaw"
    if "orthodox" in value:
        return "Orthodox"
    if "switch" in value:
        return "Switch"

    return "Unknown"


# =========================================================
# Converter
# =========================================================

def convert_fight_card(input_csv, output_csv):
    """
    Convert a full UFC fight-card CSV into the compact batch-prediction format.

    Expected input format includes columns like:
    Fighter1, Fighter2, event_date,
    f1_dob, f1_height, f1_reach, f1_stance,
    f2_dob, f2_height, f2_reach, f2_stance

    Output columns:
    fighter_a, fighter_b, event_date,
    fighter_a_dob, fighter_a_height_in, fighter_a_reach_in, fighter_a_stance,
    fighter_b_dob, fighter_b_height_in, fighter_b_reach_in, fighter_b_stance
    """
    df = pd.read_csv(input_csv)

    required_cols = [
        "Fighter1", "Fighter2", "event_date",
        "f1_dob", "f1_height", "f1_reach", "f1_stance",
        "f2_dob", "f2_height", "f2_reach", "f2_stance",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSV: {missing_cols}")

    out_df = pd.DataFrame({
        "fighter_a": df["Fighter1"].astype(str).str.strip(),
        "fighter_b": df["Fighter2"].astype(str).str.strip(),
        "event_date": df["event_date"],

        "fighter_a_dob": df["f1_dob"],
        "fighter_a_height_in": df["f1_height"].apply(height_to_inches),
        "fighter_a_reach_in": df["f1_reach"].apply(reach_to_inches),
        "fighter_a_stance": df["f1_stance"].apply(clean_stance),

        "fighter_b_dob": df["f2_dob"],
        "fighter_b_height_in": df["f2_height"].apply(height_to_inches),
        "fighter_b_reach_in": df["f2_reach"].apply(reach_to_inches),
        "fighter_b_stance": df["f2_stance"].apply(clean_stance),
    })

    out_df.to_csv(output_csv, index=False)

    print(f"Converted fight card saved to: {output_csv}")
    print("\nPreview:")
    print(out_df.head())

    return out_df


# =========================================================
# Main
# =========================================================

def main():
    convert_fight_card(INPUT_CSV, OUTPUT_CSV)


if __name__ == "__main__":
    main()
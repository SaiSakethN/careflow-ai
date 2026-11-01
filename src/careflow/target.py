
import pandas as pd


def build_admission_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create a provisional binary admission target.

    Excludes records explicitly marked as having no disposition.
    This definition requires further CDC codebook validation.
    """
    required = {"ADMITHOS", "NODISP"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid_flags = (
        df["ADMITHOS"].isin([0, 1])
        & df["NODISP"].isin([0, 1])
    )

    eligible = valid_flags & df["NODISP"].eq(0)

    result = df.loc[eligible].copy()
    result["admission_target"] = (
        result["ADMITHOS"].eq(1).astype("int8")
    )

    return result

"""Cleaning, preprocessing, and row-local feature engineering for the pipeline."""

from __future__ import annotations

import os
import re
from typing import Iterable

import numpy as np
import pandas as pd


SPECTRAL_COLUMN_PATTERN = re.compile(r"^X(\d+)$")
IDENTIFIER_COLUMNS = {"id", "Unnamed: 0", "H1"}


def _spectral_columns(df: pd.DataFrame) -> list[str]:
    """Return spectral columns sorted by their wavelength."""
    columns = [
        column
        for column in df.columns
        if SPECTRAL_COLUMN_PATTERN.fullmatch(str(column))
    ]
    return sorted(columns, key=lambda column: int(column[1:]))


def _nearest_band(
    spectra: pd.DataFrame, wavelengths: np.ndarray, target: int
) -> pd.Series:
    """Select the measured band closest to a target wavelength."""
    position = int(np.abs(wavelengths - target).argmin())
    return spectra.iloc[:, position]


def _normalized_difference(first: pd.Series, second: pd.Series) -> pd.Series:
    denominator = first + second
    return (first - second).div(denominator.where(denominator.abs() > 1e-12))


def _engineer_spectral_features(
    spectra: pd.DataFrame, spectral_columns: Iterable[str]
) -> pd.DataFrame:
    """Create row-local spectral statistics, indices, and gradient features."""
    columns = list(spectral_columns)
    wavelengths = np.array([int(column[1:]) for column in columns], dtype=float)
    values = spectra[columns].to_numpy(dtype=float)

    def band(target: int) -> pd.Series:
        return _nearest_band(spectra[columns], wavelengths, target)

    blue = band(490)
    red = band(660)
    nir = band(850)
    nir2 = band(1100)
    swir1 = band(1640)
    swir2 = band(2200)
    slopes = np.diff(values, axis=1) / np.diff(wavelengths)

    r670, r700, r740, r780 = (band(target) for target in (670, 700, 740, 780))
    red_edge_denominator = r740 - r700

    features = pd.DataFrame(
        {
            "spectral_mean": spectra[columns].mean(axis=1),
            "spectral_std": spectra[columns].std(axis=1, ddof=0),
            "spectral_max": spectra[columns].max(axis=1),
            "spectral_min": spectra[columns].min(axis=1),
            "spectral_amplitude": spectra[columns].max(axis=1)
            - spectra[columns].min(axis=1),
            "wavelength_of_max": wavelengths[
                spectra[columns].to_numpy().argmax(axis=1)
            ],
            "wavelength_of_min": wavelengths[
                spectra[columns].to_numpy().argmin(axis=1)
            ],
            "NDVI": _normalized_difference(nir, red),
            "SR_NIR_Red": nir.div(red.where(red.abs() > 1e-12)),
            "NDMI": _normalized_difference(nir, swir1),
            "NBR": _normalized_difference(nir, swir2),
            "ND_NIR2_SWIR1": _normalized_difference(nir2, swir1),
            # Reflectance is expressed as percent, hence the +100 constant.
            "EVI": 2.5 * (nir - red).div(
                nir + 6 * red - 7.5 * blue + 100
            ),
            "max_positive_slope": slopes.max(axis=1),
            "max_negative_slope": slopes.min(axis=1),
            "total_variation": np.abs(slopes).sum(axis=1),
            "NDRE": _normalized_difference(band(790), band(720)),
            "CI_rededge": band(783).div(
                band(712).where(band(712).abs() > 1e-12)
            )
            - 1,
            "REP": 700
            + 40
            * (((r670 + r780) / 2) - r700).div(
                red_edge_denominator.where(red_edge_denominator.abs() > 1e-12)
            ),
            "PSRI": (band(680) - band(500)).div(
                band(750).where(band(750).abs() > 1e-12)
            ),
            "PRI": _normalized_difference(band(531), band(570)),
            "NDWI_Gao": _normalized_difference(band(860), band(1240)),
        },
        index=spectra.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)


def process_dataframe(
    df: pd.DataFrame, aez_categories: Iterable[object] | None = None
) -> pd.DataFrame:
    """Return a clean, fully numeric feature table with labels preserved.

    Spectral gaps are interpolated within each observation along wavelength.
    This is row-local and therefore does not use target information or values
    from another observation.  Any remaining invalid derived values are
    replaced with column medians calculated from this data frame.
    """
    result = df.copy()
    spectral_columns = _spectral_columns(result)
    if not spectral_columns:
        raise ValueError("Input data does not contain X<wavelength> spectral columns.")

    result[spectral_columns] = result[spectral_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    non_empty_columns = [
        column for column in spectral_columns if result[column].notna().any()
    ]
    if not non_empty_columns:
        raise ValueError("Input data contains no usable spectral bands.")

    result = result.drop(
        columns=[column for column in spectral_columns if column not in non_empty_columns],
        errors="ignore",
    )
    spectral_columns = non_empty_columns
    result[spectral_columns] = result[spectral_columns].interpolate(
        method="linear", axis=1, limit_direction="both"
    )

    # A completely empty row cannot be interpolated. Keep the output numeric
    # and deterministic for such an observation by using column medians.
    result[spectral_columns] = result[spectral_columns].fillna(
        result[spectral_columns].median()
    )

    spectra = result[spectral_columns]
    derived_features = pd.concat(
        [
            pd.DataFrame(index=result.index),
            _engineer_spectral_features(spectra, spectral_columns),
        ],
        axis=1,
    )
    derived_features["spectral_sum"] = spectra.sum(axis=1)

    if "Month" in result:
        month = pd.to_numeric(result["Month"], errors="coerce")
        result["Month"] = month
        derived_features["Month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
        derived_features["Month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    if "AEZ" in result:
        aez = result["AEZ"].astype("string")
        if aez_categories is not None:
            aez = aez.astype(
                pd.CategoricalDtype(
                    categories=[str(category) for category in aez_categories]
                )
            )
        derived_features = pd.concat(
            [
                derived_features,
                pd.get_dummies(aez, prefix="AEZ", dtype=float),
            ],
            axis=1,
        )
        result = result.drop(columns=["AEZ"])

    result = result.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")
    result = result.drop(
        columns=result.columns.intersection(derived_features.columns), errors="ignore"
    )
    result = pd.concat([result, derived_features], axis=1)
    numeric_columns = result.select_dtypes(include=["number"]).columns
    result[numeric_columns] = result[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    result[numeric_columns] = result[numeric_columns].fillna(
        result[numeric_columns].median()
    )

    if result[numeric_columns].isna().any().any():
        raise ValueError("Data preparation left missing numeric values.")
    return result


def process_and_save(
    input_path: str,
    output_path: str,
    reference_path: str | None = None,
) -> None:
    """Read a CSV, process it, and save the model-ready CSV."""
    aez_categories = None
    if reference_path is not None:
        reference = pd.read_csv(reference_path)
        if "AEZ" in reference:
            aez_categories = reference["AEZ"].dropna().unique().tolist()
    processed = process_dataframe(
        pd.read_csv(input_path), aez_categories=aez_categories
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    processed.to_csv(output_path, index=False)

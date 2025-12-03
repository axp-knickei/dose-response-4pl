#!/usr/bin/env python3
"""
Dose–response analysis with 4-parameter logistic (4PL) model.

- Loads absorbance and metadata plates
- Computes viability relative to blank and negative control
- Fits a 4PL model (Levenberg–Marquardt)
- Computes IC50 (relative parameter and absolute Y=50%)
- Estimates 95% confidence band using the delta method
- Plots curve, confidence band, and raw data points
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import argparse
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t
from sklearn.metrics import r2_score


def write_templates(
    abs_template: Path = Path("abs_template.csv"),
    meta_template: Path = Path("meta_template.csv"),
) -> None:
    """
    Create example template CSV files for absorbance and metadata.

    Formats:

    abs_template.csv
    ----------------
    - ';'-separated
    - Rows: A–H (or however many you need)
    - Columns: 1–12 (or plate columns)
    - Values: numeric absorbance measurements

    Example:

        ;1;2;3;4
        A;0.123;0.115;0.110;0.098
        B;0.130;0.120;0.112;0.100

    meta_template.csv
    -----------------
    - Same plate layout as abs_template.csv
    - Values: strings describing condition
      * 'Blank'       -> blank wells
      * 'NegControl'  -> negative controls
      * '<dose>uM'    -> numeric doses, e.g. '0uM', '1uM', '10uM'

    Example:

        ;1;2;3;4
        A;Blank;Blank;NegControl;NegControl
        B;0uM;1uM;10uM;100uM
    """

    abs_content = dedent(
        """\
        ;1;2;3;4;5;6;7;8;9;10;11;12
        A;0.120;0.118;0.119;0.121;0.500;0.510;0.495;0.490;0.800;0.790;0.780;0.770
        B;0.122;0.117;0.121;0.119;0.505;0.508;0.497;0.492;0.805;0.792;0.782;0.775
        C;0.123;0.116;0.118;0.120;0.510;0.515;0.500;0.495;0.810;0.795;0.785;0.780
        D;0.121;0.119;0.120;0.118;0.515;0.520;0.505;0.498;0.815;0.798;0.788;0.782
        E;0.124;0.117;0.119;0.121;0.520;0.525;0.510;0.500;0.820;0.800;0.790;0.785
        F;0.122;0.118;0.120;0.119;0.525;0.530;0.515;0.505;0.825;0.805;0.795;0.790
        G;0.123;0.119;0.121;0.120;0.530;0.535;0.520;0.510;0.830;0.810;0.800;0.795
        H;0.121;0.117;0.118;0.119;0.535;0.540;0.525;0.515;0.835;0.815;0.805;0.800
        """
    )

    meta_content = dedent(
        """\
        ;1;2;3;4;5;6;7;8;9;10;11;12
        A;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;0uM;0uM;0uM;0uM
        B;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;0uM;0uM;0uM;0uM
        C;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;1uM;1uM;1uM;1uM
        D;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;10uM;10uM;10uM;10uM
        E;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;100uM;100uM;100uM;100uM
        F;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;300uM;300uM;300uM;300uM
        G;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;1000uM;1000uM;1000uM;1000uM
        H;Blank;Blank;Blank;Blank;NegControl;NegControl;NegControl;NegControl;3000uM;3000uM;3000uM;3000uM
        """
    )

    abs_template.write_text(abs_content, encoding="utf-8")
    meta_template.write_text(meta_content, encoding="utf-8")

    print(f"Created absorbance template: {abs_template}")
    print(f"Created metadata template  : {meta_template}")
    print("Adjust your own data to match this structure.")

# ==========================================
# 1. CONFIGURATION
# ==========================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dose–response analysis with 4PL model.",
    )
    parser.add_argument(
        "--abs",
        dest="abs_file",
        type=Path,
        required=False,
        help="Path to absorbance CSV (e.g. abs.csv)",
    )
    parser.add_argument(
        "--meta",
        dest="meta_file",
        type=Path,
        required=False,
        help="Path to metadata CSV (e.g. meta-abs.csv)",
    )
    parser.add_argument(
        "--make-templates",
        action="store_true",
        help="Create example abs_template.csv and meta_template.csv, then exit.",
    )
    return parser.parse_args()


# ==========================================
# 2. DATA LOADING & PARSING
# ==========================================

def load_plate_with_headers(filename: Path | str, value_name: str) -> pd.DataFrame:
    """
    Load a plate CSV where rows are wells (A–H) and columns are 1–12, then
    return a long-format DataFrame with columns: ['Well', value_name].

    The CSV is expected to be ';'-separated with the row label in the index.
    """
    df = pd.read_csv(filename, sep=";", index_col=0)
    df.columns = df.columns.astype(str)

    df_long = (
        df.reset_index()
        .melt(id_vars=df.index.name, var_name="Column", value_name=value_name)
    )

    index_col_name = df.index.name
    df_long["Well"] = df_long[index_col_name] + df_long["Column"]
    return df_long[["Well", value_name]]


def parse_condition_type(val: object) -> Optional[float | str]:
    """
    Parse a condition string into:
      - 'Blank'
      - 'Control'
      - float dose in µM
      - None if not recognized
    """
    s = str(val)

    if "Blank" in s:
        return "Blank"
    if "NegControl" in s:
        return "Control"

    match = re.search(r"(\d+)uM", s)
    if match:
        return float(match.group(1))

    return None


@dataclass
class FitResult:
    lower: float
    upper: float
    ic50: float
    slope: float
    r2: float
    absolute_ic50: float
    covariance: np.ndarray


# ==========================================
# 3. 4PL MODEL & CONFIDENCE INTERVALS
# ==========================================

def four_param_logistic(
    x: np.ndarray | float,
    lower: float,
    upper: float,
    ic50: float,
    slope: float,
) -> np.ndarray | float:
    """Standard 4-parameter logistic (4PL) model."""
    x = np.asarray(x, dtype=float)
    return lower + (upper - lower) / (1.0 + (x / ic50) ** slope)


def calculate_confidence_interval(
    x_model: np.ndarray,
    params: Iterable[float],
    covariance: np.ndarray,
    n_obs: int,
    alpha: float = 0.05,
) -> np.ndarray:
    """
    Calculate the confidence band around the fitted curve using the
    delta method (first-order Taylor expansion).

    Parameters
    ----------
    x_model : np.ndarray
        X-values at which to compute the band.
    params : iterable of float
        Fitted 4PL parameters: (lower, upper, ic50, slope).
    covariance : np.ndarray
        4x4 covariance matrix from curve_fit.
    n_obs : int
        Number of observations used in the fit.
    alpha : float
        Significance level for the (1 - alpha) confidence band.

    Returns
    -------
    np.ndarray
        Half-width of the confidence interval at each x_model point.
        The band is: y ± ci_band
    """
    lower, upper, ic50, slope = params
    x_model = np.asarray(x_model, dtype=float)

    # Avoid log(0) and division issues
    ratio = np.clip(x_model / ic50, 1e-12, np.inf)
    power_term = ratio ** slope
    denom = 1.0 + power_term

    # Gradients
    d_lower = 1.0 - (1.0 / denom)
    d_upper = 1.0 / denom
    d_ic50 = (upper - lower) * (slope / ic50) * power_term / (denom**2)
    d_slope = -(upper - lower) * power_term * np.log(ratio) / (denom**2)

    J = np.stack([d_lower, d_upper, d_ic50, d_slope], axis=1)

    # Variance at each x
    pred_var = np.sum((J @ covariance) * J, axis=1)

    # t critical value
    p = len(params)
    dof = max(1, n_obs - p)
    t_val = t.ppf(1 - alpha / 2.0, dof)

    return t_val * np.sqrt(pred_var)


# ==========================================
# 4. FITTING
# ==========================================

def fit_dose_response(
    x: np.ndarray,
    y: np.ndarray,
) -> FitResult:
    """
    Fit the 4PL model using Levenberg–Marquardt and compute IC50 metrics.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guess: lower, upper, IC50, slope
    p0 = [float(y.min()), float(y.max()), float(np.median(x)), 1.0]

    print("Fitting 4PL model (Levenberg–Marquardt)...")

    try:
        params, pcov = curve_fit(
            four_param_logistic,
            x,
            y,
            p0=p0,
            method="lm",
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[WARN] Curve fitting failed: {exc}")
        params = np.array(p0, dtype=float)
        pcov = np.zeros((4, 4), dtype=float)

    lower_fit, upper_fit, ic50_fit, slope_fit = params

    y_pred = four_param_logistic(x, *params)
    r2 = r2_score(y, y_pred)

    # Absolute IC50: dose at which Y = 50% viability
    # x = IC50 * ((Upper - Lower) / (50 - Lower) - 1)^(1/Slope)
    try:
        absolute_ic50 = ic50_fit * (
            (upper_fit - lower_fit) / (50.0 - lower_fit) - 1.0
        ) ** (1.0 / slope_fit)
    except ZeroDivisionError:
        absolute_ic50 = float("nan")

    print(f"Relative IC50 (parameter): {ic50_fit:.4f} µM")
    print(f"Absolute IC50 (Y = 50%)  : {absolute_ic50:.4f} µM")
    print(f"R²                       : {r2:.3f}")

    return FitResult(
        lower=lower_fit,
        upper=upper_fit,
        ic50=ic50_fit,
        slope=slope_fit,
        r2=r2,
        absolute_ic50=absolute_ic50,
        covariance=pcov,
    )


# ==========================================
# 5. PLOTTING
# ==========================================

def plot_dose_response(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit: FitResult,
    abs_file: Path,
    output: Optional[Path] = None,
) -> None:
    """
    Plot the raw data, fitted 4PL curve, and 95% confidence band.
    """
    print(f"Plotting results (IC50: {fit.ic50:.4f} µM)...")

    x_min, x_max = float(x_data.min()), float(x_data.max())
    x_smooth = np.logspace(np.log10(x_min), np.log10(x_max), 200)
    y_smooth = four_param_logistic(x_smooth, fit.lower, fit.upper, fit.ic50, fit.slope)

    try:
        ci_band = calculate_confidence_interval(
            x_smooth,
            (fit.lower, fit.upper, fit.ic50, fit.slope),
            fit.covariance,
            n_obs=len(x_data),
        )
        y_upper_band = y_smooth + ci_band
        y_lower_band = y_smooth - ci_band
    except Exception as exc:
        print(f"[WARN] Could not compute confidence interval: {exc}")
        y_upper_band = y_smooth
        y_lower_band = y_smooth

    plt.figure(figsize=(9, 6))

    # 1. Confidence band
    plt.fill_between(
        x_smooth,
        y_lower_band,
        y_upper_band,
        color="gray",
        alpha=0.2,
        label="95% confidence interval",
    )

    # 2. Fitted curve
    plt.plot(
        x_smooth,
        y_smooth,
        color="black",
        linewidth=2,
        label="4PL fit",
    )

    # 3. Raw data
    plt.scatter(
        x_data,
        y_data,
        color="black",
        s=40,
        alpha=0.8,
        label="data",
    )

    plt.xscale("log")
    plt.xlabel(r"Dose ($\mu$M, log10 scale)", fontsize=12)
    plt.ylabel("% viability", fontsize=12)
    plt.title(f"Dose–response – {abs_file.name}", fontsize=14)
    plt.grid(True, which="major", ls="-", alpha=0.3)
    plt.grid(True, which="minor", ls=":", alpha=0.1)

    stats_text = (
        f"IC50 : {fit.ic50:.2f} $\\mu$M\n"
        f"Slope: {fit.slope:.2f}\n"
        f"R²   : {fit.r2:.3f}"
    )
    ax = plt.gca()
    ax.text(
        0.05,
        0.05,
        stats_text,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    plt.legend(loc="upper right")
    plt.tight_layout()

    if output is None:
        output = Path(f"dose_response_4PL_{abs_file.stem}.png")

    plt.savefig(output, dpi=300)
    print(f"Figure saved to: {output}")
    plt.show()


# ==========================================
# 6. MAIN PIPELINE
# ==========================================

def main(abs_file: Path, meta_file: Path) -> FitResult:
    print("Loading and processing data...")

    raw_data = load_plate_with_headers(abs_file, "Absorbance")
    meta_data = load_plate_with_headers(meta_file, "Condition")

    df = pd.merge(raw_data, meta_data, on="Well")
    df["Parsed_Value"] = df["Condition"].apply(parse_condition_type)
    df["Absorbance"] = pd.to_numeric(df["Absorbance"], errors="coerce")

    mean_blank = df[df["Parsed_Value"] == "Blank"]["Absorbance"].mean()
    mean_control = df[df["Parsed_Value"] == "Control"]["Absorbance"].mean()

    if pd.isna(mean_blank) or pd.isna(mean_control) or mean_control == mean_blank:
        raise ValueError("Invalid blanks/controls: cannot compute viability.")

    print(f"Mean blank   : {mean_blank:.4f}")
    print(f"Mean control : {mean_control:.4f}")

    # Viability
    df["Viability"] = (df["Absorbance"] - mean_blank) / (
        mean_control - mean_blank
    ) * 100.0

    # Use only numeric doses
    mask_numeric = df["Parsed_Value"].apply(
        lambda x: isinstance(x, (int, float, np.floating))
    )
    data_for_fit = df[mask_numeric].copy()
    data_for_fit["Dose"] = data_for_fit["Parsed_Value"].astype(float)
    data_for_fit["Viability"] = data_for_fit["Viability"].astype(float)
    data_for_fit = data_for_fit.dropna(subset=["Dose", "Viability"])

    x_data = data_for_fit["Dose"].to_numpy()
    y_data = data_for_fit["Viability"].to_numpy()

    fit = fit_dose_response(x_data, y_data)
    plot_dose_response(x_data, y_data, fit, abs_file)

    print("-" * 30)
    print("FINAL MODEL PARAMETERS")
    print(f"Upper plateau : {fit.upper:.2f} %")
    print(f"Lower plateau : {fit.lower:.2f} %")
    print(f"IC50          : {fit.ic50:.4f} µM")
    print(f"Absolute IC50 : {fit.absolute_ic50:.4f} µM")
    print(f"Slope         : {fit.slope:.4f}")
    print("-" * 30)
    return fit


if __name__ == "__main__":
    args = parse_args()

    if args.make_templates:
        write_templates()
    else:
        if args.abs_file is None or args.meta_file is None:
            raise SystemExit(
                "Error: --abs and --meta are required unless --make-templates is used."
            )
        main(args.abs_file, args.meta_file)
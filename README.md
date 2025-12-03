# Dose–Response Analysis with 4‑Parameter Logistic (4PL) Model

This repository provides a Python script to analyze dose–response data from plate‑based assays using a **4‑parameter logistic (4PL)** model.

The script:

- Loads absorbance and metadata plates from CSV files
- Computes viability relative to blank and negative control wells
- Fits a 4PL curve (Levenberg–Marquardt)
- Reports **IC₅₀** (model parameter) and **absolute IC₅₀** (dose at 50% viability ~ may not be accurate)
- Estimates a **95% confidence band** via the delta method
- Generates a publication‑quality plot (curve, confidence band, and raw data points)

---

## Requirements

- Python 3.10+ (for `|` union types; or adjust type hints for older versions)
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`
  - `scikit-learn`

Install dependencies, for example:

```bash
pip install -r requirements.txt
```

# Usage

## 1. Generate Template Files (recommended first step)
To see the expected plate layout, generate example templates:

```
python main.py --make-templates
```

This creates:

- `abs_template.csv` – example absorbance values
- `meta_template.csv` – matching metadata (Blank / NegControl / doses)

You can open and edit these templates in Excel, LibreOffice, or any text editor and then save as your own `abs.csv` and `meta-abs.csv`.

## 2. Prepare Your Data
The script expects **two CSV files**, both using `;` (semicolon) as separator and a plate layout:

**Absorbance file (e.g. abs.csv)**
- Rows: well letters (`A`–`H` for a 96‑well plate)
- Columns: well numbers (`1`–`12`)
- First cell is empty (acts as corner)
- Values: numeric absorbance readings

Example (`;` separated):

```
;1;2;3;4
A;0.120;0.118;0.119;0.121
B;0.122;0.117;0.121;0.119
C;0.123;0.116;0.118;0.120
D;0.121;0.119;0.120;0.118
```

**Metadata file (e.g. meta-abs.csv)**
Same layout as the absorbance file, but values describe the condition:

- `Blank` – blank wells
- `NegControl` – negative control wells
- `<dose>uM` – numerical dose in µM, e.g. 0uM, 1uM, 10uM, 100uM

Example:

```
;1;2;3;4
A;Blank;Blank;NegControl;NegControl
B;0uM;1uM;10uM;100uM
C;0uM;1uM;10uM;100uM
D;0uM;1uM;10uM;100uM
```

The script parses doses from the `<number>uM` pattern.

# 3. Run the Analysis

Once you have your absorbance and metadata files, run:

```
python main.py --abs abs.csv --meta meta-abs.csv
```

Key outputs:

- Console summary:
    - Mean blank and control absorbance
    - Relative IC₅₀ (4PL parameter)
    - Absolute IC₅₀ (dose at 50% viability)
    - Hill slope
    - R² of the fit
- Plot:
    - Stored as dose_response_4PL_<abs_filename_stem>.png
    - Shows:
        - Raw data points
        - Fitted 4PL curve
        - 95% confidence band
        - Text box with IC₅₀, slope, and R²

## Implementation Details

The core components are:

- `load_plate_with_headers(...)`
Reads a ;‑separated plate file into long format with Well identifiers (e.g. A1, B3).

- `parse_condition_type(...)`
Converts metadata strings into:

    - "Blank", "Control", or
    - numeric dose values (extracted from <number>uM).

- Viability calculation
For each well:

```
Viability (%) = (Absorbance - mean_blank) / (mean_control - mean_blank) * 100
```

- `four_param_logistic(...)`
Standard 4PL model:

```
y(x) = lower + (upper - lower) / (1 + (x / IC50)^slope)
```

- `fit_dose_response(...)`
Uses `scipy.optimize.curve_fit` with the Levenberg–Marquardt algorithm (`method="lm"`) to estimate:

    - `lower`, `upper`, `IC50`, `slope`, plus the covariance matrix.

- Confidence band
`calculate_confidence_interval(...)` uses the delta method (first‑order Taylor expansion) and the parameter covariance to compute a 95% confidence band around the fitted curve.

- Plotting
`plot_dose_response(...)` renders:

    - Log‑scaled x‑axis (dose in µM)
    - Viability on y‑axis
    - Curve, confidence ribbon, and data points.

## Example

The script can generate example plate templates that you can run end‑to‑end.

### 1. Generate templates

```bash
python main.py --make-templates
```

This creates:

- abs_template.csv
- meta_template.csv

### 2. Copy templates to working files
For a quick test, copy them to the filenames you plan to use:

```
cp abs_template.csv abs.csv
cp meta_template.csv meta-abs.csv
```

You can inspect/edit them in Excel/LibreOffice if you like.

### 3. Run the analysis
```
python main.py --abs abs.csv --meta meta-abs.csv
```

You should see:

- Summary statistics and fitted parameters printed to the terminal
- A PNG figure saved as:

```
dose_response_4PL_abs.png
```

The plot will show:

- Raw data points from `abs.csv`
- The 4PL fitted curve
- A 95% confidence interval ribbon
- An annotation box with IC₅₀, slope, and R²

# Example output

![4PL dose response](dose_response_4PL_251203 - eal_abs4.png)

## Reproducibility and Extensions
You can extend this script by:

- Exporting fit results to CSV/JSON
- Adding support for:
    - multiple compounds per plate (e.g. by adding an extra metadata dimension),
    - different plate formats (e.g. 384‑well),
    - alternative models (3PL, 5PL).
- Integrating into a larger analysis pipeline or notebook.

Contributions, issues, and suggestions are welcome via GitHub.
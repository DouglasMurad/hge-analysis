# HGE spectral analysis

Python scripts for processing reflection measurements from LH and RH shuriken metasurfaces with achiral, R-chiral and S-chiral dye coatings.

## Scripts

- `hge_origin.py` builds Origin worksheets and CD, g-factor and reflectivity plots.
- `gfactor_heights.py` extracts LH and RH g-factor heights, saves annotated plots and writes the height ratios to Excel. It retains the accepted P1F_FIX_V7 selections and version label.

## Setup

Edit `ROOT` near the top of each script to point to the experiment folder. The supplied path, `C:\HGE_Data`, is a placeholder.

The data are arranged by coating, then P1, P2 and P3, with geometry measurements A–I inside. Filenames identify each measurement, for example:

```text
Achiral_P1_A_LH_QWP45_A0.asc
Achiral_P1_A_LH_QWP135_A0.asc
Achiral_BG_QWP0_A0.asc
```

ASC files must contain two numeric columns: wavelength in nm and intensity. Wavelengths must increase and match between paired measurements. The Origin script also requires QWP0 and QWP90 measurements and the four gold-reference channels for each coating.

Install the height script dependencies in the Python environment used to run it:

```text
python -m pip install -r requirements.txt
python gfactor_heights.py
```

For Origin, open a blank project and run the appropriate command in its Script Window after adjusting the path:

```text
run -pyf "C:\hge-analysis\hge_origin.py";
run -pyf "C:\hge-analysis\gfactor_heights.py";
```

The Origin workflow needs Origin's embedded Python with `originpro` available. The height script can also run independently of Origin.

## Calculations

Each intensity channel is corrected as `(I - mean(first 5 points)) / 20`.

- CD: `I45 - I135`
- g-factor: `2 * (I45 - I135) / (I45 + I135)`
- Reflectivity: `I0 / gold_I0`
- Height: selected positive maximum minus selected negative minimum
- Height ratio: LH height divided by RH height

The height detector searches for the first substantial opposite-sign lobe pair between 550 and 800 nm. Smoothing is used for detection only; heights use the original g-factor values. Reviewed windows in `EXTREMUM_WINDOWS` override individual selections for this dataset. These windows need checking before applying the script to a different experiment.

Plots cover 550–850 nm, with the region above 800 nm shaded. Inspect the marked extrema before interpreting ratios. The ratio measures the selected spectral heights; it does not by itself establish molecular chirality dependence.

## Outputs

The height script saves a timestamped results folder beside the script, containing up to 81 plots and `Gfactor_Height_Ratios.xlsx`. The workbook has Achiral, RChiral and SChiral sheets with Structure, LH height, RH height and LH/RH ratio. Failed structures have blank values and an entry in `Errors.txt`.

The Origin script saves a timestamped folder inside the input data folder, containing `HGE_V2.opju` and an error report if needed. Each main worksheet has 133 columns, A–EC. It does not export CSVs or additional analysis workbooks.

The height script ignores directories ending in `_0-90_Test`. The Origin script retains its existing recursive discovery and reports duplicate measurements, so keep test copies outside its input tree.

Raw data and generated results are excluded from Git by `.gitignore`. No experimental data are included here.

## Validation

This packaging revision changes comments and replaces the personal input path with a placeholder. Python syntax was checked and the parsed code was compared with the original scripts to confirm that no other executable code changed. Origin execution still requires the local Origin installation.

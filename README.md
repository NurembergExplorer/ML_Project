## Nuremberg Land-Cover Intelligence

This project predicts and analyzes land-cover composition over a 100m grid for Nuremberg, then serves interactive area-based insights in a Streamlit app.

Main classes:
- Built-up
- Vegetation
- Water
- Other

## Quick Start

If processed predictions and population files already exist, run only these commands:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

If predictions are missing, run the pipeline in the **Data Preparation and Training Pipeline** section first.

## Repository Layout

- `src/`: data pipeline, feature extraction, training, prediction, and evaluation scripts
- `data/raw/`: raw Sentinel and WorldCover inputs
- `data/processed/`: grid, features, labels, tables, predictions, and population by year
- `models/`: trained model artifacts and validation summaries
- `app/streamlit_app.py`: interactive application

## Environment Setup

### Option 1: pip + venv (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Option 2: conda geospatial baseline

```bash
conda install -c conda-forge geopandas rasterio shapely pyproj fiona
```

## Data Preparation and Training Pipeline

### 1) Build 100m grid

```bash
python src/make_grid.py --raster data/raw/sentinel/S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif --cell-size 100 --block-size 1000 --output data/processed/grid/grid_100m.gpkg
```

### 2) Extract yearly Sentinel features

```bash
python src/extract_features.py --raster data/raw/sentinel/S2_2019_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2019.parquet --year 2019
python src/extract_features.py --raster data/raw/sentinel/S2_2020_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2020.parquet --year 2020
python src/extract_features.py --raster data/raw/sentinel/S2_2021_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2021.parquet --year 2021
python src/extract_features.py --raster data/raw/sentinel/S2_2022_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2022.parquet --year 2022
python src/extract_features.py --raster data/raw/sentinel/S2_2023_Jun01_Aug31_10m_QA60SCL_F32.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/features/features_2023.parquet --year 2023
```

### 3) Extract labels from WorldCover

```bash
python src/extract_labels.py --raster data/raw/worldcover/WorldCover_2020_10m.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/labels/labels_2020.parquet --year 2020
python src/extract_labels.py --raster data/raw/worldcover/WorldCover_2021_10m.tif --grid data/processed/grid/grid_100m.gpkg --output data/processed/labels/labels_2021.parquet --year 2021
```

### 4) Build training table

```bash
python src/build_training_table.py --grid data/processed/grid/grid_100m.gpkg --features-dir data/processed/features --labels-dir data/processed/labels --years 2020 2021 --output data/processed/tables/train_table.parquet
```

### 5) Train models

```bash
python src/train_models.py --train data/processed/tables/train_table.parquet --outdir models --spatial-folds 5 --optuna-trials 30 --ensemble-size 5
```

### 6) Predict all years

```bash
python src/predict_all_years.py --features-dir data/processed/features --years 2019 2020 2021 2022 2023 --model-dir models --output-dir data/processed/predictions --include-uncertainty
```

## Run the App

```bash
streamlit run app/streamlit_app.py
```

## Current App Capabilities

- Draw polygon/rectangle area of interest on the map
- Composition mode: class shares for a selected year
- Change mode: start/end comparison with class-wise deltas
- Proportional map fill and dominant-class map fill
- Selection narrative cards for composition and change
- Model reliability context metrics
- What-if simulator in change mode with calibrated scenario density estimate
- Population context summary in change mode under reliability section
- Export options:
	- Selection report (CSV)
	- Executive summary (PDF)

## Required Data for App

- Grid: `data/processed/grid/grid_100m.gpkg`
- Predictions: `data/processed/predictions/pred_YYYY.parquet`
- Population: `data/processed/population/pop_YYYY_by_cell_id.csv`
- Model summaries (for reliability context):
	- `models/metrics_spatial_cv_summary.csv`
	- `models/uncertainty_summary.csv`

## Notes

- If map selection does not register, clear polygon and redraw a closed shape.
- For smooth rendering, very large AOIs may automatically switch to dominant-class rendering.




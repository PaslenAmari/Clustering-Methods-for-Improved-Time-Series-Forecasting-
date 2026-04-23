# Cluster-Ensemble Time Series Forecasting for Global Electricity Load
## Project Overview

This repository contains the research codebase for evaluating a cluster-ensemble approach to time series forecasting, applied to highly heterogeneous global energy data. Instead of training a single monolithic model for all regions or isolated models for each specific region, this project implements an intermediate ensemble strategy:

1. **Clustering**: Grouping electricity consumption profiles of different countries/regions based on their temporal shape and patterns.
2. **Dynamic Model Selection**: Utilizing internal Cross-Validation (CV) and Occam's Razor principle to assign the most robust forecasting model (e.g., CatBoost or Ridge Regression) to each distinct cluster.
3. **Evaluation**: Comparing the proposed Cluster-Ensemble approach against a Global baseline across a rigorous 4-window temporal backtest, validated by the non-parametric Wilcoxon Signed-Rank Test.

**Dataset:** The experiments are conducted on the [Worldwide Electricity Load Dataset](https://data.mendeley.com/datasets/ybggkc58fz/1) (Mendeley Data).

## Methodology

- **Clustering:** Dynamic Time Warping (DTW) combined with Agglomerative Clustering (and KMeans fallback for highly unbalanced distributions) to capture shape-based similarities invariant to absolute magnitude.
- **Forecasting Models:** ETNA framework implementations of `CatBoostMultiSegmentModel` and `SklearnMultiSegmentModel` (Ridge).
- **Validation:** 4-window chronological backtesting with a 30-day forecast horizon (h=30).
- **Statistical Testing:** Wilcoxon Signed-Rank Test applied to panel data (cross-sectional regional MAE) to rigorously assess the statistical significance of performance differences.

## Getting Started

### Prerequisites
You need **Docker** and **Docker Compose** installed on your machine to replicate the isolated environment.

### Data Preparation
Before running the pipeline, you must download the dataset manually:
1. Download the dataset from [Mendeley Data](https://data.mendeley.com/datasets/ybggkc58fz/1).
2. Extract the archive and place the `GloElecLoad` folder inside the repository following this exact structure:
   
   ```text
   data/
   └── Worldwide_electricity_load/
       └── Worldwide Electricity Load Dataset/
           └── GloElecLoad/
               ├── Australia/
               ├── Austria/
               └── ...
   ```

### Running the Pipeline
Once the data is in place, build the image and execute the experiment by running:

```bash
docker-compose up --build
```

The script will automatically:
1. Parse, filter, and preprocess the global regional time series.
2. Standardize series and compute the DTW distance matrix.
3. Perform clustering and dynamic cluster-model selection via internal CV.
4. Execute the 4-window backtest comparing the Global model vs. the Ensemble.
5. Calculate evaluation metrics (MAE, sMAPE) and run the Wilcoxon Signed-Rank Test.
6. Save generated plots and CSV metrics to the `results/` directory.

## Project Structure
- `powercons_etna.py`: The main execution pipeline containing data processing, clustering, and forecasting logic.
- `Dockerfile` & `docker-compose.yml`: Containerization and environment configuration.
- `requirements.txt`: Strict Python dependencies.
- `results/`: Output directory for generated visualizations and performance metrics.

## Key Results
- Successfully grouped heterogeneous global electricity loads into distinct archetypes based on normalized consumption behavior.
- Demonstrated that integrating the Occam's Razor principle (favoring linear models for stable clusters) reduces overfitting.
- The Cluster-Ensemble approach significantly outperformed the monolithic Global baseline (p < 0.05 via Wilcoxon Signed-Rank Test), proving the robustness of the methodology for heterogeneous panel data.

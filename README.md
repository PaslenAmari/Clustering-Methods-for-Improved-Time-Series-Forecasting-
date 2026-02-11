# Clustering Methods for Improved Time Series Forecasting

## Project Overview

This project explores how grouping similar time series together (clustering) can help build better forecasting models. Instead of training one giant model for everyone or many tiny models for each meter, was used an ensemble approach:
1. **Clustering**: Grouping electricity meters based on their consumption patterns.
2. **Forecasting**: Training specialized models for each group using ETNA's `CatBoostMultiSegmentModel`.
3. **Evaluation**: Comparing this ensemble approach against a global baseline using backtesting and statistical tests (Diebold-Mariano).

There were implemented three different clustering approaches to find the most robust patterns:
- **TSFresh**: Statistical feature extraction.
- **DTW (Dynamic Time Warping)**: Shape-based similarity.
- **sktime**: Native time-series clustering.

## Getting Started

### Prerequisites
You need **Docker** and **Docker Compose** installed on your machine.

### Running the Pipeline
The entire environment is containerized. To build the image and start the experiment, run:

```bash
docker-compose up --build
```

The script will automatically:
1. Download the PowerCons dataset.
2. Run the clustering benchmarks.
3. Perform a 6-window backtest for the forecasting models.
4. Save plots and metrics to the `results/` folder.

## Project Structure
- `powercons_etna.py`: The main research pipeline.
- `Dockerfile` & `docker-compose.yml`: Environment configuration.
- `requirements.txt`: Python dependencies.
- `results/`: (Generated) Contains plots and CSV metrics.

## Key Results
- Successfully identified **4 distinct archetypes** of energy consumers.
- Built an end-to-end pipeline that handles multi-segment forecasting with cross-learning.
- Established a statistical framework for model comparison.

## Future Work (Semester 2)
- Сontext-aware forecasting framework capable of handling "Drift" (sudden changes in consumer behavior).
- Ensemble self-correction: automatic re-clustering and model retraining when drift is detected.s

# UFC Fight Predictor

A machine learning project that predicts UFC fight outcomes using historical fight data, fighter statistics, and an Elo rating system.

The application can predict custom matchups between fighters, generate predictions for upcoming UFC events, and automatically update itself by scraping newly completed fights and retraining the model.

The project is built entirely in Python and includes a neural network model, automated data scraping, feature engineering, and an interactive command-line interface.

## Features

The CLI (`cli.py`) provides three main options:

### Predict Fight Outcome

Enter the names of two fighters and receive:

- Predicted winner
- Win probability for each fighter

Example:

```text
Ilia Topuria            60.0%
Arman Tsarukyan         40.0%
Predicted winner: Ilia Topuria
```

### Display Future Predictions

The system automatically:

- Scrapes the next two upcoming UFC events from UFCStats
- Retrieves fighter profile information
- Generates predictions for every fight on those cards

Results are cached locally to reduce unnecessary scraping and improve performance.

### 🔄 Retrain Model

The retraining pipeline can:

- Scrape newly completed UFC events
- Add new fights to the historical dataset
- Remove duplicate entries
- Retrain the model
- Update saved model artifacts

Before retraining, the current dataset and model files are automatically backed up.

## Quick Start

```bash
cd main_prediction_system
pip install -r requirements.txt
python cli.py
```

A trained model is already included in the `artifacts/` folder.

To train the model from scratch:

```bash
python train.py
```

## Project Layout

```text
UFC_Fight_Predictor/
├── main_prediction_system/        # Live application
│   ├── cli.py                     # Interactive menu (entry point)
│   ├── service.py                 # Orchestration layer
│   ├── preprocessor.py            # Feature engineering pipeline
│   ├── train.py                   # Model training
│   ├── predict.py                 # Matchup and card prediction
│   ├── scrape_results.py          # Completed fight scraper
│   ├── scrape_upcoming.py         # Upcoming card scraper
│   ├── http_client.py             # Shared HTTP client
│   ├── config.py                  # Paths and configuration
│   ├── requirements.txt
│   ├── mayberealdata_enriched.csv # Master training dataset
│   ├── artifacts/                 # Trained models and metadata
│   ├── cache/                     # Cached upcoming card data
│   └── backups/                   # Dataset and model backups
├── future_card_prediction/        # Legacy prediction scripts
└── archive/                       # Older experiments and notebooks
```

## How It Works

### Feature Pipeline

The preprocessing pipeline converts raw UFC fight data into machine learning features.

For each fighter, the system reconstructs their career chronologically and calculates pre-fight statistics such as:

- Win rate
- Win streak
- Striking efficiency
- Grappling efficiency
- Experience
- KO and submission rates
- Elo rating

Only information available before a fight occurred is used, preventing data leakage.

### Matchup Generation

The system compares two fighters by creating matchup features such as:

```text
Win Rate Difference
Elo Difference
Reach Difference
Height Difference
Age Difference
```

These directional comparisons become the inputs to the neural network.

### Model Training

The model is a feed-forward neural network built with PyTorch.

Training uses a chronological split:

```text
Oldest Fights  → Training Set
Middle Fights  → Validation Set
Newest Fights  → Test Set
```

This better reflects how the model will be used when predicting future fights.

Typical test performance is approximately:

```text
Accuracy: 64–66%
AUC: ~0.69–0.70
```

### Predictions

For future fights, the latest fighter snapshots are generated from historical data.

The system compares both fighters, creates matchup features, and outputs win probabilities for each side.

Fighters with no historical UFC data are flagged rather than guessed, while fighters with limited history use imputed features to allow predictions to proceed.

## Interactive CLI

Run the application using:

```bash
python cli.py
```

Menu:

```text
==============================
   UFC Fight Predictor
==============================
1) Predict fight outcome
2) Display future predictions
3) Retrain model
0) Exit
```

## Data Scraping

The project uses UFCStats as its primary data source.

The scraping system can:

- Collect upcoming fight cards
- Collect completed fight results
- Retrieve fighter profile information
- Update the training dataset automatically

UFCStats currently uses a JavaScript proof-of-work challenge to discourage automated scraping. The project includes a custom HTTP client that solves this challenge and maintains a valid session, allowing scraping to continue without requiring a browser.

## Retraining

Selecting the retraining option performs the following workflow:

1. Back up the current dataset and model artifacts.
2. Scrape newly completed UFC events.
3. Append new fights to the master dataset.
4. Remove duplicate entries.
5. Retrain the model.
6. Save updated artifacts.
7. Reload the new model into memory.

Because only a small number of fights are typically added between updates, model performance usually changes only slightly after each retraining cycle.

## Model Artifacts

Trained models and metadata are stored in:

```text
artifacts/
```

Including:

```text
ufc_nn_model.pt
preprocessor.joblib
feature_cols.json
config.json
metrics.json
train_history.json
```

## Notes & Limitations

- MMA is highly unpredictable, so probabilities should be treated as estimates rather than guarantees.
- Fighter name matching currently requires the exact spelling used on UFCStats.
- Scraping depends on UFCStats maintaining a compatible page structure.
- The `cache/` and `backups/` directories are generated automatically and do not need to be committed.

## Future Improvements

Planned enhancements include:

- Fuzzy fighter name matching and suggestions
- Scheduled automatic retraining
- Automatic generation of future card predictions
- Prediction tracking against actual fight outcomes
- Feature importance visualisations
- A web-based dashboard interface

## Disclaimer

This project is intended for educational and research purposes. Fight outcomes are inherently uncertain, and predictions should not be interpreted as guaranteed results.

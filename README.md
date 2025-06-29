# Crash Prediction

This project predicts upcoming crash values based on previous rounds. It uses a simple machine learning approach to fit a regression model on recent results and output predictions for unseen rounds.

## Requirements

Install the following Python packages:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install them with pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Place your `crash_data.csv` file in the project root (same directory as `crash_prediction_basic.py`). The CSV should contain at least a `round` column and a `crash_value` column.
2. Run the script:

```bash
python3 crash_prediction_basic.py
```

The script will visualize the data, train a regression model and display predicted versus actual crash values along with the mean squared error.

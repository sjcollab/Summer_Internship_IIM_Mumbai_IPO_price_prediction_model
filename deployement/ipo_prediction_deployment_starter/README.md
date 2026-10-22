# IPO One-Month Price Prediction App

This project packages a TensorFlow regression model in a Streamlit web app.

## Important target assumption

The training target `Price_after_1m` is assumed to contain the actual rupee share price divided by 1,000. The app reverses the fitted `MinMaxScaler`, then multiplies by 1,000 to display rupees.

Do not use this package until you confirm this definition applies to every row in your full dataset.

## 1. Prepare your dataset

1. Put your cleaned dataset in `data/ipo_data.csv`.
2. Ensure it contains `Price_after_1m`.
3. Export your cleaned Excel sheet to CSV if necessary.
4. Make sure all model input columns use the same definitions and units as the training data.

The training script drops known identifier columns, ticker/date columns, and future targets (`Price_after_3m`, `Price_after_6m`). It selects numeric features with absolute correlation >= 0.05, calculated on the training split only. Review `DROP_COLUMNS` and the selected feature list for your dataset before relying on results.

## 2. Install dependencies

Use a Python version compatible with the installed TensorFlow release, preferably a clean virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Train and save model artifacts

```bash
python train_model.py
```

This writes the model, fitted scalers, imputation values, feature list, metrics, and held-out test predictions to `artifacts/`.

## 4. Run the app locally

```bash
streamlit run app.py
```

Streamlit will show a local URL in your terminal. Open it in your browser and test predictions.

## 5. Publish it online

1. Create a GitHub repository.
2. Add `app.py`, `train_model.py`, `requirements.txt`, `README.md`, and the required model artifacts.
3. Do not commit private data, API keys, or sensitive information.
4. Create a Streamlit Community Cloud app connected to your GitHub repository and select `app.py` as the entry point.

Large model artifacts or hosting limits may require a different hosting option. Test the deployed app after publishing.

## Model caveats

- This is a baseline neural network, not evidence of profitable IPO forecasting.
- Evaluate on a time-based holdout if the goal is forecasting future IPOs; a random split can overstate performance for time-dependent data.
- Avoid features that would not be known at the time the prediction is meant to be made.
- Feature selection and imputation are fitted on training data only in this script.
- The app uses one numeric field per selected feature; if your model needs categorical variables, add explicit encoding rather than silently coercing them to numbers.
- Never describe predictions as guaranteed.

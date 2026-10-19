# TransactionLens — Banking Transaction Risk Analytics

A Streamlit-based analytics dashboard that explores banking transactions and uses an unsupervised Isolation Forest model to flag unusual transaction patterns for further review.

> **Project attribution:** This project is adapted from the original repository by its original author. Please retain the original author's credit and comply with the repository's license. Add your own contributions and experiments as you make them.

## Overview

TransactionLens demonstrates a basic transaction-risk analysis workflow:

- Explore transaction data and data-quality metrics.
- Engineer features for anomaly detection.
- Score transactions using an Isolation Forest model.
- Review flagged transactions and their risk indicators.
- Explore analytical queries using SQL.
- Inspect model outputs through an interactive dashboard.

## Technology Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** scikit-learn, Isolation Forest
- **Dashboard:** Streamlit
- **Database and analysis:** SQLite, SQL
- **Model artifacts:** Joblib / pickle-based pipeline (as used by the project)

## Project Structure

```text
TransactionLens/
├── app/                 # Streamlit dashboard
├── data/
│   ├── raw/             # Input transaction data
│   └── processed/       # Processed data and database
├── models/              # Saved model artifacts and metadata
├── notebooks/           # Exploratory analysis
├── sql/                 # SQL analytics queries
├── src/                 # Data processing and ML modules
├── train_and_save_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

1. Install Python compatible with the project's dependencies.
2. Clone your repository and navigate to the project directory.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

5. If the saved model artifacts are unavailable or need to be regenerated, review the training script and run it:

```bash
python train_and_save_model.py
```

6. Launch the dashboard:

```bash
python -m streamlit run app/app.py
```

Follow the dashboard instructions to explore the available features.

## Understanding the Results

Isolation Forest is an **unsupervised anomaly-detection algorithm**. It identifies observations that differ from patterns learned from the input data.

- A flagged transaction is an unusual observation, **not proof of fraud**.
- Risk scores and Low / Medium / High categories depend on the implementation's scoring and threshold logic.
- Anomaly flags must not be interpreted as a verified fraud rate.
- Feature-based explanations describe signals used by the implementation; they do not establish the cause of a transaction.

## Limitations

- A small dataset may not represent real-world banking activity.
- Without verified fraud labels, precision, recall, and fraud-detection accuracy cannot be established.
- The model's contamination setting is an assumption about expected anomalies, not a measured fraud prevalence.
- The model does not necessarily update automatically when new transactions arrive.
- Production deployment would require robust validation, monitoring, security, privacy controls, and domain-expert review.

## Future Improvements

- Evaluate against a labelled dataset.
- Compare anomaly-detection methods and validate thresholds.
- Add transaction-velocity and account-level behavioural features.
- Improve explanation quality and model monitoring.
- Add automated tests and reproducible evaluation metrics.
- Strengthen privacy, authentication, and deployment controls.

## Responsible Use

This project is for educational and analytical demonstration. It is not a production fraud-detection service and should not be used to make real banking decisions without proper validation and safeguards.

# AI-Powered IT Service Desk Intelligence System

An NLP and machine-learning portfolio project that classifies IT support
tickets from their text descriptions and suggests a support destination.

## What it does

- Predicts an IT ticket category from its text.
- Displays the model's estimated class scores.
- Suggests a support group using a transparent category-to-team mapping.
- Supports batch classification from a CSV file.
- Displays evaluation metrics and a confusion matrix.
- Compares Logistic Regression and Linear SVM baselines on a validation split.

This is a research/portfolio prototype, not a production IT service-management
system. Predictions should be reviewed by a human.

## Dataset

This project uses **Classification of IT Support Tickets**, distributed as
`tasksource/it-support-tickets` on Hugging Face.

- 2,229 ticket texts.
- Seven manually assigned categories.
- Tickets were collected from an IT support company in the Florianópolis
  region of Brazil and classified by three IT support professionals.
- The dataset is multilingual.
- The dataset authors report removing personally identifiable and sensitive
  information.
- License: CC BY 4.0.

Dataset page:
https://huggingface.co/datasets/tasksource/it-support-tickets

Original dataset DOI:
https://doi.org/10.5281/zenodo.7648117

Attribution:
Leonardo Santiago Benitez Pereira, *Classification of IT Support Tickets*
(2022), Zenodo, DOI: 10.5281/zenodo.7648117.

The support-team mapping in this project is an illustrative mapping created
for the application. It is not supplied by the dataset authors.

## Technology stack

- Python
- Pandas and NumPy
- Hugging Face Datasets
- Scikit-learn
- TF-IDF text features
- Logistic Regression and Linear SVM baseline comparison
- Streamlit
- Matplotlib
- Joblib

## Project structure

```text
it_service_desk_intelligence/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── ticket_classifier.joblib   # generated after training
    ├── metadata.json               # generated after training
    ├── evaluation.json             # generated after training
    └── confusion_matrix.png        # generated after training
```

## Setup (Windows)

Open a terminal inside this project folder.

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

The script downloads the dataset, compares two baseline models on a
validation split, trains the Logistic Regression pipeline, evaluates it on
the holdout test split, and saves model/evaluation artifacts in `artifacts/`.

The first run requires an internet connection.

## Run the dashboard

```bash
streamlit run app.py
```

Streamlit starts a local web application in your browser.

## CSV format for batch predictions

Create a CSV with a column named `text`:

```csv
text
"My laptop cannot connect to the office VPN"
"Outlook is not sending emails"
"I need access to a shared folder"
```

Use the Batch prediction page to upload it.

## Methodology

1. Load the public ticket dataset.
2. Inspect and clean empty ticket descriptions.
3. Convert text to TF-IDF features using unigrams and bigrams.
4. Compare Logistic Regression and Linear SVM on a validation split.
5. Train a Logistic Regression baseline.
6. Evaluate on a holdout test split using accuracy, macro F1,
   per-class precision/recall/F1 and a confusion matrix.
7. Save the trained pipeline and launch the Streamlit application.

The model score shown by the dashboard is an estimated class probability
from Logistic Regression. It is not a guarantee of correctness and has not
been calibrated in this starter version.

## Important limitations and future improvements

- The dataset is relatively small and multilingual; performance may vary by
  language and ticket-writing style.
- The seven dataset categories may not match another organisation's ticket
  taxonomy.
- The support-team routing rules are illustrative and require validation.
- No live ticketing-system integration is included.
- No production monitoring, authentication, SLA workflow or human feedback
  loop is included.

Possible next steps:
- Add language detection and language-specific evaluation.
- Tune hyperparameters using cross-validation.
- Calibrate model confidence and select a review threshold from validation data.
- Add explainability using top TF-IDF features or SHAP-compatible methods.
- Add a human feedback mechanism and retraining workflow.
- Add a REST API and containerise the app for deployment.

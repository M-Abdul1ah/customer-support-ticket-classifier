# Customer Support Ticket Classifier

An ML-powered system that classifies customer support messages into intent categories, with a calibrated confidence score used to decide whether a ticket can be **auto-handled** or needs **human review**.

This is the ML core of a larger customer-support automation platform (see `docs/` — planned n8n workflow orchestration and business-rule routing layer, not yet built).

## What it does

A customer writes a message in plain English:

> "I forgot my password"

The API returns:

```json
{
  "ticket": "I forgot my password",
  "intent": "recover_password",
  "confidence": 0.97,
  "threshold": 0.90,
  "status": "AUTO",
  "department": "account_support",
  "priority": "normal",
  "requires_human": false,
  "escalation_reason": null
}
```

If confidence is below the threshold, `status` becomes `REVIEW` and `requires_human` becomes `true`, meaning the ticket should be routed to a human agent instead of handled automatically.

### Business rules layer

`intent` and `confidence` come from the ML model. Everything else (`department`, `priority`, `requires_human`, `escalation_reason`) comes from `src/business_rules.py` — a separate, non-ML layer that decides what to *do* with a prediction. This is deliberately kept apart from the model so routing/priority policy can change without retraining anything.

One important rule lives here: if a ticket contains language suggesting fraud, unauthorized access, or a security issue, it is **always** escalated to a human and marked high priority — even if the model is highly confident about the predicted intent. Confidence measures how sure the model is about the *intent*, not how safe it is to fully automate a *sensitive* ticket.

## Dataset

[bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) — a synthetic, LLM-generated customer support dataset covering common intents (order tracking, refunds, password recovery, billing, etc.).

## Model

- **Features:** Character-level TF-IDF (n-grams 3–5)
- **Classifier:** Linear SVM, calibrated with `CalibratedClassifierCV` (sigmoid, 3-fold CV) to produce usable probability estimates
- **Test accuracy:** 99.83% (see `reports/model_comparison.md` for the full comparison against word-level TF-IDF)

**Note on accuracy:** this number is high because the dataset is synthetically generated with fairly consistent phrasing per intent, not because the task is trivial in general. On real, messier customer messages (typos, slang, mixed intents) accuracy would be expected to be meaningfully lower. Two things in this repo are aimed at that gap: character-level n-grams (more robust to misspellings than word-level TF-IDF — see `test_misspelled_order_tracking` in `tests/test_predictions.py`) and probability calibration, so the confidence score is meaningful enough to gate automation rather than just reporting the raw model accuracy.

Train/test split is done **before** fitting the vectorizer, so there is no data leakage.

## Project structure

```
├── eda.py / eda_visuals.py / inspect_data.py   # dataset exploration
├── train_baseline.py                            # word TF-IDF + Logistic Regression
├── train_char_baseline.py                       # char TF-IDF + Logistic Regression
├── train_char_svm.py                             # char TF-IDF + Linear SVM
├── train_combined_baseline.py                    # word + char combined features
├── src/
│   ├── train.py                # final uncalibrated model (char TF-IDF + SVM)
│   ├── train_calibrated.py     # final model + probability calibration
│   ├── predict.py              # inference helper (loads calibrated model)
│   ├── threshold_analysis.py   # sweeps confidence thresholds
│   ├── evaluate_confidence.py  # confidence distribution analysis
│   └── api.py                  # FastAPI service
├── tests/
│   ├── test_predictions.py     # intent correctness + misspelling robustness
│   └── test_api.py             # live API endpoint tests (requires server running)
├── reports/
│   ├── model_comparison.md
│   ├── threshold_analysis.csv / threshold_analysis_calibrated.csv
│   ├── confidence_analysis.csv / confidence_analysis_calibrated.csv
│   └── figures/                # class distribution, text length plots
└── models/                     # generated locally, not committed (see .gitignore)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the model

```powershell
python src/train_calibrated.py
```

This downloads the dataset, trains, calibrates, evaluates, and saves artifacts to `models/`:
- `tfidf_vectorizer_calibrated.joblib`
- `classifier_calibrated.joblib`
- `metadata_calibrated.json`

## Run the API

```powershell
uvicorn src.api:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API docs.

Endpoints:
- `GET /health` — service status and current threshold
- `GET /` — service info
- `POST /predict` — classify a ticket, e.g. `{"text": "I forgot my password"}`

## Run the tests

```powershell
# Unit tests (model loaded directly, no server needed)
pytest tests/test_predictions.py

# API tests (start the server first, in a separate terminal)
uvicorn src.api:app
pytest tests/test_api.py
```

## Status

| Component | Status |
|---|---|
| Dataset exploration / EDA | ✅ Done |
| Baseline model comparison | ✅ Done |
| Final model + calibration | ✅ Done |
| Confidence threshold analysis | ✅ Done |
| FastAPI service | ✅ Done |
| Tests (unit + API) | ✅ Done |
| Business-rule routing layer (priority, department, escalation) | ✅ Done |
| n8n automation workflow | ⬜ Not started |
| Docker | ⬜ Not started |
| Deployment | ⬜ Not started |

## Roadmap

Planned next: an orchestration layer (n8n) that calls this API, applies business rules on top of the raw prediction (e.g. security-related tickets always escalate regardless of confidence), and routes tickets to the right team/CRM/human agent. See project notes for the full architecture.

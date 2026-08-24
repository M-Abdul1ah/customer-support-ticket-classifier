Customer Support Ticket Classifier



An ML-powered system that classifies customer support messages into intent categories, with a calibrated confidence score used to decide whether a ticket can be auto-handled or needs human review.



This is the ML core of a larger customer-support automation platform (see project notes - planned n8n workflow orchestration on top of this API).



What it does



A customer writes a message in plain English:



"I forgot my password"



The API returns a JSON object with fields: ticket, intent, confidence, threshold, status, department, priority, requires\_human, escalation\_reason.



If confidence is below the threshold, status becomes REVIEW and requires\_human becomes true, meaning the ticket should be routed to a human agent instead of handled automatically.



Business rules layer



intent and confidence come from the ML model. Everything else (department, priority, requires\_human, escalation\_reason) comes from src/business\_rules.py - a separate, non-ML layer that decides what to do with a prediction. This is deliberately kept apart from the model so routing and priority policy can change without retraining anything.



One important rule lives here: if a ticket contains language suggesting fraud, unauthorized access, or a security issue, it is always escalated to a human and marked high priority, even if the model is highly confident about the predicted intent. Confidence measures how sure the model is about the intent, not how safe it is to fully automate a sensitive ticket.



Dataset



bitext/Bitext-customer-support-llm-chatbot-training-dataset - a synthetic, LLM-generated customer support dataset covering common intents such as order tracking, refunds, password recovery, and billing.



Model



Features: Character-level TF-IDF, n-grams 3 to 5.

Classifier: Linear SVM, calibrated with CalibratedClassifierCV, sigmoid, 3-fold CV, to produce usable probability estimates.

Test accuracy: 99.83 percent. See reports/model\_comparison.md for the full comparison against word-level TF-IDF.



Note on accuracy: this number is high because the dataset is synthetically generated with fairly consistent phrasing per intent, not because the task is trivial in general. On real, messier customer messages accuracy would be expected to be meaningfully lower. Two things in this repo are aimed at that gap: character-level n-grams, which are more robust to misspellings than word-level TF-IDF, and probability calibration, so the confidence score is meaningful enough to gate automation.



Train and test split is done before fitting the vectorizer, so there is no data leakage.



Error analysis



notebooks/04\_error\_analysis.py re-evaluates the saved model on the same held-out test split used during training and groups misclassifications by actual intent versus predicted intent. Result: only 4 errors out of 5375 test examples, 0.07 percent, and all four are genuinely ambiguous inputs rather than a systematic model weakness. Full log saved to notebooks/error\_analysis\_results.csv.



Setup



python -m venv .venv

.venv\\Scripts\\activate

pip install -r requirements.txt



Train the model



python src/train\_calibrated.py



This downloads the dataset, trains, calibrates, evaluates, and saves artifacts to models/: tfidf\_vectorizer\_calibrated.joblib, classifier\_calibrated.joblib, metadata\_calibrated.json.



Run the API



uvicorn src.api:app --reload



Visit http://127.0.0.1:8000/docs for interactive API docs.



Endpoints: GET /health for service status and current threshold, GET / for service info, POST /predict to classify a ticket.



Run the tests



pytest



All tests, test\_predictions.py, test\_business\_rules.py, test\_api.py, run directly with no server required. API tests use FastAPI's TestClient, which calls the app in-process.



Status



Dataset exploration and EDA: Done

Baseline model comparison: Done

Final model and calibration: Done

Confidence threshold analysis: Done

Error analysis, Phase 3: Done

FastAPI service: Done

Tests, model, business rules, API: Done

Business rule routing layer, priority, department, escalation: Done

n8n automation workflow: Not started

Docker: Not started

Deployment: Not started



Roadmap



Planned next: an orchestration layer, n8n, that calls this API and routes tickets to the right team, CRM, or human agent based on the department, priority, and requires\_human fields already returned by the API. See project notes for the full architecture.


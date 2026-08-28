import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from business_rules import apply_business_rules


# ==========================================
# 1. Application setup
# ==========================================

app = FastAPI(
    title="Customer Support Ticket Classifier",
    description="ML-powered customer support ticket intent classification API",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. Configuration
# ==========================================

MODEL_PATH = "models/classifier_calibrated.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_calibrated.joblib"

THRESHOLD = 0.90


# ==========================================
# 3. Load model
# ==========================================

print("Loading ML model...")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("ML model loaded successfully!")

except Exception as e:
    print(f"Failed to load model: {e}")
    raise


# ==========================================
# 4. Request schema
# ==========================================

class TicketRequest(BaseModel):

    text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Customer support ticket text"
    )


# ==========================================
# 5. Health endpoint
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Character TF-IDF + Calibrated Linear SVM",
        "threshold": THRESHOLD
    }


# ==========================================
# 6. Prediction endpoint
# ==========================================

@app.post("/predict")
def predict_ticket(request: TicketRequest):

    text = request.text.strip()

    # Extra validation
    if not text:
        raise HTTPException(
            status_code=400,
            detail="Ticket text cannot be empty."
        )

    # Convert text to TF-IDF
    text_vector = vectorizer.transform([text])

    # Predict intent
    prediction = model.predict(text_vector)[0]

    # Predict probabilities
    probabilities = model.predict_proba(text_vector)

    confidence = float(probabilities.max())

    # Decision
    if confidence >= THRESHOLD:
        status = "AUTO"
    else:
        status = "REVIEW"

    # Business rules layer: department routing, priority, and
    # keyword-based escalation overrides (e.g. fraud/security language
    # always goes to a human, regardless of model confidence).
    routing = apply_business_rules(
        text=text,
        intent=prediction,
        confidence=confidence,
        status=status,
    )

    if routing["requires_human"]:
        status = "REVIEW"

    return {
        "ticket": text,
        "intent": prediction,
        "confidence": round(confidence, 4),
        "threshold": THRESHOLD,
        "status": status,
        **routing,
    }


# ==========================================
# 7. Root endpoint
# ==========================================

@app.get("/")
def root():

    return {
        "name": "Customer Support Ticket Classifier",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
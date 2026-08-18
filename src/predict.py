from pathlib import Path

import joblib


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer_calibrated.joblib"
MODEL_PATH = MODEL_DIR / "classifier_calibrated.joblib"


# ==========================================
# LOAD TRAINED ARTIFACTS
# ==========================================

print("Loading calibrated model...")

vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)

print("Calibrated model loaded successfully!")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_ticket(ticket: str) -> dict:
    """
    Predict the intent of a customer support ticket.

    Returns:
        dict containing:
        - ticket
        - intent
        - confidence
        - status
    """

    if not isinstance(ticket, str):
        raise TypeError("Ticket must be a string.")

    ticket = ticket.strip()

    if not ticket:
        raise ValueError("Ticket cannot be empty.")

    # Convert ticket into TF-IDF features
    ticket_vector = vectorizer.transform([ticket])

    # Predict intent
    prediction = model.predict(ticket_vector)[0]

    # Get calibrated probabilities
    probabilities = model.predict_proba(ticket_vector)[0]

    # Highest probability = confidence
    confidence = float(probabilities.max())

    # Automation decision
    THRESHOLD = 0.90

    if confidence >= THRESHOLD:
        status = "AUTO"
    else:
        status = "REVIEW"

    return {
        "ticket": ticket,
        "intent": prediction,
        "confidence": round(confidence, 4),
        "status": status,
    }


# ==========================================
# TEST PREDICTIONS
# ==========================================

if __name__ == "__main__":

    test_tickets = [
        "I want to cancel my order",
        "Where is my order?",
        "I forgot my password",
        "I need to change my delivery address",
        "Why was my payment declined?",
    ]

    print("\n========== TEST PREDICTIONS ==========")

    for ticket in test_tickets:

        result = predict_ticket(ticket)

        print(f"\nTicket: {result['ticket']}")
        print(f"Predicted intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Status: {result['status']}")
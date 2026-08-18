import joblib
from pathlib import Path


# ==========================================
# Load trained model
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "classifier_calibrated.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer_calibrated.joblib"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# ==========================================
# Helper function
# ==========================================

def predict_intent(text):
    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    return prediction


# ==========================================
# Core intent tests
# ==========================================

def test_cancel_order():
    result = predict_intent("I want to cancel my order")

    assert result == "cancel_order"


def test_recover_password():
    result = predict_intent("I forgot my password")

    assert result == "recover_password"


def test_payment_issue():
    result = predict_intent("Why was my payment declined?")

    assert result == "payment_issue"


def test_change_shipping_address():
    result = predict_intent(
        "I need to change my delivery address"
    )

    assert result == "change_shipping_address"


def test_track_order():
    result = predict_intent("track my order")

    assert result == "track_order"


def test_check_invoice():
    result = predict_intent("I need a copy of my invoice")

    assert result == "check_invoice"


def test_refund():
    result = predict_intent("I want to get a refund")

    assert result == "get_refund"


def test_create_account():
    result = predict_intent("I want to create an account")

    assert result == "create_account"


# ==========================================
# Character-level robustness
# ==========================================

def test_misspelled_order_tracking():
    result = predict_intent("i cant trck my ordr")

    # The model should ideally identify this as
    # an order-tracking request.
    assert result == "track_order"
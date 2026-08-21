"""
Business rules layer.

This module is deliberately kept separate from the ML model (predict.py).
The model only knows "intent" and "confidence". Everything about what the
business should DO with that prediction — which team handles it, how
urgent it is, and whether a human must see it regardless of how confident
the model was — lives here instead.

Why separate:
- These rules change often (new policy, new team, new keyword) and
  should never require retraining the model.
- Some situations (e.g. a possible fraud/security report) must always
  go to a human, even if the model is 99% confident about the intent.
  Confidence measures "how sure the model is about the intent", not
  "how safe it is to fully automate this ticket".
"""

from typing import Optional


# ==========================================
# 1. INTENT -> DEPARTMENT
# ==========================================
# Which team should handle a ticket, based on the predicted intent.
# Source: the 27 intents / 11 categories defined by the Bitext dataset.

DEPARTMENT_MAP = {
    # ACCOUNT
    "create_account": "account_support",
    "delete_account": "account_support",
    "edit_account": "account_support",
    "switch_account": "account_support",
    "recover_password": "account_support",
    "registration_problems": "account_support",

    # CANCELLATION_FEE
    "check_cancellation_fee": "billing",

    # CONTACT
    "contact_customer_service": "general_support",
    "contact_human_agent": "general_support",

    # DELIVERY
    "delivery_options": "logistics",
    "delivery_period": "logistics",

    # FEEDBACK
    "complaint": "customer_relations",
    "review": "customer_relations",

    # INVOICE
    "check_invoice": "billing",
    "get_invoice": "billing",

    # NEWSLETTER
    "newsletter_subscription": "marketing",

    # ORDER
    "cancel_order": "orders",
    "change_order": "orders",
    "place_order": "orders",
    "track_order": "logistics",

    # PAYMENT
    "check_payment_methods": "billing",
    "payment_issue": "billing",

    # REFUND
    "check_refund_policy": "billing",
    "get_refund": "billing",
    "track_refund": "billing",

    # SHIPPING_ADDRESS
    "change_shipping_address": "logistics",
    "set_up_shipping_address": "logistics",
}

DEFAULT_DEPARTMENT = "general_support"


# ==========================================
# 2. INTENT -> BASE PRIORITY
# ==========================================
# Starting priority before any keyword-based escalation is applied.
# Most intents are routine (normal). A few are inherently time-sensitive
# or money-related and start at "high".

HIGH_PRIORITY_INTENTS = {
    "payment_issue",
    "get_refund",
    "track_refund",
    "cancel_order",
    "delete_account",
    "complaint",
}

LOW_PRIORITY_INTENTS = {
    "newsletter_subscription",
    "check_refund_policy",
    "check_cancellation_fee",
    "review",
    "delivery_options",
}


def base_priority(intent: str) -> str:
    if intent in HIGH_PRIORITY_INTENTS:
        return "high"
    if intent in LOW_PRIORITY_INTENTS:
        return "low"
    return "normal"


# ==========================================
# 3. ESCALATION KEYWORDS
# ==========================================
# If a ticket's text contains any of these signals, it is always sent
# to a human and marked high priority — regardless of the model's
# predicted intent or confidence. These are situations where being
# wrong is costly enough that automation should never be trusted.

ESCALATION_KEYWORDS = {
    "fraud",
    "fraudulent",
    "unauthorized",
    "unauthorised",
    "stolen",
    "hacked",
    "hack",
    "scam",
    "scammed",
    "identity theft",
    "legal action",
    "lawsuit",
    "sue",
    "suing",
    "threat",
    "threatening",
    "police",
    "data breach",
}


def matched_escalation_keyword(text: str) -> Optional[str]:
    """
    Return the first escalation keyword found in the text, or None.
    Simple substring match on lowercased text — intentionally simple
    and easy to audit/extend, not a full NLP pass.
    """
    lowered = text.lower()

    for keyword in ESCALATION_KEYWORDS:
        if keyword in lowered:
            return keyword

    return None


# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================

def apply_business_rules(text: str, intent: str, confidence: float, status: str) -> dict:
    """
    Take the raw ML output (intent, confidence, AUTO/REVIEW status) and
    apply business rules on top of it.

    Returns a dict with:
        - department: which team should handle this
        - priority: "low" | "normal" | "high"
        - requires_human: bool
        - escalation_reason: str | None (why requires_human is True)
    """

    department = DEPARTMENT_MAP.get(intent, DEFAULT_DEPARTMENT)
    priority = base_priority(intent)

    # Default: follow the model's confidence-based decision.
    requires_human = (status == "REVIEW")
    escalation_reason = "low_model_confidence" if requires_human else None

    # Hard override: certain language always requires a human, no matter
    # how confident the model was about the intent.
    keyword = matched_escalation_keyword(text)

    if keyword is not None:
        requires_human = True
        priority = "high"
        escalation_reason = f"escalation_keyword:{keyword}"

    return {
        "department": department,
        "priority": priority,
        "requires_human": requires_human,
        "escalation_reason": escalation_reason,
    }

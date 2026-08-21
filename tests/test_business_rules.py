import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from business_rules import apply_business_rules, base_priority, matched_escalation_keyword


def test_routine_intent_routes_to_correct_department():
    result = apply_business_rules(
        text="I forgot my password",
        intent="recover_password",
        confidence=0.97,
        status="AUTO",
    )

    assert result["department"] == "account_support"
    assert result["requires_human"] is False
    assert result["escalation_reason"] is None


def test_low_confidence_requires_human():
    result = apply_business_rules(
        text="Something about my order",
        intent="track_order",
        confidence=0.55,
        status="REVIEW",
    )

    assert result["requires_human"] is True
    assert result["escalation_reason"] == "low_model_confidence"


def test_fraud_keyword_forces_escalation_even_with_high_confidence():
    result = apply_business_rules(
        text="Someone made an unauthorized charge on my card",
        intent="payment_issue",
        confidence=0.99,
        status="AUTO",
    )

    assert result["requires_human"] is True
    assert result["priority"] == "high"
    assert "escalation_keyword" in result["escalation_reason"]


def test_high_priority_intent_defaults_to_high():
    assert base_priority("payment_issue") == "high"


def test_low_priority_intent_defaults_to_low():
    assert base_priority("newsletter_subscription") == "low"


def test_normal_intent_defaults_to_normal():
    assert base_priority("track_order") == "normal"


def test_keyword_match_is_case_insensitive():
    assert matched_escalation_keyword("I think my account was HACKED") == "hacked"


def test_no_keyword_match_returns_none():
    assert matched_escalation_keyword("I want to track my order") is None


def test_unknown_intent_falls_back_to_general_support():
    result = apply_business_rules(
        text="Random message",
        intent="some_new_intent_not_in_map",
        confidence=0.95,
        status="AUTO",
    )

    assert result["department"] == "general_support"

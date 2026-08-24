Changelog



2026-08-24

\- Fixed non-deterministic escalation keyword matching in business\_rules.py (longest keyword now matches first, e.g. "hacked" over "hack")

\- Added API test suite (tests/test\_api.py) covering health check, happy path prediction, empty/whitespace input rejection, and fraud-keyword escalation

\- Added Phase 3 error analysis (notebooks/04\_error\_analysis.py): re-evaluated calibrated model on held-out test split, found 4 misclassifications out of 5375 examples, all genuinely ambiguous rather than systematic model errors

\- Updated README to reflect current state: business rules, error analysis, and full test coverage are done, removed outdated note claiming API tests require a live server



Earlier

\- Added business rules layer (src/business\_rules.py): department routing, priority assignment, keyword-based escalation for fraud/security terms

\- Wired business rules into predict.py and api.py

\- Added requirements.txt and initial README

\- Added calibrated classifier, threshold analysis, confidence evaluation, and initial test suite

\- Completed EDA and baseline model comparison

\- Initial project setup


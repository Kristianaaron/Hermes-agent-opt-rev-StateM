import importlib.util
from pathlib import Path

_PLUGIN = Path(__file__).resolve().parent / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("test_gliner_extract_plugin", _PLUGIN)
gliner_extract = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(gliner_extract)


def _fake_extract(entities, text="do the task"):
    class FakeModel:
        def predict_entities(self, prompt, labels, threshold):
            return entities

    gliner_extract._CACHE.clear()
    original = gliner_extract._load_model
    gliner_extract._load_model = lambda: FakeModel()
    try:
        return gliner_extract._extract(text)
    finally:
        gliner_extract._load_model = original


def test_all_bounded_prompts_are_semantically_probed():
    assert gliner_extract._should_probe("Please initiate the development environment")
    assert gliner_extract._should_probe("What is host binding?")
    assert not gliner_extract._should_probe("")


def test_target_evidence_selects_fast_lane_without_command_keyword():
    evidence = _fake_extract(
        [{"label": "software target", "text": "development environment", "score": 0.82}],
        "Please initiate the development environment",
    )
    assert evidence["decision"] == "fast"
    assert evidence["category"] == "bounded_operation"


def test_semantic_mutation_evidence_handles_unlisted_edit_language():
    evidence = _fake_extract(
        [{"label": "requested code or file modification", "text": "Decrease weights", "score": 0.91}],
        "Decrease the main step header weights",
    )
    assert evidence["mutation_intent"] is True
    assert evidence["mutation_score"] == 0.91


def test_semantic_inspection_evidence_keeps_review_read_only():
    evidence = _fake_extract(
        [{"label": "request to inspect without modifying", "text": "review only", "score": 0.90}],
        "Review only; do not implement",
    )
    assert evidence["mutation_intent"] is False
    assert evidence["inspection_score"] == 0.90


def test_complex_evidence_selects_full_lane():
    evidence = _fake_extract(
        [{"label": "root cause investigation", "text": "underlying cause", "score": 0.78}],
        "Find the underlying cause of the intermittent failure",
    )
    assert evidence["decision"] == "full"
    assert evidence["complex"] is True


def test_timeout_falls_back(monkeypatch):
    monkeypatch.setattr(gliner_extract, "_enabled", lambda: True)
    monkeypatch.setattr(gliner_extract, "_signal", lambda text: None)
    result = gliner_extract.on_llm_request(
        request={
            "model": "GLM-5.3-Flash-EXL3",
            "messages": [{"role": "user", "content": "initiate the development environment"}],
        },
        model="GLM-5.3-Flash-EXL3",
    )
    assert result is None


def test_evidence_is_private_middleware_state(monkeypatch):
    monkeypatch.setattr(gliner_extract, "_enabled", lambda: True)
    monkeypatch.setattr(
        gliner_extract,
        "_signal",
        lambda text: {
            "decision": "fast",
            "category": "bounded_operation",
            "confidence": 0.9,
            "latency_ms": 2.0,
            "cached": False,
        },
    )
    request = {
        "model": "GLM-5.3-Flash-EXL3",
        "messages": [{"role": "user", "content": "initiate the development environment"}],
    }
    result = gliner_extract.on_llm_request(request=request, model=request["model"])
    assert result["request"] == request
    assert gliner_extract.consume_evidence()["decision"] == "fast"


def test_irreversible_language_never_routes_fast():
    evidence = _fake_extract(
        [{"label": "software target", "text": "production database", "score": 0.9}],
        "delete production database",
    )
    assert evidence["decision"] == "full"
    assert evidence["destructive"] is True


def test_url_false_positive_does_not_become_secret_risk():
    evidence = _fake_extract(
        [
            {"label": "software target", "text": "development server", "score": 0.74},
            {"label": "password API key or private credential", "text": "URL", "score": 0.41},
        ],
        "Start the development server and tell me the URL",
    )
    assert evidence["decision"] == "fast"
    assert evidence["destructive"] is False


def test_real_secret_evidence_routes_full():
    evidence = _fake_extract(
        [{"label": "password API key or private credential", "text": "API key", "score": 0.84}],
        "Rotate the leaked API key",
    )
    assert evidence["decision"] == "full"
    assert evidence["destructive"] is True


def test_security_audit_and_database_migration_route_full():
    for text in (
        "Audit this authentication flow for security vulnerabilities",
        "Add OAuth with database migrations",
    ):
        evidence = _fake_extract([], text)
        assert evidence["decision"] == "full"
        assert evidence["complex"] is True


def test_strong_cross_turn_reference_is_ambiguous_even_with_target():
    evidence = _fake_extract(
        [{"label": "software target", "text": "mobile", "score": 0.86}],
        "Do the same for mobile",
    )
    assert evidence["decision"] == "unknown"
    assert evidence["unresolved_reference"] is True


def test_social_greeting_routes_to_tool_free_quick_response():
    evidence = _fake_extract([], "hey there again")
    assert evidence["decision"] == "fast"
    assert evidence["category"] == "quick_response"


def test_explanation_routes_to_quick_response_not_action_tools():
    evidence = _fake_extract(
        [
            {"label": "request for explanation", "text": "what is", "score": 0.81},
            {"label": "software target", "text": "Vite host binding", "score": 0.76},
        ],
        "What is Vite host binding?",
    )
    assert evidence["decision"] == "fast"
    assert evidence["category"] == "quick_response"


def test_synthetic_recovery_nudge_does_not_replace_user_prompt():
    messages = [
        {"role": "user", "content": "Start the server"},
        {"role": "user", "content": "Continue processing", "_empty_recovery_synthetic": True},
    ]
    assert gliner_extract._latest_user_text(messages) == "Start the server"


def test_user_text_beginning_with_system_label_is_still_user_text():
    prompt = "[System: Please fix the draft release note]"
    assert gliner_extract._latest_user_text([{"role": "user", "content": prompt}]) == prompt


def test_latest_user_text_excludes_expanded_attached_url_context():
    instruction = "Change the PDP images using the referenced page"
    messages = [{
        "role": "user",
        "content": instruction + "\n\n--- Attached Context ---\n\n" + ("page text " * 2000),
    }]

    routed = gliner_extract._latest_user_text(messages)

    assert routed == instruction
    assert gliner_extract._should_probe(routed)


def test_auto_continue_display_row_does_not_replace_user_prompt():
    messages = [
        {"role": "user", "content": "Add the two data options"},
        {
            "role": "user",
            "content": "[System note: Your previous turn was interrupted mid-run — resume]",
            "display_kind": "auto_continue",
        },
    ]
    assert gliner_extract._latest_user_text(messages) == "Add the two data options"


def test_general_prose_grammar_routes_without_tools_even_if_model_labels_target():
    evidence = _fake_extract(
        [{"label": "software target", "text": "host binding", "score": 0.88}],
        "What is host binding?",
    )
    assert evidence["decision"] == "fast"
    assert evidence["category"] == "quick_response"


def test_general_action_grammar_requires_tools_when_extractor_is_uncertain():
    for text in (
        "Change the button label to Continue",
        "Read package.json and tell me the dev script",
    ):
        evidence = _fake_extract([], text)
        assert evidence["decision"] == "fast"
        assert evidence["category"] == "bounded_operation"

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
    # Hermes keeps the last middleware payload; echoing the request would erase
    # the router's lane decision when this plugin loads after it.
    assert result is None
    assert gliner_extract.consume_evidence()["decision"] == "fast"


def test_classify_is_order_independent_direct_entry(monkeypatch):
    monkeypatch.setattr(gliner_extract, "_enabled", lambda: True)
    seen = []
    monkeypatch.setattr(
        gliner_extract, "_signal",
        lambda text: seen.append(text) or {"decision": "fast", "category": "quick_response",
                                           "confidence": 0.9, "latency_ms": 1.0, "cached": False},
    )
    evidence = gliner_extract.classify("Fix the header\n\n--- Attached Context ---\n" + "x" * 5000)
    assert evidence["decision"] == "fast"
    assert seen == ["Fix the header"]


def test_cached_text_skips_worker_and_busy_window(monkeypatch):
    gliner_extract._CACHE.clear()
    evidence = {"decision": "fast", "category": "bounded_operation", "confidence": 0.8,
                "latency_ms": 3.0, "cached": False}
    gliner_extract._cache_put("start the server", evidence)
    monkeypatch.setattr(gliner_extract, "_BUSY_UNTIL", gliner_extract.time.monotonic() + 60)
    hit = gliner_extract._signal("start the server")
    assert hit["decision"] == "fast" and hit["cached"] is True


def test_inflight_prediction_is_not_queued_behind(monkeypatch):
    gliner_extract._CACHE.clear()

    class Pending:
        def done(self):
            return False

    monkeypatch.setattr(gliner_extract, "_INFLIGHT", Pending())
    monkeypatch.setattr(gliner_extract, "_BUSY_UNTIL", 0.0)
    monkeypatch.setattr(gliner_extract, "_LOAD_FAILED_UNTIL", 0.0)
    submitted = []
    monkeypatch.setattr(gliner_extract._EXECUTOR, "submit", lambda *a: submitted.append(a))
    assert gliner_extract._signal("a new uncached request") is None
    assert submitted == []


def test_load_failure_backs_off_instead_of_retrying_every_request(monkeypatch):
    gliner_extract._CACHE.clear()
    monkeypatch.setattr(gliner_extract, "_MODEL", None)
    monkeypatch.setattr(gliner_extract, "_LOAD_FAILED_UNTIL", 0.0)
    monkeypatch.setitem(gliner_extract.sys.modules, "gliner", None)  # ImportError
    try:
        gliner_extract._load_model()
    except ImportError:
        pass
    assert gliner_extract._LOAD_FAILED_UNTIL > gliner_extract.time.monotonic()
    monkeypatch.setattr(gliner_extract, "_INFLIGHT", None)
    monkeypatch.setattr(gliner_extract, "_BUSY_UNTIL", 0.0)
    submitted = []
    monkeypatch.setattr(gliner_extract._EXECUTOR, "submit", lambda *a: submitted.append(a))
    assert gliner_extract._signal("another uncached request") is None
    assert submitted == []


def test_middleware_accepts_glm53_model_alias(monkeypatch):
    monkeypatch.setattr(gliner_extract, "_enabled", lambda: True)
    monkeypatch.setattr(gliner_extract, "_signal", lambda text: {
        "decision": "fast", "category": "quick_response", "confidence": 0.9,
        "latency_ms": 1.0, "cached": True,
    })
    request = {"model": "glm53-flash", "messages": [{"role": "user", "content": "hello"}]}
    assert gliner_extract.on_llm_request(request=request, model="glm53-flash") is None
    assert gliner_extract.consume_evidence()["category"] == "quick_response"


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

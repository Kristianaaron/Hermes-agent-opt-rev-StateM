from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _configured_timeout(provider_id: str, model: str | None, model_key: str, provider_key: str) -> float | None:
    """Per-model ``providers.<id>.models.<model>.<model_key>`` wins over ``providers.<id>.<provider_key>``.

    After named custom providers collapse to ``custom``, resolve by unique model
    membership so the named provider's timeout is not silently lost. Ambiguity
    fails closed rather than borrowing a different endpoint's policy.
    """
    if not provider_id:
        return None
    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    provider_config = _resolve_provider_config(providers, provider_id, model)
    if provider_config is None:
        return None
    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get(model_key))
        if timeout is not None:
            return timeout
    return _coerce_timeout(provider_config.get(provider_key))


def _resolve_provider_config(
    providers: object,
    provider_id: str,
    model: str | None,
) -> dict[str, object] | None:
    """Resolve timeout config after named custom providers become ``custom``."""
    if not isinstance(providers, dict):
        return None
    direct = providers.get(provider_id)
    if isinstance(direct, dict):
        return direct
    normalized = str(provider_id or "").strip().lower()
    if normalized != "custom" and not normalized.startswith("custom:"):
        return None
    if not model:
        return None
    matches: list[dict[str, object]] = []
    for candidate in providers.values():
        if not isinstance(candidate, dict):
            continue
        models = candidate.get("models")
        contains_model = isinstance(models, dict) and model in models
        default_model = str(candidate.get("model") or "") == model
        if contains_model or default_model:
            matches.append(candidate)
    return matches[0] if len(matches) == 1 else None


def get_provider_request_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured provider request timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "timeout_seconds", "request_timeout_seconds")


def get_provider_stale_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any."""
    return _configured_timeout(provider_id, model, "stale_timeout_seconds", "stale_timeout_seconds")


def get_provider_hard_timeout(provider_id: str, model: str | None = None) -> float | None:
    """Return a wall-clock request cap in seconds, if any.

    Distinct from ``stale_timeout_seconds`` (quiet-stream gap) and
    ``timeout_seconds`` (HTTP read-between-chunks). A busy think-loop keeps
    both of those alive forever; this one does not.
    """
    return _configured_timeout(provider_id, model, "hard_timeout_seconds", "hard_timeout_seconds")


def _get_model_config(provider_config: dict[str, object], model: str | None) -> dict[str, object] | None:
    if not model:
        return None
    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    return model_config if isinstance(model_config, dict) else None

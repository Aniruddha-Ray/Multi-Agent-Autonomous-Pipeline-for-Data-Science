def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def safe_get(d: dict, *keys, default=None):
    """Nested dict.get() that never raises on a missing intermediate key."""
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d
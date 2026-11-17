"""Optional, idempotent Logfire setup and privacy-conscious spans."""
from __future__ import annotations
import os
from contextlib import contextmanager
from typing import Any, Iterator

try:
    import logfire as _logfire
except ImportError:
    _logfire = None

_configured = False

def configure() -> None:
    """Configure Logfire once; never fail if the optional SDK is absent."""
    global _configured
    if _configured or _logfire is None:
        return
    try:
        _logfire.configure(send_to_logfire="if-token-present", token=os.getenv("LOGFIRE_TOKEN"))
    except TypeError:
        try:
            _logfire.configure(send_to_logfire="if-token-present")
        except Exception:
            return
    except Exception:
        return
    _configured = True

configure_logfire = configure

@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Use a Logfire span when possible, else yield as a safe no-op."""
    configure()
    if _logfire is None:
        yield None
        return
    try:
        manager = _logfire.span(name, **attributes)
        entered = manager.__enter__()
    except Exception:
        yield None
        return
    try:
        yield entered
    except BaseException as exc:
        manager.__exit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        manager.__exit__(None, None, None)

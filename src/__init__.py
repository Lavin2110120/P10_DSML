"""Package src — applique le bypass SSL AVANT tout client HTTP si DISABLE_SSL_VERIFY est actif."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

if os.getenv("DISABLE_SSL_VERIFY", "0").lower() in ("1", "true", "yes", "on"):
    from .ssl_bypass import apply_ssl_bypass
    apply_ssl_bypass()
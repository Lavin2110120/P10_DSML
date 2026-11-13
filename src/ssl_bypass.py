"""Désactivation globale des vérifications SSL pour environnements avec proxy/inspection TLS."""

import os
import ssl
import warnings


def apply_ssl_bypass():
    """Applique le bypass SSL sur standard library, urllib3, requests, httpx et grpc."""
    # Désactivation active si DISABLE_SSL_VERIFY=1 ou par défaut si appelée explicitement
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Contexte SSL standard
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

    # 2. urllib3 / requests
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except ImportError:
        pass

    # 3. Variables d'environnement pour gRPC, curl, certifi
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = ""
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""


def disable_ssl_verification():
    """Alias pour compatibilité avec data_loader.py et les anciens scripts."""
    apply_ssl_bypass()


# Application automatique si la variable d'environnement est activée
if os.getenv("DISABLE_SSL_VERIFY", "1").lower() in ("1", "true", "yes"):
    apply_ssl_bypass()
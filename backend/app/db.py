"""
MOIL Manganese Mine Digital Twin - Supabase Database Client
Modular database client supporting both `supabase-py` and direct PostgREST REST fallback.
Loads credentials from backend/.env or system environment.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("supabase_client")

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    ]
    for p in env_paths:
        if os.path.exists(p):
            load_dotenv(p)
            break
except ImportError:
    # Manual lightweight .env loader fallback
    def _load_env_file(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ:
                            os.environ[k] = v
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    ]
    for p in env_paths:
        _load_env_file(p)

# Environment configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://dkmxjeuoidbxnpwrjadv.supabase.co").rstrip("/")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_KEY")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_SECRET_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")
    or ""
)
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "sb_publishable_Y_MAxfihttak5yXTLta95w_0QgL7")

# Check if supabase package is installed
SUPABASE_SDK_AVAILABLE = False
_supabase_client = None

try:
    from supabase import create_client, Client
    SUPABASE_SDK_AVAILABLE = True
except ImportError:
    Client = Any

class SupabaseDatabase:
    """
    Modular Supabase client wrapper.
    Provides standard ORM methods and falls back to PostgREST HTTP REST API if supabase-py is not installed.
    """
    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        self.url = url
        self.key = key
        self.client: Optional[Client] = None
        self._init_client()

    def _init_client(self):
        if SUPABASE_SDK_AVAILABLE:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Initialized official Supabase Python SDK client.")
            except Exception as e:
                logger.warning(f"Failed to initialize supabase-py client: {e}. Falling back to REST adapter.")
                self.client = None
        else:
            logger.info("supabase-py not found in environment; using built-in PostgREST REST client.")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }

    def fetch_all(self, table: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch records from a given Supabase table."""
        if self.client:
            try:
                res = self.client.table(table).select("*").limit(limit).execute()
                return res.data or []
            except Exception as e:
                logger.error(f"Error fetching from {table} via SDK: {e}")

        # REST fallback via requests
        try:
            import requests
            endpoint = f"{self.url}/rest/v1/{table}?select=*&limit={limit}"
            resp = requests.get(endpoint, headers=self._get_headers(), timeout=6)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"REST fetch from {table} returned {resp.status_code}: {resp.text}")
                return []
        except Exception as e:
            logger.error(f"REST fetch exception on {table}: {e}")
            return []

    def insert(self, table: str, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a single record into a given Supabase table."""
        if self.client:
            try:
                res = self.client.table(table).insert(record).execute()
                return res.data[0] if res.data else record
            except Exception as e:
                logger.error(f"Error inserting into {table} via SDK: {e}")

        # REST fallback
        try:
            import requests
            endpoint = f"{self.url}/rest/v1/{table}"
            resp = requests.post(endpoint, headers=self._get_headers(), json=record, timeout=6)
            if resp.status_code in (200, 201):
                data = resp.json()
                return data[0] if isinstance(data, list) and len(data) > 0 else record
            else:
                logger.warning(f"REST insert into {table} failed ({resp.status_code}): {resp.text}")
                return None
        except Exception as e:
            logger.error(f"REST insert exception on {table}: {e}")
            return None

    def health_check(self) -> Dict[str, Any]:
        """Verify connectivity to the Supabase endpoint."""
        try:
            import requests
            resp = requests.get(f"{self.url}/rest/v1/", headers=self._get_headers(), timeout=5)
            return {
                "connected": resp.status_code in (200, 404, 401),
                "status_code": resp.status_code,
                "url": self.url,
                "sdk_used": SUPABASE_SDK_AVAILABLE and self.client is not None
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "url": self.url,
                "sdk_used": False
            }

# Global singleton client
db = SupabaseDatabase()

def get_supabase_client() -> SupabaseDatabase:
    """Accessor for global Supabase client instance."""
    return db

if __name__ == "__main__":
    print(f"Supabase URL: {SUPABASE_URL}")
    print(f"Supabase Key: {SUPABASE_KEY[:15]}...")
    status = db.health_check()
    print("Health check result:", json.dumps(status, indent=2))

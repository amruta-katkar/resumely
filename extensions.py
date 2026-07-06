import os
import secrets
import logging
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2 import pool

import firebase_admin
from firebase_admin import credentials, auth as firebase_auth

logger = logging.getLogger(__name__)


class Database:
    """Thin wrapper around a psycopg2 connection pool (instead of opening
    a brand new TCP connection to Postgres on every single request)."""

    def __init__(self, dsn, minconn=1, maxconn=10):
        self._pool = pool.SimpleConnectionPool(minconn, maxconn, dsn)

    @contextmanager
    def cursor(self, commit=False):
        conn = self._pool.getconn()
        try:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            self._pool.putconn(conn)

    def init_schema(self):
        with self.cursor(commit=True) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS resumes (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id TEXT NOT NULL,
                    input_data JSONB,
                    output_data JSONB,
                    ats_score INTEGER DEFAULT 0,
                    job_category TEXT DEFAULT 'General',
                    created_at TIMESTAMP DEFAULT NOW()
                )
                """
            )


class FirebaseService:
    """Owns Firebase Admin initialization and token verification.

    The previous code created a `credentials.Certificate(...)` object but
    never called `firebase_admin.initialize_app(cred)` — every call to
    `auth.verify_id_token()` would raise ValueError('The default Firebase
    app does not exist') and get swallowed by a bare except, making login
    silently fail. This class makes sure the app is actually initialized
    exactly once.
    """

    def __init__(self, cred_path):
        self._enabled = False
        if not cred_path or not os.path.exists(cred_path):
            logger.warning(
                "Firebase credential file not found at %s — auth is disabled.",
                cred_path,
            )
            return
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        self._enabled = True

    def verify_id_token(self, id_token):
        if not self._enabled:
            raise RuntimeError("Firebase Admin is not configured on this server.")
        # check_revoked=True forces a lookup against Firebase so a token
        # from an already-logged-out / disabled account is rejected.
        return firebase_auth.verify_id_token(id_token, check_revoked=True)


class CSRFProtect:
    """Minimal double-submit-cookie-free CSRF protection using the
    session. Good enough for a single-page-per-form app without pulling
    in a new dependency."""

    SESSION_KEY = "_csrf_token"

    def generate_token(self, session):
        if self.SESSION_KEY not in session:
            session[self.SESSION_KEY] = secrets.token_hex(32)
        return session[self.SESSION_KEY]

    def validate(self, session, submitted_token):
        real_token = session.get(self.SESSION_KEY)
        return bool(real_token) and bool(submitted_token) and secrets.compare_digest(
            real_token, submitted_token
        )

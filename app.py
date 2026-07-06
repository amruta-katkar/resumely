import os
import time
import logging
import threading
import urllib.request
from datetime import timedelta

from flask import Flask
from dotenv import load_dotenv

from config import Config
from extensions import Database, FirebaseService, CSRFProtect
from resume.services import (
    GeminiClient, PdfTextExtractor, TemplateManager, PdfBuilder,
    LatexBuilder, ResumeRepository, RateLimiter,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def start_keep_alive(app_url):
    """Pings /ping every 14 minutes so a free-tier host doesn't spin the
    app down. This was previously defined but never actually called from
    anywhere, so it did nothing."""
    if not app_url:
        logger.info("APP_URL not set — keep-alive disabled.")
        return

    def ping():
        while True:
            time.sleep(14 * 60)
            try:
                urllib.request.urlopen(f"{app_url}/ping", timeout=10)
                logger.info("Keep-alive ping sent.")
            except Exception as e:
                logger.warning("Keep-alive failed: %s", e)

    threading.Thread(target=ping, daemon=True).start()


def create_app():
    cfg = Config()

    app = Flask(__name__)
    app.secret_key = cfg.SECRET_KEY
    app.permanent_session_lifetime = timedelta(days=30)

    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=cfg.IS_PRODUCTION,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=cfg.MAX_CONTENT_LENGTH,
        MAX_PDF_SIZE=cfg.MAX_PDF_SIZE,
        FIREBASE_CONFIG=cfg.FIREBASE_CONFIG,
        IS_PRODUCTION=cfg.IS_PRODUCTION,
    )

    # ── Services, built once and shared across requests ──────────
    db = Database(cfg.DATABASE_URL)
    db.init_schema()

    app.extensions["db"] = db
    app.extensions["repo"] = ResumeRepository(db)
    app.extensions["firebase"] = FirebaseService(cfg.FIREBASE_CRED_PATH)
    app.extensions["gemini"] = GeminiClient(cfg.GEMINI_API_KEY)
    app.extensions["pdf_extractor"] = PdfTextExtractor()
    app.extensions["templates"] = TemplateManager()
    app.extensions["pdf_builder"] = PdfBuilder(app.extensions["templates"])
    app.extensions["latex_builder"] = LatexBuilder()
    app.extensions["rate_limiter"] = RateLimiter()
    app.extensions["csrf"] = CSRFProtect()

    # ── Blueprints ─────────────────────────────────────────────
    from auth.routes import auth_bp
    from resume.routes import resume_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(resume_bp)

    @app.context_processor
    def inject_globals():
        import datetime
        return {"current_year": datetime.datetime.now(datetime.timezone.utc).year}


    if cfg.IS_PRODUCTION:
        start_keep_alive(cfg.APP_URL)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=not app.config["IS_PRODUCTION"])

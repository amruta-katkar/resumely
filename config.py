import os


class Config:
    """
    All runtime configuration lives here and is pulled from environment
    variables only. Nothing here is hardcoded, so no secret ever ships
    inside source control.
    """

    def __init__(self):
        self.FLASK_ENV = os.getenv("FLASK_ENV", "development")
        self.IS_PRODUCTION = self.FLASK_ENV == "production"

        self.SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.APP_URL = os.getenv("APP_URL", "")

        self.FIREBASE_CRED_PATH = os.getenv(
            "FIREBASE_CRED_PATH", "firebase-adminsdk.json"
        )
        self.FIREBASE_CONFIG = {
            "apiKey": os.getenv("FIREBASE_API_KEY"),
            "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
            "projectId": os.getenv("FIREBASE_PROJECT_ID"),
            "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
            "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
            "appId": os.getenv("FIREBASE_APP_ID"),
            "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID"),
        }

        self.MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB, single source of truth
        self.MAX_PDF_SIZE = 5 * 1024 * 1024

        self._validate()

    def _validate(self):
        missing = []
        if not self.SECRET_KEY:
            missing.append("FLASK_SECRET_KEY")
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        if self.IS_PRODUCTION and missing:
            # Refuse to boot in production with missing secrets instead of
            # silently falling back to an insecure default.
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        if not self.IS_PRODUCTION and not self.SECRET_KEY:
            # dev convenience only — never reachable in production because
            # of the check above.
            self.SECRET_KEY = "dev-only-insecure-key-do-not-deploy"

from flask import Flask
from datetime import timedelta

from auth.routes import auth_bp
from resume.routes import resume_bp
from dotenv import load_dotenv
import os
load_dotenv()

app = Flask(__name__)

app.secret_key = "secret"

app.permanent_session_lifetime = timedelta(days=30)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production",
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
)

# CONFIGS HERE
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# BLUEPRINTS
app.register_blueprint(auth_bp)
app.register_blueprint(resume_bp)

if __name__ == "__main__":
    app.run(debug=True)

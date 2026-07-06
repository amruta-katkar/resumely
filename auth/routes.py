import logging
from functools import wraps

from flask import (
    Blueprint, render_template, request, jsonify, session,
    redirect, url_for, flash, current_app
)

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)


class AuthService:
    """Wraps Firebase token verification + Flask session creation."""

    def __init__(self, firebase_service):
        self.firebase = firebase_service

    def login_with_id_token(self, id_token, remember_me=False):
        decoded = self.firebase.verify_id_token(id_token)
        email = decoded.get("email")
        uid = decoded.get("uid")
        if not email or not decoded.get("email_verified", True):
            # Firebase email/password + Google both set email_verified;
            # for providers where it's absent we don't block, but if it's
            # explicitly False we reject.
            if decoded.get("email_verified") is False:
                raise PermissionError("Please verify your email before logging in.")
        session.clear()
        session.permanent = bool(remember_me)
        session["user"] = email
        session["uid"] = uid
        return email


def get_auth_service():
    return AuthService(current_app.extensions["firebase"])


@auth_bp.after_request
def add_security_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("auth.login"))
        return f(*args, **kwargs)
    return decorated_function


@auth_bp.route("/create-session", methods=["POST"])
def create_session():
    data = request.get_json(silent=True) or {}
    id_token = data.get("idToken")
    remember_me = data.get("rememberMe", False)

    if not id_token:
        return jsonify({"success": False, "message": "Missing token."}), 400

    try:
        service = get_auth_service()
        service.login_with_id_token(id_token, remember_me)
        return jsonify({"success": True})
    except PermissionError as e:
        return jsonify({"success": False, "message": str(e)}), 403
    except Exception as e:
        # Log the real error server-side only; never echo internal
        # exception details (which can leak project/config info) to the client.
        logger.warning("Login failed: %s", e)
        return jsonify({"success": False, "message": "Invalid credentials."}), 401


@auth_bp.route("/")
def home():
    return render_template("auth/index.html")


@auth_bp.route("/login")
def login():
    if "user" in session:
        return redirect(url_for("resume.dashboard"))
    return render_template(
        "auth/login.html", firebase_config=current_app.config["FIREBASE_CONFIG"]
    )


@auth_bp.route("/signup")
def signup():
    if "user" in session:
        return redirect(url_for("resume.dashboard"))
    return render_template(
        "auth/signup.html", firebase_config=current_app.config["FIREBASE_CONFIG"]
    )


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    response = redirect(url_for("auth.login"))
    response.delete_cookie("session")
    return response

from flask import render_template
from flask import request
from flask import jsonify
from flask import session
from flask import redirect
from functools import wraps
from flask import flash
from dotenv import load_dotenv
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth
from firebase_config import *
from flask import Blueprint, url_for

auth_bp = Blueprint('auth', __name__)
load_dotenv()

cred = credentials.Certificate(
    "firebase-adminsdk.json"
)
# firebase_admin.initialize_app(cred)

firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}

@auth_bp.route("/")
def home():
    return render_template("auth/index.html")

@auth_bp.after_request
def add_header(response):

    response.headers["Cache-Control"] = \
        "no-cache, no-store, must-revalidate"

    response.headers["Pragma"] = "no-cache"

    response.headers["Expires"] = "0"

    return response


@auth_bp.route("/login")
def login():

    if "user" in session:
        return redirect(url_for('auth.dashboard'))
    
    return render_template("auth/login.html", firebase_config=firebase_config)
def login_required(f):

    @wraps(f)

    def decorated_function(*args, **kwargs):

        if "user" not in session:
            return redirect(url_for('auth.login'))

        return f(*args, **kwargs)

    return decorated_function

@auth_bp.route("/signup")
def signup():

    if "user" in session:
        return redirect(url_for('resume.dashboard'))

    return render_template("auth/signup.html", firebase_config=firebase_config)


@auth_bp.route("/create-session", methods=["POST"])
def create_session():

    try:

        data = request.get_json()

        id_token = data["idToken"]

        remember_me = data.get("rememberMe", False)

        decoded_token = auth.verify_id_token(id_token)

        email = decoded_token["email"]

        # REMEMBER ME LOGIC

        session.permanent = remember_me

        session["user"] = email

        return jsonify({
            "success": True
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 401

@auth_bp.route("/dashboard")
@login_required
def dashboard():
    flash("Login successful")
    return render_template(
    "resume/dashboard.html",
    user=session["user"],
    firebase_config=firebase_config
)


@auth_bp.route("/logout")
def logout():

    session.clear()
    flash("Logged out successfully")
    return redirect("/login")

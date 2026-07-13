import os
import io
import uuid
import logging
from functools import wraps

from flask import (
    Blueprint, request, render_template, session, redirect, url_for,
    jsonify, send_file, current_app
)

from auth.routes import login_required
from resume.services import ResumeAI

logger = logging.getLogger(__name__)
resume_bp = Blueprint("resume", __name__)

LIMITS = {"skills": 3000, "projects": 5000, "experience": 5000, "job_description": 5000,
          "name": 120, "email": 254, "phone": 40, "location": 120,
          "linkedin": 300, "github": 300, "portfolio": 300, "hobbies": 500,
          "custom_section_title": 80, "custom_section_content": 3000}


def validate(**fields):
    for name, val in fields.items():
        if not val or not val.strip():
            return f"'{name}' is required."
        if len(val) > LIMITS.get(name, 3000):
            return f"'{name}' is too long (max {LIMITS.get(name, 3000)} chars)."
    return None


def validate_optional(**fields):
    for name, val in fields.items():
        if val and len(val) > LIMITS.get(name, 500):
            return f"'{name}' is too long (max {LIMITS.get(name, 500)} chars)."
    return None


def ext():
    """Shorthand accessor for the services registered on the app."""
    return current_app.extensions


def rate_limit(max_calls=5, window=60):
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            uid = session.get("user_id", request.remote_addr)
            key = f"{uid}:{f.__name__}"
            if not ext()["rate_limiter"].allow(key, max_calls, window):
                return render_template("resume/error.html", message="Too many requests. Wait a minute."), 429
            return f(*args, **kwargs)
        return wrapper
    return dec


def rate_limit_json(max_calls=10, window=60):
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            uid = session.get("user_id", request.remote_addr)
            key = f"{uid}:{f.__name__}"
            if not ext()["rate_limiter"].allow(key, max_calls, window):
                return jsonify({"error": "Too many requests. Please wait a minute."}), 429
            return f(*args, **kwargs)
        return wrapper
    return dec


def csrf_required(f):
    """Validates a CSRF token for state-changing requests. Forms send it
    as a hidden field; fetch()-based JSON calls send it as a header."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not ext()["csrf"].validate(session, token):
            return jsonify({"error": "Invalid or missing CSRF token."}), 400
        return f(*args, **kwargs)
    return wrapper


@resume_bp.before_request
def ensure_session():
    if "user" in session:
        session["user_id"] = session["user"]


def firebase_ctx():
    return current_app.config["FIREBASE_CONFIG"]


@resume_bp.route("/ping")
def ping():
    return "ok", 200


@resume_bp.route("/parse-pdf", methods=["POST"])
@login_required
@rate_limit_json(10, 60)
def parse_pdf_route():
    if "resume_pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["resume_pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files accepted"}), 400

    data = f.read()
    if len(data) > current_app.config["MAX_PDF_SIZE"]:
        return jsonify({"error": "File too large (max 5MB)"}), 400

    text, err = ext()["pdf_extractor"].extract(data)
    if err:
        return jsonify({"error": err}), 422

    ai = ResumeAI(ext()["gemini"])
    fields, err = ai.parse_pdf_into_form_fields(text)
    if err:
        return jsonify({"error": err}), 502

    # This is the shape analysis.html's JS actually reads (skills/projects/
    # experience/name/email/...). The previous endpoint only returned
    # {"text": text}, which is why "PDF extraction" looked broken — the
    # frontend was reading fields that never existed in the response.
    return jsonify(fields)


@resume_bp.route("/generate", methods=["POST"])
@login_required
@rate_limit(5, 60)
@csrf_required
def generate():
    form = request.form
    skills = form.get("skills", "").strip()
    projects = form.get("projects", "").strip()
    experience = form.get("experience", "").strip()
    job_desc = form.get("job_description", "").strip()
    template_id = form.get("template_id", "classic")

    contact = {
        "name": form.get("name", "").strip(),
        "email": form.get("email", "").strip(),
        "phone": form.get("phone", "").strip(),
        "location": form.get("location", "").strip(),
        "linkedin": form.get("linkedin", "").strip(),
        "github": form.get("github", "").strip(),
        "portfolio": form.get("portfolio", "").strip(),
        "hobbies": form.get("hobbies", "").strip(),
    }
    custom_section_title = form.get("custom_section_title", "").strip()
    custom_section_content = form.get("custom_section_content", "").strip()

    extra_skills_raw = form.get("extra_skills", "")
    extra_skills = [s.strip() for s in extra_skills_raw.split(",") if s.strip()] if extra_skills_raw else []

    err = validate(skills=skills, projects=projects, experience=experience, job_description=job_desc)
    if err:
        return render_template("resume/error.html", message=err), 400
    err = validate_optional(**contact, custom_section_title=custom_section_title,
                             custom_section_content=custom_section_content)
    if err:
        return render_template("resume/error.html", message=err), 400
    if template_id not in ext()["templates"].all():
        template_id = "classic"

    contact_block = "\n".join(f"{k.upper()}: {v}" for k, v in contact.items() if v)
    custom_block = f"\n{custom_section_title.upper()}:\n{custom_section_content}" if custom_section_title else ""
    raw = (f"{contact_block}\n\nSKILLS:\n{skills}\n\nPROJECTS:\n{projects}\n\n"
           f"EXPERIENCE:\n{experience}{custom_block}")

    ai = ResumeAI(ext()["gemini"])

    parsed_resume, err = ai.parse_resume(raw)
    if err:
        return render_template("resume/error.html", message=err), 500
    # The form is the source of truth for structured contact fields —
    # don't let the model's guess silently override what the user typed.
    for k, v in contact.items():
        if v:
            if k == "hobbies":
                parsed_resume["hobbies"] = [h.strip() for h in v.split(",") if h.strip()]
            else:
                parsed_resume[k] = v

    parsed_jd, err = ai.analyze_jd(job_desc)
    if err:
        return render_template("resume/error.html", message=err), 500

    ats, err = ai.score_ats(parsed_resume, parsed_jd)
    if err:
        logger.warning("ATS scoring failed: %s", err)
        ats = {"total_score": 0, "breakdown": {}, "matched_skills": [], "missing_skills": [],
               "missing_keywords": [], "you_have_but_not_listed": [], "recommendations": []}

    if template_id == "classic" and parsed_jd:
        template_id = ext()["templates"].recommend(parsed_jd.get("job_category", "General"))

    tailored, err = ai.tailor(parsed_resume, parsed_jd, ats, template_id, extra_skills)
    if err:
        return render_template("resume/error.html", message=err), 500

    latex_builder = ext()["latex_builder"]
    tpl_meta = ext()["templates"].get(template_id)
    result = {
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd,
        "ats_report": ats,
        "selected_projects": tailored.get("selected_projects", []),
        "tailored_experience": tailored.get("tailored_experience", []),
        "resume_text": tailored.get("resume_text", ""),
        "cover_letter": tailored.get("cover_letter", ""),
        "template_id": template_id,
        "latex": latex_builder.build(parsed_resume, tailored.get("tailored_experience"),
                                      tailored.get("selected_projects"),
                                      layout=tpl_meta["layout"], accent=tpl_meta["color"]),
    }

    result_id = str(uuid.uuid4())
    repo = ext()["repo"]
    try:
        repo.save(
            result_id, session["user_id"],
            {"skills": skills, "projects": projects, "experience": experience,
             "job_description": job_desc, "contact": contact},
            result, ats.get("total_score", 0),
            parsed_jd.get("job_category", "General") if parsed_jd else "General",
        )
    except Exception as e:
        logger.error("DB save failed: %s", e)
        return render_template("resume/error.html", message="Couldn't save your result. Please try again."), 500

    return redirect(url_for("resume.result_page", result_id=result_id))


@resume_bp.route("/result/<result_id>")
@login_required
def result_page(result_id):
    try:
        uuid.UUID(result_id)
    except ValueError:
        return render_template("resume/error.html", message="Invalid result id."), 400

    row = ext()["repo"].get(result_id)
    if not row:
        return render_template("resume/error.html", message="Result not found."), 404
    if row["session_id"] != session.get("user_id"):
        return render_template("resume/error.html", message="Access denied."), 403

    result = row["output_data"]
    return render_template(
        "resume/result.html", result=result, result_id=result_id,
        templates=ext()["templates"].all(), user=session["user"],
        csrf_token=ext()["csrf"].generate_token(session),
        firebase_config=firebase_ctx(),
    )


@resume_bp.route("/update-skills/<result_id>", methods=["POST"])
@login_required
@rate_limit_json(15, 60)
@csrf_required
def update_skills(result_id):
    """User confirms skills they have but weren't listed → bumps the score."""
    try:
        uuid.UUID(result_id)
    except ValueError:
        return jsonify({"error": "Invalid result id"}), 400

    payload = request.get_json(silent=True) or {}
    new_skills = [str(s).strip() for s in payload.get("skills", []) if str(s).strip()][:50]
    if not new_skills:
        return jsonify({"error": "No skills provided"}), 400

    repo = ext()["repo"]
    row = repo.get(result_id)
    # Ownership check: previously this compared row["session_id"] to
    # session.get("user_id") without requiring @login_required, so an
    # anonymous request with no session and a resume row saved with a
    # null session_id could match None == None and edit someone else's
    # score. @login_required above guarantees session["user_id"] is a
    # real, non-null value, so this comparison is now safe.
    if not row or row["session_id"] != session.get("user_id"):
        return jsonify({"error": "Not found"}), 404

    try:
        result = row["output_data"]
        existing = result.get("parsed_resume", {}).get("skills", [])
        result.setdefault("parsed_resume", {})["skills"] = list(set(existing + new_skills))

        ats = result.get("ats_report", {})
        ats["matched_skills"] = list(set(ats.get("matched_skills", []) + new_skills))
        ats["missing_skills"] = [s for s in ats.get("missing_skills", []) if s not in new_skills]
        ats["you_have_but_not_listed"] = [s for s in ats.get("you_have_but_not_listed", []) if s not in new_skills]

        bonus = min(len(new_skills) * 3, 15)
        ats["total_score"] = min(ats.get("total_score", 0) + bonus, 100)
        result["ats_report"] = ats

        repo.update_output(result_id, result, ats["total_score"])
        return jsonify({"success": True, "new_score": ats["total_score"], "updated_skills": result["parsed_resume"]["skills"]})
    except Exception as e:
        logger.error("Update skills failed: %s", e)
        return jsonify({"error": "Update failed. Please try again."}), 500


@resume_bp.route("/export/pdf/<result_id>")
@login_required
def export_pdf(result_id):
    row = ext()["repo"].get(result_id)
    if not row:
        return render_template("resume/error.html", message="Not found."), 404
    if row["session_id"] != session.get("user_id"):
        return render_template("resume/error.html", message="Unauthorized."), 403

    result = row["output_data"]
    template_id = result.get("template_id", "classic")
    pr = result.get("parsed_resume") or {}
    name = pr.get("name", "resume")
    contact_line = " · ".join(v for v in [pr.get("phone"), pr.get("email"), pr.get("location")] if v)

    buf, err = ext()["pdf_builder"].build(result, template_id, contact_line)
    if err:
        return render_template("resume/error.html", message=err), 500
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_") or "resume"
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                      download_name=f"{safe_name}_resume.pdf")


@resume_bp.route("/export/latex/<result_id>")
@login_required
def export_latex(result_id):
    row = ext()["repo"].get(result_id)
    if not row:
        return render_template("resume/error.html", message="Not found."), 404
    if row["session_id"] != session.get("user_id"):
        return render_template("resume/error.html", message="Unauthorized."), 403

    result = row["output_data"]
    latex = result.get("latex", "")
    name = (result.get("parsed_resume") or {}).get("name", "resume")
    safe_name = "".join(c for c in name if c.isalnum() or c in " _-").strip().replace(" ", "_") or "resume"
    return send_file(io.BytesIO(latex.encode()), mimetype="text/plain",
                      as_attachment=True, download_name=f"{safe_name}_resume.tex")


@resume_bp.errorhandler(404)
def e404(e):
    return render_template("resume/error.html", message="Page not found."), 404


@resume_bp.errorhandler(413)
def e413(e):
    return render_template("resume/error.html", message="File too large (max 5MB)."), 413


@resume_bp.errorhandler(500)
def e500(e):
    logger.error(e)
    return render_template("resume/error.html", message="Server error. Try again."), 500


@resume_bp.route("/saved-resumes")
@login_required
def saved_resumes():
    resumes = ext()["repo"].list_for_session(session["user_id"])
    return render_template(
        "resume/saved_resumes.html", user=session["user"], resumes=resumes,
        firebase_config=firebase_ctx(), csrf_token=ext()["csrf"].generate_token(session),
    )


@resume_bp.route("/resume/<result_id>/delete", methods=["POST"])
@login_required
@csrf_required
def delete_resume(result_id):
    try:
        uuid.UUID(result_id)
    except ValueError:
        return jsonify({"error": "Invalid id"}), 400
    deleted = ext()["repo"].delete(result_id, session["user_id"])
    if not deleted:
        return jsonify({"error": "Not found"}), 404
    return jsonify({"success": True})


@resume_bp.route("/ats-reports")
@login_required
def ats_reports():
    rows = ext()["repo"].get_full_output_for_reports(session["user_id"])
    summary = ext()["reports"].summarize(rows)
    return render_template(
        "resume/ats_reports.html", user=session["user"], summary=summary,
        firebase_config=firebase_ctx(),
    )


@resume_bp.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    repo = ext()["repo"]
    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not ext()["csrf"].validate(session, token):
            return render_template("resume/error.html", message="Invalid CSRF token."), 400
        default_template = request.form.get("default_template", "classic")
        if default_template not in ext()["templates"].all():
            default_template = "classic"
        email_notifications = request.form.get("email_notifications") == "on"
        repo.save_settings(session["user_id"], default_template, email_notifications)

    current = repo.get_settings(session["user_id"])
    return render_template(
        "resume/settings.html", user=session["user"], settings=current,
        templates=ext()["templates"].all(), firebase_config=firebase_ctx(),
        csrf_token=ext()["csrf"].generate_token(session),
    )


@resume_bp.route("/settings/delete-account", methods=["POST"])
@login_required
@rate_limit_json(3, 300)
@csrf_required
def delete_account():
    """Permanently deletes the account: the Firebase Auth user, every
    saved resume, and the settings row. Requires the person to type
    DELETE to confirm — no accidental single-click destruction."""
    payload = request.get_json(silent=True) or {}
    if payload.get("confirm") != "DELETE":
        return jsonify({"error": "Type DELETE to confirm."}), 400

    uid = session.get("uid")
    session_id = session.get("user_id")
    repo = ext()["repo"]

    # Delete the Firebase account FIRST. If this fails, we abort without
    # having touched any app data — better to have a stray Firebase user
    # than to have silently deleted someone's resumes while their login
    # still works.
    try:
        ext()["firebase"].delete_user(uid)
    except Exception as e:
        logger.error("Firebase account deletion failed: %s", e)
        return jsonify({"error": "Couldn't delete your account right now. Please try again."}), 500

    try:
        repo.delete_all_for_session(session_id)
        repo.delete_settings(session_id)
    except Exception as e:
        # The Firebase account is already gone at this point — data
        # cleanup failing here just means orphaned rows keyed to a
        # session_id nobody can ever log in as again, not a security issue.
        logger.error("Post-deletion data cleanup failed for %s: %s", session_id, e)

    session.clear()
    return jsonify({"success": True})


@resume_bp.route("/dashboard")
@login_required
def dashboard():
    resumes = ext()["repo"].list_for_session(session["user_id"])
    best_score = max([r["ats_score"] or 0 for r in resumes], default=0)
    stats = {
        "total_resumes": len(resumes),
        "best_score": best_score,
        "last_analysis": resumes[0] if resumes else None,
    }
    return render_template(
        "resume/dashboard.html", user=session["user"], resumes=resumes, stats=stats,
        firebase_config=firebase_ctx(),
    )


@resume_bp.route("/analysis")
@login_required
def analysis():
    settings = ext()["repo"].get_settings(session["user_id"])
    return render_template(
        "resume/analysis.html", user=session["user"], firebase_config=firebase_ctx(),
        csrf_token=ext()["csrf"].generate_token(session),
        default_template=settings.get("default_template", "classic"),
        templates=ext()["templates"].all(),
    )
from flask import Flask, request, render_template, session, redirect, url_for, jsonify, send_file
import os
from dotenv import load_dotenv
from google import genai
import psycopg2
import psycopg2.extras
import json
import uuid
import io
import logging
from functools import wraps
import time

# PDF parsing
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    PDF_GEN = True
except ImportError:
    PDF_GEN = False

load_dotenv()

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── SECURITY CONFIG ──────────────────────────────────────────────────────────
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production",
    SESSION_COOKIE_SAMESITE='Lax',
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,  # 5MB max upload
)

# ─── GEMINI CLIENT ────────────────────────────────────────────────────────────
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── SIMPLE IN-MEMORY RATE LIMITER ────────────────────────────────────────────
_rate_store = {}

def rate_limit(max_calls=5, window=60):
    """Allow max_calls per window seconds per session."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user_id = session.get("user_id", request.remote_addr)
            now = time.time()
            key = f"{user_id}:{f.__name__}"
            calls = [t for t in _rate_store.get(key, []) if now - t < window]
            if len(calls) >= max_calls:
                return render_template("error.html",
                    message="Too many requests. Please wait a minute before trying again."), 429
            calls.append(now)
            _rate_store[key] = calls
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ─── SESSION SETUP ────────────────────────────────────────────────────────────
@app.before_request
def set_session():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

# ─── DB CONNECTION ────────────────────────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(
        os.getenv("DATABASE_URL"),
        cursor_factory=psycopg2.extras.RealDictCursor
    )

def init_db():
    """Create tables if they don't exist."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT NOT NULL,
                input_data JSONB,
                output_data JSONB,
                ats_score INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("DB initialized.")
    except Exception as e:
        logger.error(f"DB init failed: {e}")

# ─── INPUT VALIDATION ─────────────────────────────────────────────────────────
MAX_FIELD_LEN = {
    "skills": 3000,
    "projects": 5000,
    "experience": 5000,
    "job_description": 5000,
}

def validate_inputs(**fields):
    for name, value in fields.items():
        if not value or not value.strip():
            return f"'{name}' field is required."
        limit = MAX_FIELD_LEN.get(name, 3000)
        if len(value) > limit:
            return f"'{name}' is too long (max {limit} characters)."
    return None

# ─── PDF PARSER ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes):
    if not PDF_SUPPORT:
        return None, "PDF support not available."
    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts), None
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        return None, "Failed to parse PDF. Please paste your resume as text."

# ─── GEMINI HELPERS ───────────────────────────────────────────────────────────
def call_gemini(prompt, expect_json=True):
    try:
        # 1. Determine which model ID to use (Priority: 2.5 -> 3 -> 3-Lite)
        try:
            # Tier 1: Preferred Model
            target_model = "gemini-2.5-flash"
            print(f"Attempting Tier 1: {target_model}")
            client.models.get(target_model)
        except Exception:
            try:
                # Tier 2: First Fallback
                print("Gemini 2.5 limited. Attempting Tier 2: Gemini 3 Flash...")
                target_model = "gemini-3-flash-preview"
                client.models.get(target_model)
            except Exception:
                # Tier 3: Final Fallback (Most stable for high volume)
                print("Gemini 3 busy. Attempting Tier 3: Gemini 3.1 Flash-Lite...")
                target_model = "gemini-3.1-flash-lite-preview"

        response = client.models.generate_content(
            model=target_model,
            contents=prompt
        )
        text = response.text.strip()

        if expect_json:
            # Strip ALL markdown fence variants robustly
            import re
            # Remove ```json ... ``` or ``` ... ```
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```\s*$', '', text)
            text = text.strip()

            # Find the first { or [ to locate JSON start
            for i, ch in enumerate(text):
                if ch in ('{', '['):
                    text = text[i:]
                    break

            # Find the last } or ] to locate JSON end
            for i in range(len(text) - 1, -1, -1):
                if text[i] in ('}', ']'):
                    text = text[:i+1]
                    break

            try:
                return json.loads(text), None
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}\nRaw text (first 500): {text[:500]}")
                return None, "AI returned invalid JSON. Please try again."

        return text, None
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None, f"AI service error: {str(e)}"

def parse_resume_with_ai(raw_text):
    prompt = f"""
You are a resume parser. Extract structured data from the resume text below.

RULES:
- Extract ONLY what is explicitly present. Do NOT invent or assume anything.
- If a section is missing, use empty string "" or empty array [].
- Output ONLY valid JSON, no markdown, no explanation.

Resume Text:
{raw_text[:6000]}

OUTPUT FORMAT:
{{
  "name": "string",
  "email": "string",
  "phone": "string",
  "linkedin": "string",
  "summary": "string",
  "skills": ["skill1", "skill2"],
  "experience": [
    {{
      "role": "string",
      "company": "string",
      "duration": "string",
      "bullets": ["string"]
    }}
  ],
  "projects": [
    {{
      "name": "string",
      "description": "string",
      "tech": ["string"]
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "institution": "string",
      "year": "string"
    }}
  ]
}}
"""
    return call_gemini(prompt)

def analyze_jd_with_ai(jd_text):
    prompt = f"""
You are a job description analyzer.

Job Description:
{jd_text[:4000]}

Extract and output ONLY valid JSON:
{{
  "job_title": "string",
  "required_skills": ["string"],
  "preferred_skills": ["string"],
  "keywords": ["string"],
  "experience_level": "string",
  "tools_technologies": ["string"]
}}
"""
    return call_gemini(prompt)

def calculate_ats_score(parsed_resume, parsed_jd):
    prompt = f"""
You are an ATS (Applicant Tracking System) scoring engine.

RESUME DATA:
{json.dumps(parsed_resume, indent=2)[:3000]}

JOB DESCRIPTION DATA:
{json.dumps(parsed_jd, indent=2)[:2000]}

Calculate a detailed ATS match score. Be accurate and strict.

Output ONLY valid JSON:
{{
  "total_score": <integer 0-100>,
  "breakdown": {{
    "skill_match": <integer 0-100>,
    "keyword_coverage": <integer 0-100>,
    "experience_alignment": <integer 0-100>,
    "section_completeness": <integer 0-100>
  }},
  "matched_skills": ["string"],
  "missing_skills": ["string"],
  "matched_keywords": ["string"],
  "missing_keywords": ["string"],
  "weak_areas": ["string"],
  "recommendations": ["string"]
}}
"""
    return call_gemini(prompt)

def tailor_resume_with_ai(parsed_resume, parsed_jd, ats_report):
    prompt = f"""
You are an expert ATS resume optimizer and professional resume writer.

ORIGINAL RESUME:
{json.dumps(parsed_resume, indent=2)[:3000]}

JOB DESCRIPTION ANALYSIS:
{json.dumps(parsed_jd, indent=2)[:2000]}

ATS GAPS TO FIX:
Missing skills: {ats_report.get('missing_skills', [])}
Missing keywords: {ats_report.get('missing_keywords', [])}
Weak areas: {ats_report.get('weak_areas', [])}

STRICT RULES:
- DO NOT invent experience, skills, metrics, or achievements
- DO NOT add fake numbers or dates
- ONLY rephrase existing content for clarity and impact
- Add relevant keywords NATURALLY where they genuinely fit
- Use strong action verbs
- Keep bullets concise (1 line each)
- If a skill truly doesn't exist, leave it out

TASKS:
1. Select 3-5 most relevant projects for this JD
2. Rewrite experience bullets to align with JD
3. Generate final tailored resume as plain text (ATS-safe, no special characters)
4. Generate a 200-300 word cover letter

Output ONLY valid JSON:
{{
  "selected_projects": [
    {{
      "name": "string",
      "reason": "string",
      "bullets": ["string"]
    }}
  ],
  "tailored_experience": [
    {{
      "role": "string",
      "company": "string",
      "duration": "string",
      "bullets": ["string"]
    }}
  ],
  "resume_text": "string (full plain-text resume, ATS-friendly)",
  "cover_letter": "string"
}}
"""
    return call_gemini(prompt)

# ─── PDF EXPORT ───────────────────────────────────────────────────────────────
def generate_pdf_from_resume(resume_text, name="Resume"):
    if not PDF_GEN:
        return None, "PDF generation not available."
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        styles = getSampleStyleSheet()
        story = []

        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Normal'],
            fontSize=18,
            fontName='Helvetica-Bold',
            textColor=HexColor('#1a1a2e'),
            spaceAfter=4
        )
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            textColor=HexColor('#16213e'),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['Normal'],
            fontSize=9.5,
            fontName='Helvetica',
            textColor=HexColor('#333333'),
            spaceAfter=2,
            leading=14
        )

        lines = resume_text.split('\n')
        first_line = True
        in_section = False

        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue

            # Detect section headers (all caps or ends with :)
            if (line.isupper() and len(line) > 3) or (line.endswith(':') and len(line) < 40):
                story.append(Spacer(1, 6))
                story.append(Paragraph(line.rstrip(':'), section_style))
                story.append(HRFlowable(width="100%", thickness=0.5,
                                        color=HexColor('#cccccc'), spaceAfter=4))
                in_section = True
            elif first_line:
                story.append(Paragraph(line, name_style))
                first_line = False
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                clean = line.lstrip('•-* ').strip()
                story.append(Paragraph(f"• {clean}", body_style))
            else:
                story.append(Paragraph(line, body_style))

        doc.build(story)
        buffer.seek(0)
        return buffer, None
    except Exception as e:
        logger.error(f"PDF generation error: {e}")
        return None, str(e)

# ─── LATEX EXPORT ─────────────────────────────────────────────────────────────
def generate_latex(parsed_resume, tailored_experience=None, selected_projects=None):
    r = parsed_resume
    exp = tailored_experience or r.get("experience", [])
    proj = selected_projects or r.get("projects", [])

    def escape(s):
        if not s:
            return ""
        for old, new in [('&','\\&'),('%','\\%'),('$','\\$'),('#','\\#'),
                          ('{','\\{'),('}',"\\}"),('~','\\textasciitilde{}'),
                          ('^','\\textasciicircum{}'),('_','\\_')]:
            s = s.replace(old, new)
        return s

    skills_str = ", ".join(escape(s) for s in r.get("skills", []))

    exp_blocks = ""
    for e in exp:
        bullets = "\n".join(f"      \\resumeItem{{{escape(b)}}}" for b in e.get("bullets", []))
        exp_blocks += f"""
    \\resumeSubheading
      {{{escape(e.get('role',''))}}}{{\\textit{{{escape(e.get('duration',''))}}}}}
      {{{escape(e.get('company',''))}}}{{}}
      \\resumeItemListStart
{bullets}
      \\resumeItemListEnd
"""

    proj_blocks = ""
    for p in proj:
        tech = ", ".join(escape(t) for t in p.get("tech", []))
        proj_blocks += f"""
    \\resumeProjectHeading
      {{\\textbf{{{escape(p.get('name',''))}}} $|$ \\emph{{{tech}}}}}{{}}
      \\resumeItemListStart
        \\resumeItem{{{escape(p.get('description',''))}}}
      \\resumeItemListEnd
"""

    edu_blocks = ""
    for e in r.get("education", []):
        edu_blocks += f"""
    \\resumeSubheading
      {{{escape(e.get('institution',''))}}}{{}}
      {{{escape(e.get('degree',''))}}}{{\\textit{{{escape(e.get('year',''))}}}}}
"""

    latex = rf"""
%-------------------------
% Resumely LaTeX Export (Jake's Template - ATS Safe)
%-------------------------
\documentclass[letterpaper,11pt]{{article}}

\usepackage{{latexsym}}
\usepackage[empty]{{fullpage}}
\usepackage{{titlesec}}
\usepackage{{marvosym}}
\usepackage[usenames,dvipsnames]{{color}}
\usepackage{{verbatim}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage[english]{{babel}}
\usepackage{{tabularx}}
\input{{glyphtounicode}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyfoot{{}}
\renewcommand{{\headrulewidth}}{{0pt}}
\renewcommand{{\footrulewidth}}{{0pt}}

\addtolength{{\oddsidemargin}}{{-0.5in}}
\addtolength{{\evensidemargin}}{{-0.5in}}
\addtolength{{\textwidth}}{{1in}}
\addtolength{{\topmargin}}{{-.5in}}
\addtolength{{\textheight}}{{1.0in}}

\urlstyle{{same}}
\raggedbottom
\raggedright
\setlength{{\tabcolsep}}{{0in}}

\titleformat{{\section}}{{\vspace{{-4pt}}\scshape\raggedright\large}}{{}}{{0em}}{{}}[\color{{black}}\titlerule \vspace{{-5pt}}]

\pdfgentounicode=1

\newcommand{{\resumeItem}}[1]{{
  \item\small{{#1 \vspace{{-2pt}}}}
}}

\newcommand{{\resumeSubheading}}[4]{{
  \vspace{{-2pt}}\item
    \begin{{tabular*}}{{0.97\textwidth}}[t]{{l@{{\extracolsep{{\fill}}}}r}}
      \textbf{{#1}} & #2 \\
      \textit{{\small#3}} & \textit{{\small #4}} \\
    \end{{tabular*}}\vspace{{-7pt}}
}}

\newcommand{{\resumeProjectHeading}}[2]{{
    \item
    \begin{{tabular*}}{{0.97\textwidth}}{{l@{{\extracolsep{{\fill}}}}r}}
      \small#1 & #2 \\
    \end{{tabular*}}\vspace{{-7pt}}
}}

\newcommand{{\resumeSubItem}}[1]{{\resumeItem{{#1}}\vspace{{-4pt}}}}
\newcommand{{\resumeSubHeadingListStart}}{{\begin{{itemize}}[leftmargin=0.15in, label={{}}]}}
\newcommand{{\resumeSubHeadingListEnd}}{{\end{{itemize}}}}
\newcommand{{\resumeItemListStart}}{{\begin{{itemize}}}}
\newcommand{{\resumeItemListEnd}}{{\end{{itemize}}\vspace{{-5pt}}}}

\begin{{document}}

\begin{{center}}
    \textbf{{\Huge \scshape {escape(r.get('name','Your Name'))}}} \\ \vspace{{1pt}}
    \small {escape(r.get('phone',''))} $|$
    \href{{mailto:{escape(r.get('email',''))}}}{{\ underline{{{escape(r.get('email',''))}}}}} $|$
    \href{{{escape(r.get('linkedin',''))}}}{{\ underline{{linkedin}}}}
\end{{center}}

\section{{Education}}
  \resumeSubHeadingListStart
{edu_blocks}
  \resumeSubHeadingListEnd

\section{{Experience}}
  \resumeSubHeadingListStart
{exp_blocks}
  \resumeSubHeadingListEnd

\section{{Projects}}
    \resumeSubHeadingListStart
{proj_blocks}
    \resumeSubHeadingListEnd

\section{{Technical Skills}}
 \begin{{itemize}}[leftmargin=0.15in, label={{}}]
    \small{{\item{{
     \textbf{{Skills}}{{: {skills_str}}}
    }}}}
 \end{{itemize}}

\end{{document}}
"""
    return latex

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/parse-pdf", methods=["POST"])
@rate_limit(max_calls=10, window=60)
def parse_pdf():
    """Parse uploaded PDF and return extracted text."""
    if "resume_pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["resume_pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted"}), 400

    file_bytes = f.read()
    if len(file_bytes) > 5 * 1024 * 1024:
        return jsonify({"error": "File too large (max 5MB)"}), 400

    text, err = extract_text_from_pdf(file_bytes)
    if err:
        return jsonify({"error": err}), 500

    return jsonify({"text": text})

@app.route("/generate", methods=["POST"])
@rate_limit(max_calls=5, window=60)
def generate():
    skills     = request.form.get("skills", "").strip()
    projects   = request.form.get("projects", "").strip()
    experience = request.form.get("experience", "").strip()
    job_desc   = request.form.get("job_description", "").strip()

    # Validation
    err = validate_inputs(
        skills=skills,
        projects=projects,
        experience=experience,
        job_description=job_desc
    )
    if err:
        return render_template("error.html", message=err), 400

    # Build a raw resume text from form inputs for parsing
    raw_resume = f"""
SKILLS:
{skills}

PROJECTS:
{projects}

EXPERIENCE:
{experience}
"""

    # Step 1: Parse resume
    parsed_resume, err = parse_resume_with_ai(raw_resume)
    if err:
        return render_template("error.html", message=err), 500

    # Step 2: Analyze JD
    parsed_jd, err = analyze_jd_with_ai(job_desc)
    if err:
        return render_template("error.html", message=err), 500

    # Step 3: ATS Score
    ats_report, err = calculate_ats_score(parsed_resume, parsed_jd)
    if err:
        logger.warning(f"ATS score failed: {err}")
        ats_report = {"total_score": 0, "breakdown": {}, "matched_skills": [],
                      "missing_skills": [], "recommendations": []}

    # Step 4: Tailor resume + cover letter
    tailored, err = tailor_resume_with_ai(parsed_resume, parsed_jd, ats_report)
    if err:
        return render_template("error.html", message=err), 500

    # Assemble final output
    result = {
        "parsed_resume": parsed_resume,
        "parsed_jd": parsed_jd,
        "ats_report": ats_report,
        "selected_projects": tailored.get("selected_projects", []),
        "tailored_experience": tailored.get("tailored_experience", []),
        "resume_text": tailored.get("resume_text", ""),
        "cover_letter": tailored.get("cover_letter", ""),
        "latex": generate_latex(
            parsed_resume,
            tailored.get("tailored_experience"),
            tailored.get("selected_projects")
        )
    }

    # Save to DB
    result_id = str(uuid.uuid4())
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resumes (id, session_id, input_data, output_data, ats_score)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                result_id,
                session["user_id"],
                json.dumps({"skills": skills, "projects": projects,
                            "experience": experience, "job_description": job_desc}),
                json.dumps(result),
                ats_report.get("total_score", 0)
            )
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Saved result {result_id} for session {session['user_id'][:8]}")
    except Exception as e:
        logger.error(f"DB save failed: {e}")
        # Store in session as fallback so user isn't left hanging
        session["last_result"] = result
        session["last_result_id"] = result_id

    return redirect(url_for("result_page", result_id=result_id))


@app.route("/result/<result_id>")
def result_page(result_id):
    result = None

    # Try DB first
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT output_data, session_id FROM resumes WHERE id = %s",
            (result_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            # Ownership check
            if row["session_id"] != session.get("user_id"):
                return render_template("error.html",
                    message="You don't have access to this result."), 403
            # psycopg2 JSONB may return dict or string depending on driver
            raw = row["output_data"]
            result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        # Try session fallback
        if session.get("last_result_id") == result_id:
            result = session.get("last_result")

    if not result:
        return render_template("error.html", message="Result not found."), 404

    return render_template("result.html", result=result, result_id=result_id)


@app.route("/export/pdf/<result_id>")
def export_pdf(result_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT output_data, session_id FROM resumes WHERE id = %s",
            (result_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB error on PDF export: {e}")
        return "DB error", 500

    if not row:
        return "Result not found", 404
    if row["session_id"] != session.get("user_id"):
        return "Unauthorized", 403

    raw = row["output_data"]
    result = json.loads(raw) if isinstance(raw, str) else raw
    resume_text = result.get("resume_text", "")
    name = result.get("parsed_resume", {}).get("name", "resume") if result.get("parsed_resume") else "resume"

    pdf_buffer, err = generate_pdf_from_resume(resume_text, name)
    if err:
        return f"PDF generation failed: {err}", 500

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{name.replace(' ', '_')}_resume.pdf"
    )


@app.route("/export/latex/<result_id>")
def export_latex(result_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT output_data, session_id FROM resumes WHERE id = %s",
            (result_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
    except Exception as e:
        return "DB error", 500

    if not row:
        return "Result not found", 404
    if row["session_id"] != session.get("user_id"):
        return "Unauthorized", 403

    raw = row["output_data"]
    result = json.loads(raw) if isinstance(raw, str) else raw
    latex = result.get("latex", "")
    name = result.get("parsed_resume", {}).get("name", "resume") if result.get("parsed_resume") else "resume"

    buf = io.BytesIO(latex.encode("utf-8"))
    return send_file(
        buf,
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"{name.replace(' ', '_')}_resume.tex"
    )


# ─── ERROR PAGES ──────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404

@app.errorhandler(413)
def too_large(e):
    return render_template("error.html", message="File too large. Max 5MB."), 413

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    return render_template("error.html", message="Something went wrong. Please try again."), 500


if __name__ == "__main__":
    init_db()
    app.run(debug=os.getenv("FLASK_ENV") != "production")

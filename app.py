"""
Resumely — Production Flask App
Features: PDF text input, ATS scoring, gap analysis, skill editor,
role-aware templates, grammar check, PDF export, LaTeX export, keep-alive
"""
from flask import Flask, request, render_template, session, redirect, url_for, jsonify, send_file
import os, json, uuid, io, logging, time, re, threading
from functools import wraps
from dotenv import load_dotenv

try:
    from google import genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

import psycopg2, psycopg2.extras

try:
    import pdfplumber 
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    PDF_GEN = True
except ImportError:
    PDF_GEN = False

load_dotenv()

# ── LOGGING ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=os.getenv("FLASK_ENV") == "production",
    SESSION_COOKIE_SAMESITE="Lax",
    MAX_CONTENT_LENGTH=5 * 1024 * 1024,
)

if GEMINI_OK:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── RATE LIMITER ─────────────────────────────────────────────────
_rl = {}
def rate_limit(max_calls=5, window=60):
    def dec(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            uid = session.get("user_id", request.remote_addr)
            key = f"{uid}:{f.__name__}"
            now = time.time()
            calls = [t for t in _rl.get(key, []) if now - t < window]
            if len(calls) >= max_calls:
                return render_template("error.html", message="Too many requests. Wait a minute."), 429
            calls.append(now)
            _rl[key] = calls
            return f(*args, **kwargs)
        return wrapper
    return dec
 
@app.before_request
def ensure_session():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

# ── DB ────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.RealDictCursor)

def init_db():
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS resumes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id TEXT NOT NULL,
                input_data JSONB,
                output_data JSONB,
                ats_score INTEGER DEFAULT 0,
                job_category TEXT DEFAULT 'General',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit(); cur.close(); conn.close()
        logger.info("DB ready.")
    except Exception as e:
        logger.error(f"DB init: {e}")

# ── KEEP-ALIVE (prevents Render spin-down) ───────────────────────
def keep_alive():
    """Pings self every 14 minutes to prevent Render free-tier spin-down."""
    import urllib.request
    base_url = os.getenv("APP_URL", "")
    if not base_url:
        return
    def ping():
        while True:
            time.sleep(14 * 60)
            try:
                urllib.request.urlopen(f"{base_url}/ping", timeout=10)
                logger.info("Keep-alive ping sent.")
            except Exception as e:
                logger.warning(f"Keep-alive failed: {e}")
    t = threading.Thread(target=ping, daemon=True)
    t.start()

@app.route("/ping")
def ping():
    return "ok", 200

# ── VALIDATION ────────────────────────────────────────────────────
LIMITS = {"skills": 3000, "projects": 5000, "experience": 5000, "job_description": 5000}
def validate(**fields):
    for name, val in fields.items():
        if not val or not val.strip():
            return f"'{name}' is required."
        if len(val) > LIMITS.get(name, 3000):
            return f"'{name}' is too long (max {LIMITS.get(name,3000)} chars)."
    return None

# ── PDF TEXT EXTRACTION ───────────────────────────────────────────
def pdf_to_text(file_bytes):
    if not PDF_SUPPORT:
        return None, "pdfplumber not installed."
    try:
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t: parts.append(t)
        return "\n".join(parts), None
    except Exception as e:
        logger.error(f"PDF extract: {e}")
        return None, str(e)

# ── GEMINI WRAPPER ────────────────────────────────────────────────
def gemini(prompt):
    try:
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

        resp = client.models.generate_content(model=target_model, contents=prompt)
        text = resp.text.strip()
        # Strip all markdown fence variants
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```\s*$', '', text).strip()
        # Find JSON boundaries
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                text = text[i:]; break
        for i in range(len(text)-1, -1, -1):
            if text[i] in ('}', ']'):
                text = text[:i+1]; break
        return json.loads(text), None
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse: {e}")
        return None, "AI returned invalid JSON. Try again."
    except Exception as e:
        logger.error(f"Gemini: {e}")
        return None, str(e)

# ── AI MODULES ────────────────────────────────────────────────────
def parse_resume(raw):
    return gemini(f"""
You are a resume parser. Extract ONLY what is present. Do NOT invent anything.
Output ONLY valid JSON — no markdown, no explanation.

Resume:
{raw[:6000]}

FORMAT:
{{"name":"","email":"","phone":"","linkedin":"","summary":"",
"skills":[],"experience":[{{"role":"","company":"","duration":"","bullets":[]}}],
"projects":[{{"name":"","description":"","tech":[]}}],
"education":[{{"degree":"","institution":"","year":""}}]}}
""")

def analyze_jd(jd):
    return gemini(f"""
Analyze this job description and extract structured data.
Output ONLY valid JSON.

JD:
{jd[:4000]}

FORMAT:
{{"job_title":"","job_category":"Engineering",
"required_skills":[],"preferred_skills":[],"keywords":[],
"experience_level":"","tools_technologies":[],"industry":""}}

For job_category use one of: Engineering, Finance, Design, Academia, Operations,
Healthcare, Legal, Marketing, Sales, Data Science, Government, Management
""")

def ats_score(parsed_resume, parsed_jd):
    return gemini(f"""
You are an ATS scoring engine. Be accurate and strict.
Output ONLY valid JSON.

RESUME:
{json.dumps(parsed_resume)[:3000]}

JD:
{json.dumps(parsed_jd)[:2000]}

FORMAT:
{{"total_score":0,
"breakdown":{{"skill_match":0,"keyword_coverage":0,"experience_alignment":0,"section_completeness":0}},
"matched_skills":[],"missing_skills":[],"matched_keywords":[],"missing_keywords":[],
"you_have_but_not_listed":[],"weak_areas":[],"recommendations":[]}}

Note: "you_have_but_not_listed" = skills the resume text suggests the person likely has
based on their projects/experience, but didn't explicitly list as skills.
""")

def tailor_resume(parsed_resume, parsed_jd, ats, template_id, extra_skills):
    extra_str = ", ".join(extra_skills) if extra_skills else "none added"
    return gemini(f"""
You are an expert ATS resume optimizer for ALL job types including Engineering, Finance,
Banking, Government, Design, Healthcare, Academia, Operations.

RESUME:
{json.dumps(parsed_resume)[:3000]}

JD:
{json.dumps(parsed_jd)[:2000]}

GAPS: missing={ats.get('missing_skills',[])} keywords={ats.get('missing_keywords',[])}
EXTRA SKILLS USER CONFIRMED: {extra_str}

STRICT RULES:
- Do NOT invent experience, metrics, or achievements
- Add extra_skills naturally where they genuinely fit
- Use strong action verbs appropriate for the industry/role
- Keep bullets concise (1 line)

Output ONLY valid JSON:
{{"selected_projects":[],"tailored_experience":[],"resume_text":"","cover_letter":""}}

selected_projects: [{{"name":"","reason":"","bullets":[]}}]
tailored_experience: [{{"role":"","company":"","duration":"","bullets":[]}}]
resume_text: full ATS-safe plain text resume
cover_letter: 200-300 words, strong opening, aligned with JD, role-appropriate
""")

def grammar_check_text(text):
    return gemini(f"""
Check this resume text for grammar, passive voice, and weak verbs.
Return a JSON list of issues. Keep it concise.
Output ONLY valid JSON array.

Text:
{text[:3000]}

FORMAT:
[{{"original":"","suggestion":"","type":"grammar|passive|weak_verb","severity":"low|medium|high"}}]
""")

# ── RESUME TEMPLATES ──────────────────────────────────────────────
TEMPLATES = {
    "classic":    {"name":"Classic",    "cat":"All",         "desc":"Clean, traditional ATS-safe"},
    "modern":     {"name":"Modern",     "cat":"Engineering",  "desc":"Two-column with skills sidebar"},
    "minimal":    {"name":"Minimal",    "cat":"Design",       "desc":"Ultra-clean, whitespace focused"},
    "executive":  {"name":"Executive",  "cat":"Management",   "desc":"Formal, suited for senior roles"},
    "academic":   {"name":"Academic",   "cat":"Academia",     "desc":"CV-style with publications"},
    "finance":    {"name":"Finance",    "cat":"Finance",      "desc":"Metrics-focused, conservative"},
    "government": {"name":"Government", "cat":"Government",   "desc":"GS-compatible, detailed format"},
    "creative":   {"name":"Creative",   "cat":"Design",       "desc":"Portfolio-style layout"},
    "tech":       {"name":"Tech",       "cat":"Engineering",  "desc":"GitHub/project-centric"},
    "healthcare": {"name":"Healthcare", "cat":"Healthcare",   "desc":"Certifications prominently displayed"},
}

def recommend_template(job_category):
    mapping = {
        "Engineering":   "tech",
        "Finance":       "finance",
        "Banking":       "finance",
        "Design":        "creative",
        "Academia":      "academic",
        "Operations":    "classic",
        "Healthcare":    "healthcare",
        "Legal":         "executive",
        "Marketing":     "modern",
        "Sales":         "modern",
        "Government":    "government",
        "Management":    "executive",
        "Data Science":  "tech",
    }
    return mapping.get(job_category, "classic")

# ── PDF GENERATION ────────────────────────────────────────────────
def make_pdf(resume_text, template_id="classic", name="Resume"):
    if not PDF_GEN:
        return None, "ReportLab not installed."
    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter,
            leftMargin=0.75*inch, rightMargin=0.75*inch,
            topMargin=0.75*inch, bottomMargin=0.75*inch)

        # Template color mapping
        colors = {
            "classic":    "#1D4ED8", "modern":     "#0F766E",
            "minimal":    "#374151", "executive":  "#1F2937",
            "academic":   "#4338CA", "finance":    "#1D4ED8",
            "government": "#1E40AF", "creative":   "#7C3AED",
            "tech":       "#0369A1", "healthcare": "#0E7490",
        }
        primary_hex = colors.get(template_id, "#1D4ED8")
        primary = HexColor(primary_hex)

        styles = getSampleStyleSheet()
        name_style = ParagraphStyle('N', parent=styles['Normal'],
            fontSize=18, fontName='Helvetica-Bold', textColor=HexColor('#0F172A'), spaceAfter=3)
        section_style = ParagraphStyle('S', parent=styles['Normal'],
            fontSize=11, fontName='Helvetica-Bold', textColor=primary,
            spaceBefore=10, spaceAfter=4)
        body_style = ParagraphStyle('B', parent=styles['Normal'],
            fontSize=9.5, fontName='Helvetica', textColor=HexColor('#334155'),
            spaceAfter=2, leading=14)

        story = []; first = True
        for line in resume_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1,4)); continue
            if first:
                story.append(Paragraph(line, name_style)); first = False
            elif line.isupper() and len(line) > 2:
                story.append(Spacer(1,6))
                story.append(Paragraph(line, section_style))
                story.append(HRFlowable(width="100%", thickness=0.8, color=primary, spaceAfter=4))
            elif line.startswith(('•','▸','-','*')):
                story.append(Paragraph(f"• {line.lstrip('•▸-* ').strip()}", body_style))
            else:
                story.append(Paragraph(line, body_style))
        doc.build(story)
        buf.seek(0)
        return buf, None
    except Exception as e:
        logger.error(f"PDF gen: {e}")
        return None, str(e)

# ── LATEX GENERATION ──────────────────────────────────────────────
def make_latex(parsed_resume, tailored_exp=None, selected_proj=None):
    r = parsed_resume or {}
    exp  = tailored_exp  or r.get("experience", [])
    proj = selected_proj or r.get("projects", [])

    def esc(s):
        if not s: return ""
        for o, n in [('&','\\&'),('%','\\%'),('$','\\$'),('#','\\#'),
                     ('{','\\{'),('}',"\\}"),('~','\\textasciitilde{}'),
                     ('^','\\textasciicircum{}'),('_','\\_')]:
            s = s.replace(o, n)
        return s

    skills_str = ", ".join(esc(s) for s in r.get("skills", []))
    exp_blocks = ""
    for e in exp:
        bullets = "\n".join(f"      \\resumeItem{{{esc(b)}}}" for b in e.get("bullets", []))
        exp_blocks += f"""
    \\resumeSubheading{{{esc(e.get('role',''))}}}{{{esc(e.get('duration',''))}}}
      {{{esc(e.get('company',''))}}}{{}}
      \\resumeItemListStart
{bullets}
      \\resumeItemListEnd"""

    proj_blocks = ""
    for p in proj:
        tech = ", ".join(esc(t) for t in p.get("tech",[]))
        desc = esc(p.get("description","") or (p.get("bullets",[""])[0] if p.get("bullets") else ""))
        proj_blocks += f"""
    \\resumeProjectHeading{{\\textbf{{{esc(p.get('name',''))}}} $|$ \\emph{{{tech}}}}}{{}}
      \\resumeItemListStart
        \\resumeItem{{{desc}}}
      \\resumeItemListEnd"""

    edu_blocks = ""
    for e in r.get("education", []):
        edu_blocks += f"""
    \\resumeSubheading{{{esc(e.get('institution',''))}}}{{{esc(e.get('year',''))}}}
      {{{esc(e.get('degree',''))}}}{{}}"""

    return rf"""
%-- Resumely LaTeX Export (Jake's Template, ATS-Safe) --%
\documentclass[letterpaper,11pt]{{article}}
\usepackage{{latexsym}}\usepackage[empty]{{fullpage}}\usepackage{{titlesec}}
\usepackage{{marvosym}}\usepackage[usenames,dvipsnames]{{color}}
\usepackage{{verbatim}}\usepackage{{enumitem}}\usepackage[hidelinks]{{hyperref}}
\usepackage{{fancyhdr}}\usepackage[english]{{babel}}\usepackage{{tabularx}}
\input{{glyphtounicode}}
\pagestyle{{fancy}}\fancyhf{{}}\fancyfoot{{}}\renewcommand{{\headrulewidth}}{{0pt}}
\addtolength{{\oddsidemargin}}{{-0.5in}}\addtolength{{\evensidemargin}}{{-0.5in}}
\addtolength{{\textwidth}}{{1in}}\addtolength{{\topmargin}}{{-.5in}}\addtolength{{\textheight}}{{1.0in}}
\urlstyle{{same}}\raggedbottom\raggedright\setlength{{\tabcolsep}}{{0in}}
\titleformat{{\section}}{{\vspace{{-4pt}}\scshape\raggedright\large}}{{}}{{0em}}{{}}[\color{{black}}\titlerule\vspace{{-5pt}}]
\pdfgentounicode=1
\newcommand{{\resumeItem}}[1]{{\item\small{{#1\vspace{{-2pt}}}}}}
\newcommand{{\resumeSubheading}}[4]{{\vspace{{-2pt}}\item
    \begin{{tabular*}}{{0.97\textwidth}}[t]{{l@{{\extracolsep{{\fill}}}}r}}
      \textbf{{#1}} & #2 \\\\ \textit{{\small#3}} & \textit{{\small #4}} \\\\
    \end{{tabular*}}\vspace{{-7pt}}}}
\newcommand{{\resumeProjectHeading}}[2]{{\item
    \begin{{tabular*}}{{0.97\textwidth}}{{l@{{\extracolsep{{\fill}}}}r}}
      \small#1 & #2 \\\\
    \end{{tabular*}}\vspace{{-7pt}}}}
\newcommand{{\resumeSubHeadingListStart}}{{\begin{{itemize}}[leftmargin=0.15in,label={{}}]}}
\newcommand{{\resumeSubHeadingListEnd}}{{\end{{itemize}}}}
\newcommand{{\resumeItemListStart}}{{\begin{{itemize}}}}
\newcommand{{\resumeItemListEnd}}{{\end{{itemize}}\vspace{{-5pt}}}}
\begin{{document}}
\begin{{center}}
    \textbf{{\Huge\scshape {esc(r.get('name','Your Name'))}}} \\\\ \vspace{{1pt}}
    \small {esc(r.get('phone',''))} $|$ \href{{mailto:{esc(r.get('email',''))}}}{{\underline{{{esc(r.get('email',''))}}}}} $|$
    \href{{{esc(r.get('linkedin',''))}}}{{\underline{{linkedin}}}}
\end{{center}}
\section{{Education}}\resumeSubHeadingListStart{edu_blocks}\resumeSubHeadingListEnd
\section{{Experience}}\resumeSubHeadingListStart{exp_blocks}\resumeSubHeadingListEnd
\section{{Projects}}\resumeSubHeadingListStart{proj_blocks}\resumeSubHeadingListEnd
\section{{Technical Skills}}
\begin{{itemize}}[leftmargin=0.15in,label={{}}]
  \small{{\item{{\textbf{{Skills}}{{: {skills_str}}}}}}}
\end{{itemize}}
\end{{document}}
"""

# ── ROUTES ────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/parse-pdf", methods=["POST"])
@rate_limit(10, 60)
def parse_pdf_route():
    if "resume_pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["resume_pdf"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files accepted"}), 400
    data = f.read()
    if len(data) > 5*1024*1024:
        return jsonify({"error": "File too large (max 5MB)"}), 400
    text, err = pdf_to_text(data)
    if err:
        return jsonify({"error": err}), 500
    return jsonify({"text": text})

@app.route("/grammar-check", methods=["POST"])
@rate_limit(10, 60)
def grammar_check():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"issues": []})
    issues, err = grammar_check_text(text)
    if err:
        return jsonify({"issues": []})
    return jsonify({"issues": issues or []})

@app.route("/generate", methods=["POST"])
@rate_limit(5, 60)
def generate():
    skills      = request.form.get("skills", "").strip()
    projects    = request.form.get("projects", "").strip()
    experience  = request.form.get("experience", "").strip()
    job_desc    = request.form.get("job_description", "").strip()
    template_id = request.form.get("template_id", "classic")
    extra_skills_raw = request.form.get("extra_skills", "")
    extra_skills = [s.strip() for s in extra_skills_raw.split(",") if s.strip()] if extra_skills_raw else []

    err = validate(skills=skills, projects=projects, experience=experience, job_description=job_desc)
    if err:
        return render_template("error.html", message=err), 400

    raw = f"SKILLS:\n{skills}\n\nPROJECTS:\n{projects}\n\nEXPERIENCE:\n{experience}"

    # Step 1: Parse
    parsed_resume, err = parse_resume(raw)
    if err:
        return render_template("error.html", message=err), 500

    # Step 2: Analyze JD
    parsed_jd, err = analyze_jd(job_desc)
    if err:
        return render_template("error.html", message=err), 500

    # Step 3: ATS Score
    ats, err = ats_score(parsed_resume, parsed_jd)
    if err:
        logger.warning(f"ATS failed: {err}")
        ats = {"total_score":0,"breakdown":{},"matched_skills":[],"missing_skills":[],
               "missing_keywords":[],"you_have_but_not_listed":[],"recommendations":[]}

    # Auto-select template if not manually chosen
    if template_id == "classic" and parsed_jd:
        template_id = recommend_template(parsed_jd.get("job_category","General"))

    # Step 4: Tailor
    tailored, err = tailor_resume(parsed_resume, parsed_jd, ats, template_id, extra_skills)
    if err:
        return render_template("error.html", message=err), 500

    result = {
        "parsed_resume":      parsed_resume,
        "parsed_jd":          parsed_jd,
        "ats_report":         ats,
        "selected_projects":  tailored.get("selected_projects", []),
        "tailored_experience":tailored.get("tailored_experience", []),
        "resume_text":        tailored.get("resume_text", ""),
        "cover_letter":       tailored.get("cover_letter", ""),
        "template_id":        template_id,
        "latex":              make_latex(parsed_resume,
                                         tailored.get("tailored_experience"),
                                         tailored.get("selected_projects")),
    }

    result_id = str(uuid.uuid4())
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute(
            "INSERT INTO resumes (id,session_id,input_data,output_data,ats_score,job_category) VALUES (%s,%s,%s,%s,%s,%s)",
            (result_id, session["user_id"],
             json.dumps({"skills":skills,"projects":projects,"experience":experience,"job_description":job_desc}),
             json.dumps(result),
             ats.get("total_score",0),
             parsed_jd.get("job_category","General") if parsed_jd else "General")
        )
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        logger.error(f"DB save: {e}")
        session["last_result"] = result
        session["last_result_id"] = result_id

    return redirect(url_for("result_page", result_id=result_id))


@app.route("/result/<result_id>")
def result_page(result_id):
    result = None
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT output_data, session_id FROM resumes WHERE id=%s", (result_id,))
        row = cur.fetchone(); cur.close(); conn.close()
        if row:
            if row["session_id"] != session.get("user_id"):
                return render_template("error.html", message="Access denied."), 403
            raw = row["output_data"]
            result = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        logger.error(f"DB fetch: {e}")
        if session.get("last_result_id") == result_id:
            result = session.get("last_result")

    if not result:
        return render_template("error.html", message="Result not found."), 404
    return render_template("result.html", result=result, result_id=result_id,
                           templates=TEMPLATES)


@app.route("/update-skills/<result_id>", methods=["POST"])
def update_skills(result_id):
    """User adds skills they have but weren't listed → update result."""
    new_skills = request.json.get("skills", [])
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT output_data, session_id FROM resumes WHERE id=%s", (result_id,))
        row = cur.fetchone()
        if not row or row["session_id"] != session.get("user_id"):
            return jsonify({"error": "Not found"}), 404
        raw = row["output_data"]
        result = json.loads(raw) if isinstance(raw, str) else raw

        # Add new skills to parsed_resume.skills
        existing = result.get("parsed_resume", {}).get("skills", [])
        updated = list(set(existing + new_skills))
        result["parsed_resume"]["skills"] = updated

        # Update ats matched skills
        ats = result.get("ats_report", {})
        ats["matched_skills"] = list(set(ats.get("matched_skills", []) + new_skills))
        ats["missing_skills"] = [s for s in ats.get("missing_skills", []) if s not in new_skills]
        ats["you_have_but_not_listed"] = [s for s in ats.get("you_have_but_not_listed", []) if s not in new_skills]

        # Recalculate score approximately
        total = ats.get("total_score", 0)
        bonus = min(len(new_skills) * 3, 15)
        ats["total_score"] = min(total + bonus, 100)
        result["ats_report"] = ats

        cur.execute("UPDATE resumes SET output_data=%s, ats_score=%s WHERE id=%s",
                    (json.dumps(result), ats["total_score"], result_id))
        conn.commit(); cur.close(); conn.close()
        return jsonify({"success": True, "new_score": ats["total_score"], "updated_skills": updated})
    except Exception as e:
        logger.error(f"Update skills: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/export/pdf/<result_id>")
def export_pdf(result_id):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT output_data, session_id FROM resumes WHERE id=%s", (result_id,))
        row = cur.fetchone(); cur.close(); conn.close()
    except Exception as e:
        return f"DB error: {e}", 500
    if not row: return "Not found", 404
    if row["session_id"] != session.get("user_id"): return "Unauthorized", 403
    raw = row["output_data"]
    result = json.loads(raw) if isinstance(raw, str) else raw
    resume_text = result.get("resume_text", "")
    template_id = result.get("template_id", "classic")
    name = (result.get("parsed_resume") or {}).get("name", "resume")
    buf, err = make_pdf(resume_text, template_id, name)
    if err: return f"PDF error: {err}", 500
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{name.replace(' ','_')}_resume.pdf")


@app.route("/export/latex/<result_id>")
def export_latex(result_id):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT output_data, session_id FROM resumes WHERE id=%s", (result_id,))
        row = cur.fetchone(); cur.close(); conn.close()
    except Exception as e:
        return f"DB error: {e}", 500
    if not row: return "Not found", 404
    if row["session_id"] != session.get("user_id"): return "Unauthorized", 403
    raw = row["output_data"]
    result = json.loads(raw) if isinstance(raw, str) else raw
    latex = result.get("latex", "")
    name = (result.get("parsed_resume") or {}).get("name", "resume")
    return send_file(io.BytesIO(latex.encode()), mimetype="text/plain",
                     as_attachment=True, download_name=f"{name.replace(' ','_')}_resume.tex")


@app.errorhandler(404)
def e404(e): return render_template("error.html", message="Page not found."), 404
@app.errorhandler(413)
def e413(e): return render_template("error.html", message="File too large (max 5MB)."), 413
@app.errorhandler(500)
def e500(e): logger.error(e); return render_template("error.html", message="Server error. Try again."), 500


if __name__ == "__main__":
    init_db()
    keep_alive()
    app.run(debug=os.getenv("FLASK_ENV") != "production")

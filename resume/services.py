import io
import re
import json
import time
import logging

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    PDF_GEN = True
except ImportError:
    PDF_GEN = False

try:
    from google import genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False


# ── RATE LIMITER ──────────────────────────────────────────────────
class RateLimiter:
    """In-memory sliding-window limiter.

    NOTE: this state lives in one process's memory. If you deploy with
    multiple gunicorn workers each worker enforces its own limit
    independently, so the *effective* limit is (per-worker limit x worker
    count). That's fine for a single small dyno; move this to Redis
    (INCR + EXPIRE) if you scale to multiple workers/instances.
    """

    def __init__(self):
        self._hits = {}

    def allow(self, key, max_calls, window_seconds):
        now = time.time()
        calls = [t for t in self._hits.get(key, []) if now - t < window_seconds]
        if len(calls) >= max_calls:
            self._hits[key] = calls
            return False
        calls.append(now)
        self._hits[key] = calls
        return True


# ── GEMINI CLIENT ─────────────────────────────────────────────────
class GeminiClient:
    """Wraps model selection/fallback and strict JSON extraction so the
    rest of the app never has to think about markdown fences or which
    model tier is currently healthy."""

    MODEL_TIERS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview"]

    def __init__(self, api_key):
        self.enabled = GEMINI_OK and bool(api_key)
        self.client = genai.Client(api_key=api_key) if self.enabled else None

    def _pick_model(self):
        for model in self.MODEL_TIERS[:-1]:
            try:
                self.client.models.get(model)
                return model
            except Exception:
                continue
        return self.MODEL_TIERS[-1]

    @staticmethod
    def _extract_json(text):
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```\s*$", "", text).strip()
        start = next((i for i, ch in enumerate(text) if ch in "{["), None)
        end = next((i for i in range(len(text) - 1, -1, -1) if text[i] in "}]"), None)
        if start is None or end is None:
            raise ValueError("No JSON object found in model output.")
        return text[start:end + 1]

    def generate_json(self, prompt):
        if not self.enabled:
            return None, "AI service is not configured."
        try:
            model = self._pick_model()
            resp = self.client.models.generate_content(model=model, contents=prompt)
            cleaned = self._extract_json(resp.text)
            return json.loads(cleaned), None
        except json.JSONDecodeError as e:
            logger.error("Gemini JSON parse error: %s", e)
            return None, "AI returned invalid JSON. Please try again."
        except Exception as e:
            logger.error("Gemini call failed: %s", e)
            return None, "AI request failed. Please try again."


# ── PDF TEXT EXTRACTION ───────────────────────────────────────────
class PdfTextExtractor:
    @staticmethod
    def extract(file_bytes):
        if not PDF_SUPPORT:
            return None, "PDF parsing isn't available on the server (pdfplumber missing)."
        try:
            parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        parts.append(text)
            full_text = "\n".join(parts).strip()
            if not full_text:
                return None, "Couldn't read any text from that PDF. Is it a scanned image?"
            return full_text, None
        except Exception as e:
            logger.error("PDF extract failed: %s", e)
            return None, "That file couldn't be read as a PDF."


# ── AI-BACKED RESUME PIPELINE ─────────────────────────────────────
class ResumeAI:
    """All Gemini prompts live here, in one place, sharing one client."""

    def __init__(self, gemini_client: GeminiClient):
        self.ai = gemini_client

    def parse_resume(self, raw_text):
        return self.ai.generate_json(f"""
You are a resume parser. Extract ONLY what is present. Do NOT invent anything.
Output ONLY valid JSON — no markdown, no explanation.

Resume:
{raw_text[:6000]}

FORMAT:
{{"name":"","email":"","phone":"","location":"","linkedin":"","github":"","portfolio":"",
"summary":"","skills":[],"hobbies":[],
"experience":[{{"role":"","company":"","duration":"","bullets":[]}}],
"projects":[{{"name":"","description":"","tech":[]}}],
"education":[{{"degree":"","institution":"","year":""}}]}}
""")

    def analyze_jd(self, jd_text):
        return self.ai.generate_json(f"""
Analyze this job description and extract structured data.
Output ONLY valid JSON.

JD:
{jd_text[:4000]}

FORMAT:
{{"job_title":"","job_category":"Engineering",
"required_skills":[],"preferred_skills":[],"keywords":[],
"experience_level":"","tools_technologies":[],"industry":""}}

For job_category use one of: Engineering, Finance, Design, Academia, Operations,
Healthcare, Legal, Marketing, Sales, Data Science, Government, Management
""")

    def score_ats(self, parsed_resume, parsed_jd):
        return self.ai.generate_json(f"""
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

    def tailor(self, parsed_resume, parsed_jd, ats, template_id, extra_skills):
        extra_str = ", ".join(extra_skills) if extra_skills else "none added"
        return self.ai.generate_json(f"""
You are an expert ATS resume optimizer for ALL job types including Engineering, Finance,
Banking, Government, Design, Healthcare, Academia, Operations.

RESUME:
{json.dumps(parsed_resume)[:3000]}

JD:
{json.dumps(parsed_jd)[:2000]}

GAPS: missing={ats.get('missing_skills', [])} keywords={ats.get('missing_keywords', [])}
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

    def parse_pdf_into_form_fields(self, raw_text):
        """Used by /parse-pdf: turns raw extracted PDF text into the exact
        shape the analysis form's textareas expect (flat strings), plus
        contact fields, so the frontend can drop it straight into inputs."""
        parsed, err = self.parse_resume(raw_text)
        if err:
            return None, err

        skills_str = ", ".join(parsed.get("skills", []))

        project_lines = []
        for p in parsed.get("projects", []):
            line = p.get("name", "")
            if p.get("description"):
                line += f": {p['description']}"
            if p.get("tech"):
                line += f" (Tech: {', '.join(p['tech'])})"
            project_lines.append(line)

        exp_lines = []
        for e in parsed.get("experience", []):
            header = f"{e.get('role','')} at {e.get('company','')} ({e.get('duration','')})"
            exp_lines.append(header)
            for b in e.get("bullets", []):
                exp_lines.append(f"- {b}")

        return {
            "skills": skills_str,
            "projects": "\n".join(project_lines),
            "experience": "\n".join(exp_lines),
            "name": parsed.get("name", ""),
            "email": parsed.get("email", ""),
            "phone": parsed.get("phone", ""),
            "location": parsed.get("location", ""),
            "linkedin": parsed.get("linkedin", ""),
            "github": parsed.get("github", ""),
            "portfolio": parsed.get("portfolio", ""),
            "hobbies": ", ".join(parsed.get("hobbies", [])),
        }, None


# ── TEMPLATES ─────────────────────────────────────────────────────
class TemplateManager:
    TEMPLATES = {
        "classic":    {"name": "Classic",    "cat": "All",        "desc": "Clean, traditional ATS-safe",       "color": "#1D4ED8", "lead_section": "experience"},
        "modern":     {"name": "Modern",     "cat": "Engineering", "desc": "Two-column with skills sidebar",    "color": "#0F766E", "lead_section": "skills"},
        "minimal":    {"name": "Minimal",    "cat": "Design",      "desc": "Ultra-clean, whitespace focused",   "color": "#374151", "lead_section": "experience"},
        "executive":  {"name": "Executive",  "cat": "Management",  "desc": "Formal, suited for senior roles",   "color": "#1F2937", "lead_section": "experience"},
        "academic":   {"name": "Academic",   "cat": "Academia",    "desc": "CV-style with publications",        "color": "#4338CA", "lead_section": "education"},
        "finance":    {"name": "Finance",    "cat": "Finance",     "desc": "Metrics-focused, conservative",     "color": "#1D4ED8", "lead_section": "experience"},
        "government": {"name": "Government", "cat": "Government",  "desc": "GS-compatible, detailed format",    "color": "#1E40AF", "lead_section": "experience"},
        "creative":   {"name": "Creative",   "cat": "Design",      "desc": "Portfolio-style layout",            "color": "#7C3AED", "lead_section": "projects"},
        "tech":       {"name": "Tech",       "cat": "Engineering", "desc": "GitHub/project-centric",            "color": "#0369A1", "lead_section": "projects"},
        "healthcare": {"name": "Healthcare", "cat": "Healthcare",  "desc": "Certifications prominently displayed", "color": "#0E7490", "lead_section": "education"},
    }

    CATEGORY_TO_TEMPLATE = {
        "Engineering": "tech", "Finance": "finance", "Banking": "finance",
        "Design": "creative", "Academia": "academic", "Operations": "classic",
        "Healthcare": "healthcare", "Legal": "executive", "Marketing": "modern",
        "Sales": "modern", "Government": "government", "Management": "executive",
        "Data Science": "tech",
    }

    def all(self):
        return self.TEMPLATES

    def get(self, template_id):
        return self.TEMPLATES.get(template_id, self.TEMPLATES["classic"])

    def recommend(self, job_category):
        return self.CATEGORY_TO_TEMPLATE.get(job_category, "classic")


# ── PDF BUILDER ───────────────────────────────────────────────────
class PdfBuilder:
    def __init__(self, template_manager: TemplateManager):
        self.templates = template_manager

    def build(self, resume_text, template_id, contact_line=""):
        if not PDF_GEN:
            return None, "PDF generation isn't available on the server (reportlab missing)."
        try:
            buf = io.BytesIO()
            doc = SimpleDocTemplate(
                buf, pagesize=letter,
                leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                topMargin=0.75 * inch, bottomMargin=0.75 * inch,
            )
            primary = HexColor(self.templates.get(template_id)["color"])

            styles = getSampleStyleSheet()
            name_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=18,
                                         fontName='Helvetica-Bold', textColor=HexColor('#0F172A'), spaceAfter=2)
            contact_style = ParagraphStyle('C', parent=styles['Normal'], fontSize=9,
                                            fontName='Helvetica', textColor=HexColor('#64748B'), spaceAfter=10)
            section_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=11,
                                            fontName='Helvetica-Bold', textColor=primary, spaceBefore=10, spaceAfter=4)
            body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=9.5,
                                         fontName='Helvetica', textColor=HexColor('#334155'), spaceAfter=2, leading=14)

            story = []
            first = True
            for line in resume_text.split('\n'):
                line = line.strip()
                if not line:
                    story.append(Spacer(1, 4))
                    continue
                if first:
                    story.append(Paragraph(line, name_style))
                    if contact_line:
                        story.append(Paragraph(contact_line, contact_style))
                    first = False
                elif line.isupper() and len(line) > 2:
                    story.append(Spacer(1, 6))
                    story.append(Paragraph(line, section_style))
                    story.append(HRFlowable(width="100%", thickness=0.8, color=primary, spaceAfter=4))
                elif line.startswith(('•', '▸', '-', '*')):
                    story.append(Paragraph(f"• {line.lstrip('•▸-* ').strip()}", body_style))
                else:
                    story.append(Paragraph(line, body_style))
            doc.build(story)
            buf.seek(0)
            return buf, None
        except Exception as e:
            logger.error("PDF gen failed: %s", e)
            return None, "Couldn't generate the PDF. Please try again."


# ── LATEX BUILDER ─────────────────────────────────────────────────
class LatexBuilder:
    @staticmethod
    def _esc(s):
        if not s:
            return ""
        s = str(s)
        for o, n in [('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('$', '\\$'),
                     ('#', '\\#'), ('{', '\\{'), ('}', '\\}'),
                     ('~', '\\textasciitilde{}'), ('^', '\\textasciicircum{}'), ('_', '\\_')]:
            s = s.replace(o, n)
        return s

    def build(self, parsed_resume, tailored_exp=None, selected_proj=None):
        esc = self._esc
        r = parsed_resume or {}
        exp = tailored_exp or r.get("experience", [])
        proj = selected_proj or r.get("projects", [])

        skills_str = ", ".join(esc(s) for s in r.get("skills", []))
        hobbies_str = ", ".join(esc(h) for h in r.get("hobbies", []))

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
            tech = ", ".join(esc(t) for t in p.get("tech", []))
            desc = esc(p.get("description", "") or (p.get("bullets", [""])[0] if p.get("bullets") else ""))
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

        # Build the contact line piece-by-piece so empty fields don't
        # leave dangling "$|$" separators or empty \href{}{} links.
        contact_parts = []
        if r.get("phone"):
            contact_parts.append(esc(r["phone"]))
        if r.get("location"):
            contact_parts.append(esc(r["location"]))
        if r.get("email"):
            contact_parts.append(f"\\href{{mailto:{esc(r['email'])}}}{{\\underline{{{esc(r['email'])}}}}}")
        if r.get("linkedin"):
            contact_parts.append(f"\\href{{{esc(r['linkedin'])}}}{{\\underline{{LinkedIn}}}}")
        if r.get("github"):
            contact_parts.append(f"\\href{{{esc(r['github'])}}}{{\\underline{{GitHub}}}}")
        if r.get("portfolio"):
            contact_parts.append(f"\\href{{{esc(r['portfolio'])}}}{{\\underline{{Portfolio}}}}")
        contact_line = " $|$ ".join(contact_parts)

        hobbies_section = ""
        if hobbies_str:
            hobbies_section = f"""
\\section{{Interests}}
\\begin{{itemize}}[leftmargin=0.15in,label={{}}]
  \\small{{\\item{{{hobbies_str}}}}}
\\end{{itemize}}"""

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
    \small {contact_line}
\end{{center}}
\section{{Education}}\resumeSubHeadingListStart{edu_blocks}\resumeSubHeadingListEnd
\section{{Experience}}\resumeSubHeadingListStart{exp_blocks}\resumeSubHeadingListEnd
\section{{Projects}}\resumeSubHeadingListStart{proj_blocks}\resumeSubHeadingListEnd
\section{{Technical Skills}}
\begin{{itemize}}[leftmargin=0.15in,label={{}}]
  \small{{\item{{\textbf{{Skills}}{{: {skills_str}}}}}}}
\end{{itemize}}{hobbies_section}
\end{{document}}
"""


# ── DB REPOSITORY ─────────────────────────────────────────────────
class ResumeRepository:
    """All SQL lives here, always parameterized (psycopg2 %s placeholders,
    never string-formatted SQL — that's what actually prevents injection;
    the important fix here is centralizing it so nobody 'accidentally'
    builds a query with an f-string later)."""

    def __init__(self, db):
        self.db = db

    def save(self, result_id, session_id, input_data, output_data, ats_score, job_category):
        with self.db.cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO resumes (id, session_id, input_data, output_data, ats_score, job_category) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (result_id, session_id, json.dumps(input_data), json.dumps(output_data),
                 ats_score, job_category),
            )

    def get(self, result_id):
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id, session_id, output_data FROM resumes WHERE id = %s", (result_id,)
            )
            return cur.fetchone()

    def update_output(self, result_id, output_data, ats_score):
        with self.db.cursor(commit=True) as cur:
            cur.execute(
                "UPDATE resumes SET output_data = %s, ats_score = %s WHERE id = %s",
                (json.dumps(output_data), ats_score, result_id),
            )

    def list_for_session(self, session_id):
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id, ats_score, job_category, created_at FROM resumes "
                "WHERE session_id = %s ORDER BY created_at DESC",
                (session_id,),
            )
            return cur.fetchall()

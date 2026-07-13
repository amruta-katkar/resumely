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
    """Each template is explicitly modeled on a real, named resume
    template rather than being an arbitrary color swap. `layout` picks
    which PDF/LaTeX builder method renders it, `source` is shown in the
    UI so it's clear what each one is based on:

      - "jake"       → Jake's Resume Template (Jake Gutierrez / Sourabh
                        Bajaj) — the most-used ATS-safe one-column
                        template on Overleaf. Section rules, tabular
                        subheadings.
      - "faangpath"  → FAANGPath Simple Template — even more stripped
                        down than Jake's: no rules, no small-caps, just
                        bold section labels and tight spacing. Popular
                        for FAANG-style ATS screens.
      - "sidebar"    → Deedy-CV / Awesome-CV style — two-column, colored
                        left rail for contact/skills/education, main
                        column for experience/projects.
      - "academic"   → Academic CV style — centered serif header,
                        education/publications-first, no color blocks.
    """

    TEMPLATES = {
        "classic":    {"name": "Classic",    "cat": "All",         "desc": "Minimal, no-frills, maximum ATS parsing",   "source": "FAANGPath Simple Template", "color": "#1D4ED8", "layout": "faangpath", "font": "sans",  "lead_section": "experience"},
        "modern":     {"name": "Modern",     "cat": "Engineering", "desc": "Two-column with skills sidebar",            "source": "Deedy-CV / Awesome-CV",     "color": "#0F766E", "layout": "sidebar",   "font": "sans",  "lead_section": "skills"},
        "minimal":    {"name": "Minimal",    "cat": "Design",      "desc": "Ultra-clean, whitespace focused",           "source": "FAANGPath Simple Template", "color": "#374151", "layout": "faangpath", "font": "sans",  "lead_section": "experience"},
        "executive":  {"name": "Executive",  "cat": "Management",  "desc": "Formal, suited for senior roles",           "source": "Jake's Resume Template",    "color": "#1F2937", "layout": "jake",      "font": "serif", "lead_section": "experience"},
        "academic":   {"name": "Academic",   "cat": "Academia",    "desc": "CV-style with publications",                "source": "Academic CV",               "color": "#4338CA", "layout": "academic",  "font": "serif", "lead_section": "education"},
        "finance":    {"name": "Finance",    "cat": "Finance",     "desc": "Metrics-focused, conservative",             "source": "Jake's Resume Template",    "color": "#1D4ED8", "layout": "jake",      "font": "serif", "lead_section": "experience"},
        "government": {"name": "Government", "cat": "Government",  "desc": "GS-compatible, detailed format",            "source": "Jake's Resume Template",    "color": "#1E40AF", "layout": "jake",      "font": "serif", "lead_section": "experience"},
        "creative":   {"name": "Creative",   "cat": "Design",      "desc": "Portfolio-style layout",                    "source": "Deedy-CV / Awesome-CV",     "color": "#7C3AED", "layout": "sidebar",   "font": "sans",  "lead_section": "projects"},
        "tech":       {"name": "Tech",       "cat": "Engineering", "desc": "The standard SWE/tech resume format",       "source": "Jake's Resume Template",    "color": "#0369A1", "layout": "jake",      "font": "sans",  "lead_section": "experience"},
        "healthcare": {"name": "Healthcare", "cat": "Healthcare",  "desc": "Certifications prominently displayed",      "source": "Academic CV",               "color": "#0E7490", "layout": "academic",  "font": "serif", "lead_section": "education"},
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
    """Builds a genuinely different page layout per template family,
    instead of just recoloring the same single column.

    - "jake"/"academic" render from the AI-generated ATS-safe plain
      text with rules/section styling (reliable, works for any content
      shape).
    - "faangpath" renders the same plain text but stripped of every
      visual flourish: no colored rule under section headers, no color
      on the name, tighter line spacing — matching the real FAANGPath
      Simple Template philosophy of "nothing an ATS parser could trip on".
    - "sidebar" renders from the structured data (parsed_resume +
      tailored_experience + selected_projects) into a real two-column
      table: narrow left rail for contact/skills/education, wide right
      column for experience/projects — the Deedy-CV style layout you'd
      find as a two-column template on Overleaf.
    """

    def __init__(self, template_manager: TemplateManager):
        self.templates = template_manager

    def build(self, result, template_id, contact_line=""):
        if not PDF_GEN:
            return None, "PDF generation isn't available on the server (reportlab missing)."
        tpl = self.templates.get(template_id)
        try:
            if tpl["layout"] == "sidebar":
                return self._build_sidebar(result, tpl), None
            return self._build_flowing(result.get("resume_text", ""), tpl, contact_line), None
        except Exception as e:
            logger.error("PDF gen failed: %s", e)
            return None, "Couldn't generate the PDF. Please try again."

    # ---- shared style helpers -----------------------------------
    def _fonts(self, tpl):
        return ("Times-Bold", "Times-Roman") if tpl["font"] == "serif" else ("Helvetica-Bold", "Helvetica")

    # ---- "jake" / "faangpath" / "academic": one flowing column from resume_text ----
    def _build_flowing(self, resume_text, tpl, contact_line):
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=letter,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        )
        is_faangpath = tpl["layout"] == "faangpath"
        primary = HexColor('#0F172A') if is_faangpath else HexColor(tpl["color"])
        bold_font, reg_font = self._fonts(tpl)
        serif = tpl["font"] == "serif"

        styles = getSampleStyleSheet()
        name_style = ParagraphStyle('N', parent=styles['Normal'], fontSize=17 if is_faangpath else (19 if not serif else 20),
                                     fontName=bold_font, textColor=HexColor('#0F172A'), spaceAfter=2,
                                     alignment=1 if tpl["layout"] == "academic" else 0)
        contact_style = ParagraphStyle('C', parent=styles['Normal'], fontSize=9,
                                        fontName=reg_font, textColor=HexColor('#64748B'), spaceAfter=10,
                                        alignment=1 if tpl["layout"] == "academic" else 0)
        section_style = ParagraphStyle('S', parent=styles['Normal'], fontSize=11 if is_faangpath else 11.5,
                                        fontName=bold_font, textColor=primary,
                                        spaceBefore=10 if is_faangpath else 12, spaceAfter=3 if is_faangpath else 4)
        body_style = ParagraphStyle('B', parent=styles['Normal'], fontSize=9.5,
                                     fontName=reg_font, textColor=HexColor('#334155'),
                                     spaceAfter=1 if is_faangpath else 2, leading=13 if is_faangpath else 14)

        story = []
        first = True
        for line in (resume_text or "").split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 3 if is_faangpath else 4))
                continue
            if first:
                story.append(Paragraph(line, name_style))
                if contact_line:
                    story.append(Paragraph(contact_line, contact_style))
                if tpl["layout"] == "academic":
                    story.append(HRFlowable(width="100%", thickness=1.2, color=primary, spaceAfter=8))
                first = False
            elif line.isupper() and len(line) > 2:
                story.append(Spacer(1, 5 if is_faangpath else 6))
                story.append(Paragraph(line, section_style))
                # FAANGPath's whole point is zero visual decoration beyond
                # a bold label — no rule line under the header.
                if tpl["layout"] not in ("faangpath",):
                    story.append(HRFlowable(width="100%", thickness=0.8, color=primary, spaceAfter=4))
            elif line.startswith(('•', '▸', '-', '*')):
                story.append(Paragraph(f"• {line.lstrip('•▸-* ').strip()}", body_style))
            else:
                story.append(Paragraph(line, body_style))
        doc.build(story)
        buf.seek(0)
        return buf

    # ---- "sidebar": real two-column table layout -----------------
    def _build_sidebar(self, result, tpl):
        from reportlab.platypus import Table, TableStyle

        pr = result.get("parsed_resume") or {}
        exp = result.get("tailored_experience") or pr.get("experience", [])
        proj = result.get("selected_projects") or pr.get("projects", [])
        primary = HexColor(tpl["color"])
        bold_font, reg_font = self._fonts(tpl)

        styles = getSampleStyleSheet()
        name_style = ParagraphStyle('N', fontSize=20, fontName=bold_font, textColor=HexColor('#FFFFFF'), spaceAfter=4, leading=24)
        role_style = ParagraphStyle('R', fontSize=10.5, fontName=reg_font, textColor=HexColor('#E2E8F0'), spaceAfter=0)
        side_head = ParagraphStyle('SH', fontSize=10.5, fontName=bold_font, textColor=HexColor('#FFFFFF'), spaceBefore=14, spaceAfter=6)
        side_body = ParagraphStyle('SB', fontSize=9, fontName=reg_font, textColor=HexColor('#E2E8F0'), leading=13, spaceAfter=3)
        main_head = ParagraphStyle('MH', fontSize=12, fontName=bold_font, textColor=primary, spaceBefore=10, spaceAfter=6)
        main_sub = ParagraphStyle('MS', fontSize=10, fontName=bold_font, textColor=HexColor('#0F172A'), spaceAfter=0)
        main_meta = ParagraphStyle('MM', fontSize=8.5, fontName=reg_font, textColor=HexColor('#64748B'), spaceAfter=3)
        main_body = ParagraphStyle('MB', fontSize=9, fontName=reg_font, textColor=HexColor('#334155'), leading=13, spaceAfter=6)

        # -- left sidebar content --
        left = [Paragraph(pr.get("name", "Your Name"), name_style)]
        summary = pr.get("summary")
        if summary:
            left.append(Paragraph(summary[:140], role_style))

        left.append(Paragraph("CONTACT", side_head))
        for v in [pr.get("email"), pr.get("phone"), pr.get("location")]:
            if v:
                left.append(Paragraph(v, side_body))
        for label, v in [("LinkedIn", pr.get("linkedin")), ("GitHub", pr.get("github")), ("Portfolio", pr.get("portfolio"))]:
            if v:
                left.append(Paragraph(label, side_body))

        if pr.get("skills"):
            left.append(Paragraph("SKILLS", side_head))
            for s in pr["skills"][:16]:
                left.append(Paragraph(f"• {s}", side_body))

        if pr.get("education"):
            left.append(Paragraph("EDUCATION", side_head))
            for e in pr["education"]:
                left.append(Paragraph(e.get("degree", ""), side_body))
                left.append(Paragraph(f"{e.get('institution','')} · {e.get('year','')}", side_body))

        if pr.get("hobbies"):
            left.append(Paragraph("INTERESTS", side_head))
            left.append(Paragraph(", ".join(pr["hobbies"]), side_body))

        # -- right main column content --
        right = []
        if exp:
            right.append(Paragraph("EXPERIENCE", main_head))
            for e in exp:
                right.append(Paragraph(f"{e.get('role','')} — {e.get('company','')}", main_sub))
                right.append(Paragraph(e.get("duration", ""), main_meta))
                for b in e.get("bullets", []):
                    right.append(Paragraph(f"• {b}", main_body))
        if proj:
            right.append(Paragraph("PROJECTS", main_head))
            for p in proj:
                tech = ", ".join(p.get("tech", []))
                right.append(Paragraph(p.get("name", ""), main_sub))
                if tech:
                    right.append(Paragraph(tech, main_meta))
                for b in p.get("bullets", [p.get("description", "")]):
                    if b:
                        right.append(Paragraph(f"• {b}", main_body))

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0, rightMargin=0, topMargin=0, bottomMargin=0)

        table = Table([[left, right]], colWidths=[2.3 * inch, 6.2 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), primary),
            ('LEFTPADDING', (0, 0), (0, 0), 20), ('RIGHTPADDING', (0, 0), (0, 0), 16),
            ('LEFTPADDING', (1, 0), (1, 0), 24), ('RIGHTPADDING', (1, 0), (1, 0), 24),
            ('TOPPADDING', (0, 0), (-1, -1), 28), ('BOTTOMPADDING', (0, 0), (-1, -1), 28),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        doc.build([table])
        buf.seek(0)
        return buf


# ── LATEX BUILDER ─────────────────────────────────────────────────
class LatexBuilder:
    """Produces a genuinely different .tex skeleton per layout family,
    not just a recolored copy of the same template:

      - "jake"       → Jake's Resume Template (the most common ATS-safe
                        one-column template on Overleaf).
      - "faangpath"  → FAANGPath Simple Template (even more stripped
                        down — no rules, no small-caps, bold labels only).
      - "sidebar"    → a two-column layout built with minipages (the
                        Deedy-CV / Awesome-CV style — colored left rail
                        for contact/skills/education, main column for
                        experience/projects).
      - "academic"   → a centered, serif CV layout (education-first, no
                        color blocks) — the common "Academic CV" style.
    """

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

    def build(self, parsed_resume, tailored_exp=None, selected_proj=None, layout="jake", accent="#1D4ED8"):
        r = parsed_resume or {}
        exp = tailored_exp or r.get("experience", [])
        proj = selected_proj or r.get("projects", [])
        if layout == "sidebar":
            return self._build_sidebar(r, exp, proj, accent)
        if layout == "academic":
            return self._build_academic(r, exp, proj, accent)
        if layout == "faangpath":
            return self._build_faangpath(r, exp, proj, accent)
        return self._build_jake(r, exp, proj, accent)

    def _contact_line(self, r, sep=" $|$ "):
        esc = self._esc
        parts = []
        if r.get("phone"):
            parts.append(esc(r["phone"]))
        if r.get("location"):
            parts.append(esc(r["location"]))
        if r.get("email"):
            parts.append(f"\\href{{mailto:{esc(r['email'])}}}{{\\underline{{{esc(r['email'])}}}}}")
        if r.get("linkedin"):
            parts.append(f"\\href{{{esc(r['linkedin'])}}}{{\\underline{{LinkedIn}}}}")
        if r.get("github"):
            parts.append(f"\\href{{{esc(r['github'])}}}{{\\underline{{GitHub}}}}")
        if r.get("portfolio"):
            parts.append(f"\\href{{{esc(r['portfolio'])}}}{{\\underline{{Portfolio}}}}")
        return sep.join(parts)

    # ---------------------------------------------------------------
    # "jake" — Jake's Resume Template
    # ---------------------------------------------------------------
    def _build_jake(self, r, exp, proj, accent):
        esc = self._esc
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

        contact_line = self._contact_line(r)
        hobbies_section = ""
        if hobbies_str:
            hobbies_section = f"""
\\section{{Interests}}
\\begin{{itemize}}[leftmargin=0.15in,label={{}}]
  \\small{{\\item{{{hobbies_str}}}}}
\\end{{itemize}}"""

        return rf"""
%-- Resumely LaTeX Export (Classic — Jake's Resume Template, ATS-Safe) --%
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

    # ---------------------------------------------------------------
    # "faangpath" — FAANGPath Simple Template
    # ---------------------------------------------------------------
    def _build_faangpath(self, r, exp, proj, accent):
        esc = self._esc
        skills_str = ", ".join(esc(s) for s in r.get("skills", []))
        hobbies_str = ", ".join(esc(h) for h in r.get("hobbies", []))

        exp_blocks = ""
        for e in exp:
            bullets = "\n".join(f"    \\item {esc(b)}" for b in e.get("bullets", []))
            exp_blocks += f"""
\\textbf{{{esc(e.get('role',''))}}}, {esc(e.get('company',''))} \\hfill {esc(e.get('duration',''))} \\\\
\\vspace{{-4pt}}
\\begin{{itemize}}
{bullets}
\\end{{itemize}}
"""

        proj_blocks = ""
        for p in proj:
            tech = ", ".join(esc(t) for t in p.get("tech", []))
            desc = esc(p.get("description", "") or (p.get("bullets", [""])[0] if p.get("bullets") else ""))
            proj_blocks += f"""
\\textbf{{{esc(p.get('name',''))}}}{f' — {tech}' if tech else ''} \\\\
\\vspace{{-4pt}}
\\begin{{itemize}}
    \\item {desc}
\\end{{itemize}}
"""

        edu_blocks = ""
        for e in r.get("education", []):
            edu_blocks += f"\\textbf{{{esc(e.get('institution',''))}}} \\hfill {esc(e.get('year',''))} \\\\ {esc(e.get('degree',''))} \\\\[4pt]\n"

        contact_line = self._contact_line(r)
        hobbies_line = f"\n\\textbf{{Interests}} \\\\ {hobbies_str} \\\\[4pt]\n" if hobbies_str else ""

        # No colored rules, no small-caps, no boxed headers — FAANGPath's
        # whole premise is that anything decorative is a risk for an ATS
        # parser, so the only formatting signal is bold section labels.
        return rf"""
%-- Resumely LaTeX Export (Classic — FAANGPath Simple Template, ATS-Safe) --%
\documentclass[a4paper,11pt]{{article}}
\usepackage[margin=0.85in]{{geometry}}
\usepackage{{enumitem}}\usepackage[hidelinks]{{hyperref}}\usepackage{{titlesec}}
\setlist[itemize]{{leftmargin=16pt,itemsep=1pt,topsep=2pt}}
\titleformat{{\section}}{{\bfseries\large}}{{}}{{0em}}{{}}
\titlespacing*{{\section}}{{0pt}}{{10pt}}{{4pt}}
\pagestyle{{empty}}\setlength{{\parindent}}{{0pt}}

\begin{{document}}

{{\Large \textbf{{{esc(r.get('name','Your Name'))}}}}} \\[2pt]
{contact_line} \\[6pt]

\section{{Education}}
{edu_blocks}
\section{{Experience}}
{exp_blocks}
\section{{Projects}}
{proj_blocks}
\section{{Skills}}
{skills_str}
{hobbies_line}
\end{{document}}
"""

    # ---------------------------------------------------------------
    # "sidebar" — two-column, colored left rail (Deedy-CV / Awesome-CV style)
    # ---------------------------------------------------------------
    def _build_sidebar(self, r, exp, proj, accent):
        esc = self._esc
        hex_color = accent.lstrip("#").upper()

        skills_items = "\n".join(f"\\cvskill{{{esc(s)}}}" for s in r.get("skills", []))
        edu_items = "\n".join(
            f"\\cvsidesub{{{esc(e.get('degree',''))}}}{{{esc(e.get('institution',''))} $\\cdot$ {esc(e.get('year',''))}}}"
            for e in r.get("education", [])
        )
        contact_items = "\n".join(
            f"\\cvcontact{{{esc(v)}}}" for v in
            [r.get("email"), r.get("phone"), r.get("location"), r.get("linkedin"), r.get("github"), r.get("portfolio")]
            if v
        )
        hobbies_line = ", ".join(esc(h) for h in r.get("hobbies", []))

        exp_blocks = ""
        for e in exp:
            bullets = "\n".join(f"\\cvitem{{{esc(b)}}}" for b in e.get("bullets", []))
            exp_blocks += f"""
\\cvmainsub{{{esc(e.get('role',''))}}}{{{esc(e.get('duration',''))}}}{{{esc(e.get('company',''))}}}
{bullets}
"""
        proj_blocks = ""
        for p in proj:
            tech = ", ".join(esc(t) for t in p.get("tech", []))
            desc = esc(p.get("description", "") or (p.get("bullets", [""])[0] if p.get("bullets") else ""))
            proj_blocks += f"""
\\cvmainsub{{{esc(p.get('name',''))}}}{{}}{{{tech}}}
\\cvitem{{{desc}}}
"""

        hobbies_block = f"\\cvsidehead{{Interests}}\\cvcontact{{{hobbies_line}}}" if hobbies_line else ""

        return rf"""
%-- Resumely LaTeX Export (Modern — two-column sidebar layout, ATS-Safe) --%
\documentclass[10pt,letterpaper]{{article}}
\usepackage[left=0cm,right=0cm,top=0cm,bottom=0cm]{{geometry}}
\usepackage{{xcolor}}\usepackage{{paracol}}\usepackage[hidelinks]{{hyperref}}
\usepackage{{enumitem}}\usepackage{{titlesec}}\usepackage{{tikz}}
\definecolor{{accent}}{{HTML}}{{{hex_color}}}
\pagestyle{{empty}}\setlength{{\parindent}}{{0pt}}

\newcommand{{\cvcontact}}[1]{{{{\color{{white}}\footnotesize #1}}\\[3pt]}}
\newcommand{{\cvskill}}[1]{{{{\color{{white}}\footnotesize\textbullet\ #1}}\\[3pt]}}
\newcommand{{\cvsidesub}}[2]{{{{\color{{white}}\small\textbf{{#1}}}}\\{{\color{{gray!30}}\footnotesize #2}}\\[6pt]}}
\newcommand{{\cvsidehead}}[1]{{\vspace{{10pt}}{{\color{{white}}\large\textbf{{#1}}}}\\[2pt]{{\color{{white!60}}\rule{{2.4cm}}{{0.6pt}}}}\\[6pt]}}
\newcommand{{\cvmainhead}}[1]{{\vspace{{8pt}}{{\color{{accent}}\Large\textbf{{#1}}}}\\{{\color{{accent!40}}\rule{{\linewidth}}{{0.8pt}}}}\\[4pt]}}
\newcommand{{\cvmainsub}}[3]{{\textbf{{#1}} \hfill {{\footnotesize\color{{gray}}#2}}\\{{\footnotesize\color{{gray}}#3}}\\[2pt]}}
\newcommand{{\cvitem}}[1]{{\begin{{itemize}}[leftmargin=10pt,itemsep=1pt,topsep=1pt]\item[\textbullet]\footnotesize #1\end{{itemize}}}}

\begin{{document}}
\noindent
\begin{{tikzpicture}}[remember picture,overlay]
  \fill[accent] (current page.north west) rectangle ([xshift=6.2cm]current page.south west);
\end{{tikzpicture}}
\begin{{paracol}}{{2}}
\columnratio{{0.32}}
\begin{{leftcolumn}}
\vspace{{1.4cm}}\hspace{{0.6cm}}
\begin{{minipage}}{{5.2cm}}
{{\color{{white}}\Huge\textbf{{{esc(r.get('name','Your Name'))}}}}}\\[10pt]
\cvsidehead{{Contact}}
{contact_items}
\cvsidehead{{Skills}}
{skills_items}
\cvsidehead{{Education}}
{edu_items}
{hobbies_block}
\end{{minipage}}
\end{{leftcolumn}}
\begin{{rightcolumn}}
\vspace{{1.4cm}}\hspace{{0.4cm}}
\begin{{minipage}}{{10.5cm}}
\cvmainhead{{Experience}}
{exp_blocks}
\cvmainhead{{Projects}}
{proj_blocks}
\end{{minipage}}
\end{{rightcolumn}}
\end{{paracol}}
\end{{document}}
"""

    # ---------------------------------------------------------------
    # "academic" — centered, serif, education/publications first
    # ---------------------------------------------------------------
    def _build_academic(self, r, exp, proj, accent):
        esc = self._esc
        hex_color = accent.lstrip("#").upper()
        contact_line = self._contact_line(r, sep=" \\quad$\\vert$\\quad ")

        edu_blocks = "\n".join(
            rf"\cventry{{{esc(e.get('degree',''))}}}{{{esc(e.get('institution',''))}}}{{{esc(e.get('year',''))}}}"
            for e in r.get("education", [])
        )
        exp_blocks = ""
        for e in exp:
            bullets = "\n".join(f"  \\item {esc(b)}" for b in e.get("bullets", []))
            exp_blocks += f"""
\\cventry{{{esc(e.get('role',''))}}}{{{esc(e.get('company',''))}}}{{{esc(e.get('duration',''))}}}
\\begin{{itemize}}[leftmargin=18pt,itemsep=1pt,topsep=2pt]
{bullets}
\\end{{itemize}}
"""
        proj_blocks = ""
        for p in proj:
            desc = esc(p.get("description", "") or (p.get("bullets", [""])[0] if p.get("bullets") else ""))
            proj_blocks += f"\\cventry{{{esc(p.get('name',''))}}}{{{', '.join(esc(t) for t in p.get('tech', []))}}}{{}}\n{desc}\\\\[4pt]\n"

        skills_str = ", ".join(esc(s) for s in r.get("skills", []))

        return rf"""
%-- Resumely LaTeX Export (Academic CV — serif, education-first) --%
\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{times}}\usepackage[hidelinks]{{hyperref}}\usepackage{{enumitem}}\usepackage{{titlesec}}
\usepackage{{xcolor}}\definecolor{{accent}}{{HTML}}{{{hex_color}}}
\pagestyle{{empty}}\setlength{{\parindent}}{{0pt}}
\titleformat{{\section}}{{\centering\large\scshape\color{{accent}}}}{{}}{{0em}}{{}}[\vspace{{2pt}}{{\color{{accent}}\hrule}}\vspace{{6pt}}]
\newcommand{{\cventry}}[3]{{\textbf{{#1}} \hfill \textit{{#3}}\\#2\\[4pt]}}

\begin{{document}}
\begin{{center}}
  {{\Huge \textsc{{{esc(r.get('name','Your Name'))}}}}}\\[4pt]
  \small {contact_line}
\end{{center}}
\vspace{{6pt}}
\section{{Education}}
{edu_blocks}
\section{{Experience}}
{exp_blocks}
\section{{Projects}}
{proj_blocks}
\section{{Skills}}
\centering {skills_str}
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

    def delete(self, result_id, session_id):
        """Ownership-scoped delete — the WHERE clause on session_id means
        this can never delete a row that doesn't belong to the caller,
        even if someone guesses another user's result_id."""
        with self.db.cursor(commit=True) as cur:
            cur.execute(
                "DELETE FROM resumes WHERE id = %s AND session_id = %s RETURNING id",
                (result_id, session_id),
            )
            return cur.fetchone() is not None

    def delete_all_for_session(self, session_id):
        """Used by account deletion — wipes every saved resume for this
        user. Always scoped to session_id, never a bare DELETE."""
        with self.db.cursor(commit=True) as cur:
            cur.execute("DELETE FROM resumes WHERE session_id = %s", (session_id,))
            return cur.rowcount

    def delete_settings(self, session_id):
        with self.db.cursor(commit=True) as cur:
            cur.execute("DELETE FROM user_settings WHERE session_id = %s", (session_id,))

    def get_full_output_for_reports(self, session_id):
        """Pulls just what the ATS Reports page needs (score, category,
        date, and the ats_report json) without shipping the whole
        resume/cover-letter payload for every row."""
        with self.db.cursor() as cur:
            cur.execute(
                "SELECT id, ats_score, job_category, created_at, output_data "
                "FROM resumes WHERE session_id = %s ORDER BY created_at ASC",
                (session_id,),
            )
            return cur.fetchall()

    def get_settings(self, session_id):
        with self.db.cursor() as cur:
            cur.execute("SELECT * FROM user_settings WHERE session_id = %s", (session_id,))
            row = cur.fetchone()
        return row or {"session_id": session_id, "default_template": "classic", "email_notifications": True}

    def save_settings(self, session_id, default_template, email_notifications):
        with self.db.cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO user_settings (session_id, default_template, email_notifications, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (session_id) DO UPDATE
                SET default_template = EXCLUDED.default_template,
                    email_notifications = EXCLUDED.email_notifications,
                    updated_at = NOW()
                """,
                (session_id, default_template, email_notifications),
            )


# ── ATS REPORT AGGREGATION ─────────────────────────────────────────
class ATSReportAggregator:
    """Turns the raw list of saved resumes into the summary numbers the
    ATS Reports page shows: average/best/latest score, a score trend,
    category breakdown, and the skills that show up as "missing" most
    often across every analysis you've run."""

    def summarize(self, rows):
        if not rows:
            return {
                "count": 0, "average_score": 0, "best_score": 0, "latest_score": 0,
                "trend": [], "by_category": [], "top_missing_skills": [],
            }

        scores = [r["ats_score"] or 0 for r in rows]
        categories = {}
        missing_counter = {}

        for row in rows:
            cat = row.get("job_category") or "General"
            categories[cat] = categories.get(cat, 0) + 1

            output = row.get("output_data") or {}
            ats = output.get("ats_report", {}) if isinstance(output, dict) else {}
            for skill in ats.get("missing_skills", []) or []:
                missing_counter[skill] = missing_counter.get(skill, 0) + 1

        trend = [
            {"date": row["created_at"].strftime("%d %b"), "score": row["ats_score"] or 0}
            for row in rows[-10:]
        ]
        by_category = sorted(
            [{"category": c, "count": n} for c, n in categories.items()],
            key=lambda x: -x["count"],
        )
        top_missing = sorted(
            [{"skill": s, "count": n} for s, n in missing_counter.items()],
            key=lambda x: -x["count"],
        )[:8]

        return {
            "count": len(rows),
            "average_score": round(sum(scores) / len(scores)),
            "best_score": max(scores),
            "latest_score": scores[-1],
            "trend": trend,
            "by_category": by_category,
            "top_missing_skills": top_missing,
        }
# Resumely

AI-powered resume tailoring: paste a job description (or import a PDF resume), and Resumely
parses your resume, scores it against the job with an ATS-style breakdown, tailors your
experience/projects to close the gaps, and exports the result as a plain-text resume, a
formatted PDF, or LaTeX source ready for Overleaf.

## Stack

- **Backend:** Flask (blueprints, class-based services)
- **Auth:** Firebase Authentication (email/password + Google), verified server-side with the
  Firebase Admin SDK
- **Database:** Postgres (Supabase), accessed via a pooled `psycopg2` connection
- **AI:** Google Gemini (`google-genai`), with an automatic model-tier fallback chain
- **PDF:** ReportLab
- **PDF text extraction:** pdfplumber

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env      # fill in real values — see below
python app.py
```

Required environment variables (see `.env.example`):

| Variable | Purpose |
|---|---|
| `FLASK_SECRET_KEY` | Session signing key. App refuses to boot in production without one. |
| `DATABASE_URL` | Postgres/Supabase connection string. |
| `GEMINI_API_KEY` | Google Gemini API key. |
| `FIREBASE_CRED_PATH` | Path to the Firebase Admin service account JSON. Never commit this file. |
| `FIREBASE_API_KEY` / `FIREBASE_AUTH_DOMAIN` / `FIREBASE_PROJECT_ID` / `FIREBASE_STORAGE_BUCKET` / `FIREBASE_MESSAGING_SENDER_ID` / `FIREBASE_APP_ID` | Firebase **client** config (safe to expose — these aren't secrets, just project identifiers). |
| `APP_URL` | Only needed in production; used to keep a free-tier host from idling out. |

Tables (`resumes`, `user_settings`) are created automatically on boot if they don't exist
(`Database.init_schema()` in `extensions.py`) — no manual migration step needed for a fresh
Supabase project.

## Project structure

```
app.py                    → app factory: wires config + services, starts keep-alive
config.py                 → Config: loads + validates env vars
extensions.py              → Database (pool), FirebaseService, CSRFProtect
auth/routes.py             → AuthService + auth blueprint (login/signup/session/logout)
resume/services.py         → GeminiClient, ResumeAI, PdfTextExtractor, TemplateManager,
                              PdfBuilder, LatexBuilder, ResumeRepository,
                              ATSReportAggregator, RateLimiter
resume/routes.py           → resume blueprint — thin, delegates to the services above
templates/auth/            → landing page, login, signup
templates/resume/          → dashboard, analysis form, result, saved resumes, ATS reports,
                              settings, error page, shared partials (_sidebar, _footer,
                              _firebase_logout)
static/css/                → per-page stylesheets
```

## Features

**Auth**
- Firebase handles credentials (password hashing, Google OAuth); the backend only ever sees a
  signed ID token, verified server-side before a Flask session is created.
- Session cookie is httponly, samesite=Lax, secure in production.
- CSRF token issued per session, required on every state-changing request (form field for
  normal POSTs, `X-CSRF-Token` header for `fetch()` calls).

**Resume analysis** (`/analysis` → `/generate`)
- Optional PDF import: upload an existing resume and it autofills the form (name, email,
  phone, location, LinkedIn, GitHub, portfolio, hobbies, skills, projects, experience).
- Client-side validation before submit — required fields, minimum lengths, email format, and
  a check that a custom section has both a title and content (or neither) — so a bad
  submission never reaches the server and the loading screen is never shown for something a
  simple check could have caught.
- Server-side: Gemini parses the resume, analyzes the job description, scores the match
  (skill match / keyword coverage / experience alignment / section completeness), and
  tailors experience bullets + selects the most relevant projects — never inventing
  experience, only rephrasing and reordering what's actually there.
- "Skills you have but didn't list" — the AI infers skills implied by your projects/experience
  and lets you one-click-confirm them on the result page, which bumps your score and re-saves.

**Templates** — each of the 10 templates is modeled on a real, named resume template, not an
arbitrary color variant. Both the PDF and the LaTeX export are structurally different per
family, not just recolored:

| Template | Based on | Layout |
|---|---|---|
| Classic, Minimal | **FAANGPath Simple Template** | single column, no rules/color — nothing an ATS parser could trip on |
| Tech, Executive, Finance, Government | **Jake's Resume Template** | single column, tabular subheadings, section rules |
| Modern, Creative | **Deedy-CV / Awesome-CV** | two-column, colored sidebar for contact/skills/education |
| Academic, Healthcare | **Academic CV** | centered serif, education-first, no color blocks |

The template auto-recommends based on the job's detected category, or you can pick manually;
your last choice is remembered as a default in Settings.

**Saved Resumes** (`/saved-resumes`) — every analysis you've run, with client-side
search-by-category, sort by date/score, inline PDF download, and delete (ownership-checked,
CSRF-protected).

**ATS Reports** (`/ats-reports`) — average/best/latest score, a trend chart across your last
10 analyses, a breakdown by job category, and the skills that show up as "missing" most often
across every analysis you've run — all computed server-side in `ATSReportAggregator`, no
charting library needed.

**Settings** (`/settings`)
- Default template preference (persisted, pre-selects on the analysis form).
- Email notification toggle.
- **Delete account permanently** — type `DELETE` to confirm, then it deletes the Firebase
  Auth account first (if that fails, nothing else is touched, avoiding a half-deleted state),
  then wipes every saved resume and the settings row, then clears the session. No recovery,
  by design — this is a real destructive action, not a soft-delete.

**Exports** — plain ATS-safe text (shown inline, copyable), a formatted PDF, and a `.tex` file
matching the chosen template's real LaTeX skeleton, ready to paste into Overleaf.

## Security notes

- No hardcoded secrets — everything comes from environment variables, and the app won't boot
  in production without the required ones set.
- Firebase Admin SDK is properly initialized (a state that's easy to get subtly wrong — an
  uninitialized app makes every `verify_id_token()` call silently fail).
- `verify_id_token` is called with `clock_skew_seconds=10` to tolerate normal dev-machine
  clock drift without rejecting valid tokens.
- Every resource lookup (`/result/<id>`, `/update-skills/<id>`, `/export/*`, delete) checks
  `row.session_id == session["user_id"]` — ownership, not just authentication.
- All SQL is parameterized (`psycopg2` `%s` placeholders) via a single `ResumeRepository`,
  never string-built.
- Rate limiting on generation, PDF parsing, and skill updates (in-memory — see limitations).
- No PII logged (session contents / raw form data are never printed).
- Security headers set on every response (`X-Content-Type-Options`, `X-Frame-Options`,
  `Referrer-Policy`, no-cache on auth pages).

## Known limitations

- **In-memory rate limiter** — resets on restart, and each gunicorn worker enforces its own
  independent limit if you run more than one process. Move to Redis before scaling past a
  single instance.
- **No automated tests yet.** The service classes (`ResumeAI`, `LatexBuilder`,
  `TemplateManager`, `ATSReportAggregator`) are pure and mockable, so this is cheap to add —
  just not done yet.
- **No schema validation/retry on Gemini's JSON output** — a malformed response degrades to a
  shown error rather than a retry-with-correction loop.
- **Single AI provider** — if all Gemini model tiers are down or rate-limited, generation
  fails entirely; no fallback provider or caching.
- Email verification is not currently enforced at login (kept simple deliberately — enabling
  it requires wiring `sendEmailVerification()` into the signup flow and a resend path on
  login; ask if you want this built out).

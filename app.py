from flask import Flask, request, render_template, session
import os
from dotenv import load_dotenv
from google import genai
import psycopg2
import json
import uuid

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
app.secret_key = os.getenv("SECRET_KEY")

@app.before_request
def set_session():
    if "user_id" not in session:
        session["user_id"] = str(uuid.uuid4())

def get_db_connection():
    url = os.getenv("DATABASE_URL")

    # Fix for Render postgres URL issue
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgres://", 1)

    return psycopg2.connect(url)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    skills = request.form.get("skills")
    projects = request.form.get("projects")
    experience = request.form.get("experience")
    job_description = request.form.get("job_description")

    prompt = f"""
    You are an expert recruiter.

    Skills: {skills}
    Projects: {projects}
    Experience: {experience}
    Job Description: {job_description}

    1. Select best projects
    2. Rewrite them
    3. Generate ATS resume
    4. Generate cover letter
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    session_id = session.get("user_id")

    input_data = {
        "skills": skills,
        "projects": projects,
        "experience": experience,
        "job_description": job_description
    }

    output_data = {
        "result": response.text
    }

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO resumes (session_id, input_data, output_data)
        VALUES (%s, %s, %s)
        """,
        (session_id, json.dumps(input_data), json.dumps(output_data))
    )

    conn.commit()
    cur.close()
    conn.close()

    return render_template("index.html", result=response.text)

@app.route("/test")
def test():
    prompt = "Generate a short resume for Python developer"

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    ) 

    return response.text

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
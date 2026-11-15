"""Synthetic DEMO data generator.

!!! EVERYTHING GENERATED HERE IS FAKE !!!
It exists only so the app can be demonstrated without the Kaggle file. The data
has the same columns as the LinkedIn dataset, and a few skill trends are
PLANTED ON PURPOSE (e.g. "Generative AI" rises, "jQuery" falls) so you can see
what the trend pages look like. The app labels all demo results as DEMO.
Real results only ever come from your real CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import SAMPLE_DIR

DEMO_CSV = SAMPLE_DIR / "demo_postings.csv"

ROLES = {
    "Data Scientist": (["Python", "Pandas", "Machine Learning", "Statistics", "SQL", "scikit-learn", "NumPy", "Data Visualization", "A/B Testing"], 115000),
    "Machine Learning Engineer": (["Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Docker", "MLOps", "AWS", "Model Deployment", "NLP"], 135000),
    "Data Analyst": (["SQL", "Excel", "Tableau", "Power BI", "Statistics", "Python", "Data Visualization", "Business Intelligence"], 80000),
    "Data Engineer": (["Python", "SQL", "Spark", "Airflow", "ETL", "AWS", "Kafka", "Snowflake", "Data Warehousing", "Hadoop"], 120000),
    "Software Engineer": (["Java", "Python", "Git", "REST APIs", "Microservices", "Docker", "SQL", "AWS", "Data Structures & Algorithms", "Agile"], 125000),
    "Frontend Developer": (["JavaScript", "React", "HTML", "CSS", "TypeScript", "Git", "Redux", "Angular", "jQuery"], 100000),
    "DevOps Engineer": (["Docker", "Kubernetes", "Terraform", "AWS", "Linux", "CI/CD", "Jenkins", "Ansible", "Git", "Bash/Shell"], 128000),
    "Cybersecurity Analyst": (["Cybersecurity", "SIEM", "Network Security", "Penetration Testing", "Linux", "IAM", "Encryption", "Networking"], 105000),
    "QA Automation Engineer": (["Selenium", "Test Automation", "Python", "Java", "JIRA", "CI/CD", "Unit Testing"], 90000),
    "Mobile Developer": (["Android", "iOS", "Kotlin", "Swift", "Flutter", "React Native", "Git"], 112000),
}
OTHER_TITLES = ["Registered Nurse", "Sales Associate", "Accountant", "Warehouse Associate", "High School Teacher", "Store Manager"]

# PLANTED trends (synthetic): skill -> roles that may list it
RISING = {"Generative AI": ["Data Scientist", "Machine Learning Engineer", "Software Engineer"],
          "LLMs": ["Machine Learning Engineer", "Data Scientist"],
          "RAG": ["Machine Learning Engineer"], "LangChain": ["Machine Learning Engineer", "Software Engineer"],
          "Prompt Engineering": ["Data Scientist", "Software Engineer"],
          "MLOps": ["Machine Learning Engineer", "DevOps Engineer"],
          "Kubernetes": ["DevOps Engineer", "Software Engineer"], "Databricks": ["Data Engineer"]}
FALLING = {"jQuery": ["Frontend Developer", "Software Engineer"], "AngularJS": ["Frontend Developer"],
           "Hadoop": ["Data Engineer"], "SVN": ["Software Engineer", "QA Automation Engineer"],
           "Perl": ["Software Engineer", "DevOps Engineer"], "Mainframe": ["Software Engineer"]}
JUNIOR_EXTRA = ["Python", "SQL", "Excel", "HTML", "CSS", "Git", "Statistics"]
SENIOR_EXTRA = ["Microservices", "System Design", "Terraform", "Kubernetes", "Data Warehousing", "Agile"]
PHRASE = {"Bash/Shell": "Bash scripting", "Gradient Boosting": "XGBoost", "SIEM": "Splunk", "Spring": "Spring Boot",
          "Sass/SCSS": "SCSS", "OpenAI/GPT": "OpenAI", "Flash/ActionScript": "ActionScript"}
POOL = sorted({s for core, _ in ROLES.values() for s in core} | set(RISING) | set(FALLING) | set(JUNIOR_EXTRA) | set(SENIOR_EXTRA))
LEVELS = ["Entry level", "Associate", "Mid-Senior level", "Director"]
FILLER = ["We offer competitive benefits and a flexible, inclusive culture.",
          "You will collaborate with a friendly cross-functional team.",
          "We are an equal opportunity employer.",
          "Strong communication and problem-solving skills are valued."]


def _phrase(skill: str) -> str:
    return PHRASE.get(skill, skill)


def generate_demo_dataframe(n: int = 6000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    role_names = list(ROLES)
    rows = []
    for i in range(n):
        tech = rng.random() < 0.82
        t_frac = rng.random()                                   # position in the year (0..1)
        level = rng.choice(LEVELS, p=[0.25, 0.20, 0.45, 0.10])
        if tech:
            role = role_names[rng.integers(len(role_names))]
            core, base = ROLES[role]
            skills = {s for s in core if rng.random() < 0.55}
            skills |= set(rng.choice(POOL, size=rng.poisson(1.5)))           # noise skills
            for s, roles in RISING.items():
                if role in roles and rng.random() < 0.04 + 0.30 * t_frac:    # rises during the year
                    skills.add(s)
            for s, roles in FALLING.items():
                if role in roles and rng.random() < 0.30 - 0.25 * t_frac:    # falls during the year
                    skills.add(s)
            if level in ("Entry level", "Associate"):
                skills |= {s for s in JUNIOR_EXTRA if rng.random() < 0.15}
            else:
                skills |= {s for s in SENIOR_EXTRA if rng.random() < 0.15}
            prefix = {"Entry level": "Junior ", "Associate": "", "Mid-Senior level": "Senior ", "Director": "Lead "}[level]
            title = f"{prefix}{role}"
            sal = base * {"Entry level": 0.7, "Associate": 0.85, "Mid-Senior level": 1.1, "Director": 1.4}[level]
            sal *= 1 + 0.06 * len(skills & {"Kubernetes", "LLMs", "MLOps", "Spark"})      # planted premium
        else:
            role = "Other"
            title = str(rng.choice(OTHER_TITLES))
            skills = set()
            sal = rng.uniform(32000, 70000)
        skills = sorted(skills)
        company = f"DemoCorp {rng.integers(1, 120)}"
        mention = rng.random() < 0.4
        intro = f"<p><strong>About the role</strong></p><p>{company} is hiring a {title}.</p>" if mention else \
            f"<p><strong>About the role</strong></p><p>{company} is growing its team.</p>"
        reqs = "".join(f"<li>Experience with {_phrase(s)}</li>" for s in skills) or "<li>Relevant experience</li>"
        desc = f"{intro}<p>Requirements &amp; responsibilities:</p><ul>{reqs}</ul><p>{rng.choice(FILLER)}</p>"
        listed = start + pd.Timedelta(days=float(t_frac * 365.9))
        has_sal = rng.random() < 0.45
        rows.append({
            "job_id": 1_000_000 + i, "company_name": company, "title": title, "description": desc,
            "location": "Demo City", "formatted_experience_level": level if rng.random() < 0.9 else None,
            "formatted_work_type": "Full-time", "listed_time": int(listed.timestamp() * 1000),
            "normalized_salary": round(float(sal * rng.normal(1, 0.10)), -2) if has_sal else np.nan,
            "skills_desc": ("Skills: " + ", ".join(_phrase(s) for s in skills[:3])) if skills and rng.random() < 0.15 else None,
            "pay_period": "YEARLY" if has_sal else None,
            "demo_data_flag": "SYNTHETIC",
        })
    df = pd.DataFrame(rows)
    dups = df.sample(frac=0.01, random_state=seed)                # a few duplicates to exercise cleaning
    return pd.concat([df, dups], ignore_index=True)


def generate_demo_csv(path: Optional[Path] = None, n: int = 6000, seed: int = 7) -> Path:
    path = Path(path or DEMO_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_dataframe(n, seed).to_csv(path, index=False)
    return path


def ensure_demo_csv() -> Path:
    """Return the demo CSV path, generating it if it is missing."""
    if not DEMO_CSV.exists():
        generate_demo_csv()
    return DEMO_CSV

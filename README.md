# 🎓 AI Academic Assistant

An AI-powered academic support platform built for educators and students. Automates grading, question generation, exam paper creation, research briefs, citation formatting, and more — using a multi-provider LLM architecture with intelligent caching.

**Live:** [academicassist.com](https://academicassist.com) &nbsp;|&nbsp; **Deployed on:** Railway &nbsp;|&nbsp; **Built with:** Python, Flask, PostgreSQL

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Auto Grading** | Grade student submissions against an answer key (MCQ, short & long answer) with configurable strictness |
| **Question Generation** | Generate MCQ, True/False, and open-ended questions from uploaded PDFs or slides |
| **Exam Papers** | Generate multiple unique exam variants from lecture slides |
| **Core Concepts** | Extract key concepts and definitions from chapter documents |
| **Research Brief** | Produce structured research briefs with summary, findings, and sources |
| **Recommendation Letters** | Generate professional letters from uploaded resumes |
| **Citation Generator** | Format citations in APA, MLA, Chicago, IEEE, and Harvard styles |
| **Syllabus Analysis** | Analyse course syllabi and receive structured improvement suggestions |

---

## 🛠️ Tech Stack

**Backend**
- Python 3.11, Flask
- SQLAlchemy + PostgreSQL (Railway) / SQLite (local fallback)
- Flask-Bcrypt, Flask-WTF (CSRF), Flask-Limiter
- Gunicorn (production WSGI server)

**AI Providers**
- Groq (LLaMA 3.1) — primary grading engine
- OpenAI (GPT-4o-mini) — question generation, syllabus, exam papers
- Anthropic (Claude) — grading fallback
- Google Gemini — OCR for scanned PDFs

**Frontend**
- Jinja2 templates, vanilla JS
- Dark theme with CSS variables
- Mobile responsive

**Infrastructure**
- Railway (hosting + PostgreSQL)
- Docker (containerised, AWS/Azure ready)
- Formspree (contact + feedback forms)

---

## 🏗️ Project Structure

```
├── app.py                  # Main Flask application
├── core/
│   ├── cache.py            # AI response + grading cache (SQLite/PostgreSQL)
│   ├── docstore.py         # Document store for parsed Q&A pairs
│   ├── grading.py          # Grading orchestration
│   ├── grading_agent.py    # Multi-provider AI grading agent
│   ├── parsing.py          # Document parsing and Q&A splitting
│   └── translations.py     # EN/ES translation support
├── templates/
│   ├── base.html           # Shared base template
│   ├── home.html           # Landing page
│   ├── index.html          # Main dashboard
│   ├── login.html          # Login page
│   ├── signup.html         # Signup page with role selection
│   ├── profile.html        # User profile (edit info, email, password)
│   ├── admin.html          # Admin dashboard (restricted)
│   ├── about.html          # About page
│   ├── privacy.html        # Privacy policy (GDPR/FERPA)
│   ├── terms.html          # Terms of service
│   └── contact.html        # Contact form (Formspree)
├── static/                 # Static assets (images, icons)
├── uploads/                # Temporary file uploads
├── output/                 # Generated output files
├── temp/                   # Temporary grading files
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Local development with PostgreSQL
├── Procfile                # Railway/Heroku start command
├── requirements.txt        # Python dependencies
└── migrate.py              # Database migration script
```

---

## 🚀 Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/eli7diaz/academicassistant.git
cd academicassistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp example.env.txt .env
# Edit .env with your API keys

# Run the app
python app.py
```

Visit `http://localhost:5000`

### Using Docker (recommended)

```bash
docker-compose up --build
```

This spins up the Flask app + PostgreSQL together.

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# Flask
FLASK_SECRET_KEY=your-secret-key-here
FLASK_ENV=development

# Database (leave empty to use SQLite locally)
DATABASE_URL=

# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...

# Admin access
ADMIN_EMAIL_1=your@email.com
ADMIN_EMAIL_2=professor@university.edu

# Mail (optional)
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=your-app-password
```

---

## 🗄️ Database Schema

The app uses PostgreSQL in production and SQLite locally. Key tables:

- **user** — accounts with role, university, department, language preference
- **university** — institution registry
- **document** — metadata for every uploaded file (type, size, page count)
- **session_log** — usage log for every AI feature invocation
- **grade_cache** — cached grading results (30-day TTL)
- **ai_response_cache** — cached AI responses (30-day TTL)
- **docs / qas** — parsed document Q&A pairs

---

## 🔒 Security

- Passwords hashed with bcrypt
- CSRF protection on all forms (Flask-WTF)
- Rate limiting on `/login` (10 requests/minute)
- Session cookies: HttpOnly, SameSite=Lax, Secure in production
- No hardcoded secrets — all loaded from environment variables
- Admin dashboard restricted to authorised emails only
- Debug endpoints removed in production

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t academic-assistant .

# Run container
docker run -p 5000:5000 --env-file .env academic-assistant
```

The Docker image is production-ready and compatible with:
- AWS ECS / Elastic Beanstalk
- Azure Container Apps
- Google Cloud Run
- Railway (current deployment)

---

## 📊 Admin Dashboard

Available at `/admin` — restricted to authorised emails defined in environment variables.

Shows:
- Total users and active accounts
- Session count and cache hit rate
- Feature usage breakdown
- AI provider usage breakdown
- University-level activity
- Recent session log
- Full user table

---

## 🌐 Internationalisation

Supports English and Spanish. Language can be toggled via the topbar and is saved to the user's profile.

---

## 📄 Legal

- [Privacy Policy](https://academicassist.com/privacy) — GDPR and FERPA compliant
- [Terms of Service](https://academicassist.com/terms)
- [About](https://academicassist.com/about)

---

## 👥 Team

- **Ishan Rajvi** — Developer
- **Professor Elizabeth Diaz** — Project Supervisor, University of Texas at Arlington

---

## 📬 Contact

- General: [Contact form](https://academicassist.com/contact)
- Support: support@academicassist.com

---

*Built as part of a volunteer research project at the University of Texas at Arlington.*

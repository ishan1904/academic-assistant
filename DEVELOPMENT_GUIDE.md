# Development Guide - for the grading feature

This document provides technical guidance for contributors developing or extending the Grading Agent research project under the supervision of Professor Elizabeth Diaz, Department of Computer Science and Engineering, University of Texas at Arlington.

---

## 1. Overview

The Grading Agent is an AI-assisted system designed to automate the evaluation of mixed-format student submissions (typed, handwritten, or hybrid).  
It aligns student answers with rubric keys, analyzes content similarity, and assigns scores using both deterministic logic and language-model reasoning.

---

## 2. System Architecture

```
+---------------------------+
|        app.py             |  ← Flask application and API routes
+---------------------------+
             ↓
+---------------------------+
|     grading_agent.py      |  ← AI model interface and scoring logic
+---------------------------+
             ↓
+---------------------------+
|     split_into_qas()      |  ← Core question-answer parsing logic
+---------------------------+
             ↓
+---------------------------+
|   align_and_grade()       |  ← Alignment of student and rubric responses
+---------------------------+
             ↓
+---------------------------+
|   Groq / Gemini API       |  ← AI evaluation for feedback and scoring
+---------------------------+
```

---

## 3. Local Setup

### 3.1 Environment Configuration

```bash
git clone https://github.com/<professor-repo>/grading-agent.git
cd grading-agent

python -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, for developers
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=<your_groq_key>
GEMINI_API_KEY=<your_gemini_key>    # optional, for OCR and diagram support
```

---

## 4. Code Structure

| File / Directory | Description |
|------------------|-------------|
| `app.py` | Flask routes, file handling, synchronous and asynchronous grading endpoints. |
| `grading_agent.py` | Language-model interface, JSON parsing, scoring normalization, and rubric evaluation. |
| `split_into_qas()` | Adaptive parsing logic that separates questions and answers from extracted text. |
| `align_and_grade()` | Aligns rubric and student responses by similarity or order. |
| `templates/` | HTML templates for web pages and results. |
| `uploads/`, `output/`, `temp/` | Directories for file input/output and intermediate processing. |

---

## 5. Parsing Logic (`split_into_qas()`)

### 5.1 Function Role

Converts extracted plain text from PDF or DOCX files into structured dictionaries of questions and answers.

### 5.2 Detection Patterns

| Input Type | Example | Parser Behavior |
|-------------|----------|----------------|
| Numbered Questions | `1.`, `Q1`, `Exercise 1:` | Starts a new question block |
| Subparts | `a)`, `b)` | Groups as subsections within the same question |
| MCQ Options | `A)`, `B)`, `C)`, `D)` | Recognized as multiple-choice options |
| Answer Indicators | `Answer:` or `Ans:` | Marks beginning of answer text |
| Metadata | “Exam Key”, “Dataset” | Ignored from scoring logic |

Regular expression example:

```python
r"(?:(?<=\n)|^)(?:Q(?:uestion)?\s*\d+[\).:]|Exercise\s*\d+[\).:]|\b\d+[\).])\s+"
```

---

## 6. Grading Pipeline

1. Parse both rubric and student answers using `split_into_qas()`.
2. Align corresponding Q–A pairs.
3. Send structured prompts to Groq’s LLaMA 3.1 (and optionally Gemini).
4. Normalize, clamp, and aggregate the scores.
5. Render detailed feedback and weighted totals.

---

## 7. Debugging Guide

| Issue | Likely Cause | Recommended Action |
|-------|---------------|--------------------|
| All scores are zero | Parsing mismatch between rubric and student QAs | Check `split_into_qas` output counts. |
| Jumbled order | Unclear numbering or formatting in source file | Verify that question numbering follows a consistent pattern. |
| OCR text missing | Tesseract not installed or misconfigured | Confirm installation and PATH environment variable. |
| Grading timeout | Large file or long answers | Use `/grade_async` endpoint to run in background. |

---

## 8. Extending the System

### 8.1 Adding a New Question Type
To support a new type such as graph interpretation or mathematical derivation:

1. Add detection keywords in `detect_qtype()`.
2. Define a new scoring method (`grade_math()`, `grade_graph()`).  
3. Register it inside `process_grading()`.

### 8.2 Integrating a New Model
1. Add the model SDK to `requirements.txt`.
2. Create a wrapper, for example:

```python
def run_model(prompt, model="gemini-pro"):
    # Custom logic to invoke the new model
    ...
```
3. Update `grading_agent.py` to include the new model in its evaluation path.

---

## 9. Coding Standards

- Follow **PEP 8** conventions.  
- Format all code using `black`:
  ```bash
  black .
  ```
- Run lint checks before committing:
  ```bash
  flake8 .
  ```
- Execute tests to ensure stability:
  ```bash
  pytest
  ```

---

## 10. Collaboration Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/improve-parser
   ```
2. Commit your work with a clear message:
   ```bash
   git add .
   git commit -m "Improved question parser for hybrid exams"
   ```
3. Push and create a pull request:
   ```bash
   git push origin feature/improve-parser
   ```

---

## 11. Future Enhancements

- Integration of Gemini Vision for diagram, graph, and handwritten answer evaluation.  
- Exporting graded reports as annotated PDFs with color-coded feedback.  
- Model performance comparison dashboard (Groq vs Gemini).  
- Dataset expansion for benchmarking grading accuracy.  

---


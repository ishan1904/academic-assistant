# Grading Agent

Artificial Intelligence Assisted Grading System  
Developed under the supervision of Professor Elizabeth Diaz  
Department of Computer Science and Engineering, University of Texas at Arlington

---

## Overview

in AI Assistant project this feature is aimed at automating the grading of mixed-format student submissions, including typed, handwritten, and hybrid exam papers.  
It aligns extracted student answers with rubric keys and evaluates them using a combination of deterministic logic (for MCQs and short answers) and language-model reasoning (for descriptive and analytical questions).

The system is implemented in Python using Flask for the web interface and backend API integration with Groq’s LLaMA 3.1 model.  
Future work will extend to Gemini Vision for diagram, handwriting, and graph-based evaluation.

---


- Supports typed and OCR-based PDF submissions
- Weighted scoring with configurable grading modes (lenient, normal, strict)
- Deterministic grading for multiple-choice questions
- Context-aware parsing for any exam structure
- Detailed feedback per question with normalized 0–10 scoring
- Asynchronous background grading for large exams

---

## Installation

### Prerequisites
- Python 3.10 or later
- Tesseract OCR (for text extraction from images or scanned PDFs)

Install Tesseract:
- Ubuntu/Linux: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`
- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH



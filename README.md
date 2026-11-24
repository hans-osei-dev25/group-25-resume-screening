# Group 25 — AI-Based Resume Screening Tool for Employers

**COMP-5112-FB · Research Methodology in Computer Science · Lakehead University**

## Team

| Member | Student number |
| --- | --- |
| Hans Kwame Osei | 1372721 |
| Christian Ruado | 1301415 |

**Repository:** https://github.com/hans-osei-dev25/group-25-resume-screening

A Python and Streamlit research prototype comparing keyword-based TF-IDF matching with semantic sentence embeddings for initial resume shortlist relevance. Rankings support human review; they do not make hiring decisions.

## Features

- Paste a job description or upload TXT, DOCX or a text-based PDF.
- Screen up to 20 resumes against up to 20 equally weighted requirements.
- Compare TF-IDF and semantic scores, rankings and matching times.
- Inspect each method's best-matching resume passage for each requirement.
- Assign reviewer labels and calculate Precision@k and Recall@k.
- Export rankings, reviewer labels and evaluation metrics as CSV.

## How matching works

Resumes are split into passages of up to 80 words, with a 15-word overlap on long lines. Each requirement is compared with every resume passage. The highest passage similarity for each requirement is averaged to produce the candidate's score. Candidates are ranked separately for each method; ties use candidate ID.

- **Keyword baseline:** scikit-learn TF-IDF, word unigrams and bigrams, cosine similarity. Weights are fitted on the current batch of requirements and resumes.
- **Semantic method:** `sentence-transformers/all-MiniLM-L6-v2`, using normalized embeddings and cosine similarity on CPU. This is a pretrained sentence-transformer model; this project does not train it.

The interface labels the semantic method “BERT.” Similarity scores are not probabilities or percentages of qualifications. Scores from the two methods should not be interpreted as directly equivalent.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface and screening workflow |
| `core.py` | Text extraction, limited redaction, matching, metrics and CSV export |
| `evaluation.py` | Validation of reviewer labels and evaluation of both methods |
| `requirements.txt` | Direct Python dependencies |
| `.gitignore` | Excludes environments, secrets and generated caches from Git tracking |

All three Python source files must be in the same directory. Compiled files in `__pycache__` do not replace the source files.

## Installation — Windows / VS Code

Python 3.12 is recommended to match the development environment (3.12.10). Open a terminal in the folder containing `app.py`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal. Stop the server with **Ctrl+C**.

For an existing environment with dependencies already installed, run only the last command. No activation command is required.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Dependencies are currently unpinned; this is not a captured lockfile of the original environment. Internet access is required for package installation and the first semantic-model download. If semantic loading fails, the app displays a warning and provides TF-IDF results only. No API key or `.env` file is required by the supplied source code.

## Using the prototype

1. Enter job requirements, preferably one requirement per line. Very short fragments and headings may be skipped; inspect the requirements actually used.
2. Upload resumes and select **Screen Applicants**.
3. Review **Applicant Ranking** and **Candidate Evidence**.
4. In **Evaluate**, assign each candidate a label and a reason against the same requirements:
   - **0:** inadequate evidence.
   - **1:** partial or uncertain evidence.
   - **2:** explicit evidence for all essential requirements.
5. Select k and export the results. Only label **2** counts as relevant.

Precision@k is the number of relevant candidates in the top k divided by k. Recall@k divides that number by all relevant candidates in the uploaded batch; recall is undefined when there are no relevant candidates. With fewer than five resumes, report the actual k instead of claiming Precision@5 or Recall@5.

For a study, reviewers should label candidates independently before seeing model rankings. The current interface accepts one set of reviewer labels; it does not calculate inter-reviewer agreement. Evaluate multiple jobs and candidate batches before making general claims. No benchmark results or claim that either method is superior are included here.

## Limits and data handling

- Maximum 5 MB per uploaded file, 30 PDF pages and 50,000 extracted characters.
- Scanned PDFs and images are unsupported: there is no OCR.
- Redaction removes labelled personal fields, emails, links and phone-like numbers. It is not guaranteed anonymization and does not establish fairness or remove all bias.
- Matching may miss negation, context, or qualifications expressed in unusual ways; inspect the source evidence.
- TF-IDF scores depend on the current candidate batch. Requirements are equally weighted.
- Matching times exclude file extraction and semantic-model loading.
- The supplied application uses session memory rather than a persistent applicant database. Downloaded CSV files are saved by the user; model files may be cached locally.
- Use synthetic or appropriately consented data. Do not commit real applicant documents, credentials or private exports to the public repository.

## Demonstration video

The demonstration video will be uploaded separately. Add its final link here once available.

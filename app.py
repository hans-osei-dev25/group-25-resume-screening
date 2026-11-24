import re

import pandas as pd
import streamlit as st

from core import MODEL_NAME, clean_text, extract_text, match, safe_csv
from evaluation import evaluate_methods, validate_labels

# set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="AI Resume Screening for Employers- Prototype",page_icon='📄',
    layout="wide"
)

if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

MAX_RESUMES = 20
MAX_REQUIREMENTS = 20

# Table width option that works on both older and newer Streamlit versions
_ver = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}


# ===========================================
# Helpers
# ===========================================
@st.cache_resource(show_spinner=False)
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu", trust_remote_code=False)


def anonymize(text):
    """Limited redaction from core.py (labelled fields, emails, links, phone numbers).
    To remove names, schools or postal codes as well."""
    return clean_text(text)


def split_requirements(text):
    """Turn a pasted job description into one requirement per line (equal weight each)."""
    items, seen = [], set()
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[•·▪●◦*\-–—]|\d{1,2}[.)])\s+", "", line).strip()
        if not line:
            continue
        pieces = re.split(r"(?<=[.!?;])\s+", line) if len(line.split()) > 15 else [line]
        for piece in pieces:
            piece = piece.strip()[:400]
            if piece.endswith(":") or len(piece.split()) < 3:
                continue  # headings and fragments
            key = piece.lower()
            if key not in seen:
                seen.add(key)
                items.append(piece)
    return items


def tabulate(results):
    """One row per candidate with each method's score and rank."""
    rows = {}
    for method, key in (("keyword", "kw"), ("semantic", "sem")):
        if method in results:
            for r in results[method]["ranking"]:
                rows.setdefault(r["candidate_id"], {"cid": r["candidate_id"]}).update(
                    {f"{key}_score": r["score"], f"{key}_rank": r["rank"]})
    return sorted(rows.values(), key=lambda r: r.get("sem_rank", r["kw_rank"]))


# ===========================================
# UI
# ===========================================
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
.section-header {
    font-size: 20px;
    font-weight: 700;
    margin-top: 2px;
    margin-bottom: 1px;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Resume Screening for Employers - Prototype")
st.markdown(
    """
Compare keyword and semantic matching against the same job requirements, then evaluate the rankings with human judgments.
"""
)
st.caption("Decision support only: a human recruiter makes every hiring decision. "
           "Emails, links, phone numbers and labelled personal fields (name, address, etc.) are removed before scoring; "
           "this is limited redaction, not guaranteed anonymization. Nothing is stored.")

# ===========================================
# Job Description
# ===========================================
st.markdown("""
<div class="section-header">
Job Description
</div>
""", unsafe_allow_html=True)

jd_key = f"job_description_{st.session_state.reset_counter}"
jd_upload_key = f"jd_upload_{st.session_state.reset_counter}"


def load_jd_file():
    """Runs when a job description file is uploaded: put its extracted text in the text area."""
    f = st.session_state.get(jd_upload_key)
    if f is None:
        return
    try:
        st.session_state[jd_key] = extract_text(f.name, f.getvalue())
        st.session_state["jd_error"] = ""
    except Exception as exc:
        st.session_state["jd_error"] = f"Could not read {f.name}: {exc}"


job_description = st.text_area(
    "Paste Job Description Here",
    height=200,
    key=jd_key
)

st.file_uploader(
    "OR Upload Job Description",
    type=["txt", "pdf", "docx"],
    key=jd_upload_key,
    on_change=load_jd_file
)
st.caption("The text extracted from an uploaded file appears in the box above, where you can edit it before screening.")

if st.session_state.get("jd_error"):
    st.error(st.session_state["jd_error"])

# ===========================================
# Resume Upload
# ===========================================
st.markdown("""
<div class="section-header">
Applicant Resumes
</div>
""", unsafe_allow_html=True)

resume_files = st.file_uploader(
    "Upload Multiple Resumes",
    type=["txt", "pdf", "docx"],
    key=f"resume_upload_{st.session_state.reset_counter}",
    accept_multiple_files=True
)
st.caption(f"Up to {MAX_RESUMES} resumes · 5 MB per file · 30 PDF pages · 50,000 characters. "
           "Scanned PDFs and images are not supported (no OCR).")

col1, col2, col3 = st.columns([1, 1, 6])

with col1:
    screen = st.button("Screen Applicants")

with col2:
    clear = st.button("Clear All")

# ===========================================
# Clear Button
# ===========================================
if clear:
    st.session_state.reset_counter += 1
    st.session_state.pop("run", None)
    st.session_state.pop("jd_error", None)
    st.rerun()

# ===========================================
# Screening Button
# ===========================================
if screen:

    if not job_description.strip():
        st.error("Please provide a Job Description.")
        st.stop()

    if not resume_files:
        st.error("Please upload resumes.")
        st.stop()

    if len(resume_files) > MAX_RESUMES:
        st.error(f"Please upload no more than {MAX_RESUMES} resumes.")
        st.stop()

    criteria = split_requirements(job_description)
    if not criteria:
        st.error("No job requirements found. Add lines or sentences describing what the job needs.")
        st.stop()
    if len(criteria) > MAX_REQUIREMENTS:
        st.warning(f"The job description has {len(criteria)} requirements; only the first "
                   f"{MAX_REQUIREMENTS} are used. Shorten it to the essentials to control which are scored.")
        criteria = criteria[:MAX_REQUIREMENTS]

    resumes, files = {}, {}
    progress = st.progress(0)

    for idx, resume_file in enumerate(resume_files):
        try:
            text = anonymize(extract_text(resume_file.name, resume_file.getvalue()))
        except Exception as exc:
            st.warning(f"Skipped {resume_file.name}: {exc}")
        else:
            if text.strip():
                cid = f"C{len(resumes) + 1:03d}"
                resumes[cid] = text
                files[cid] = resume_file.name
            else:
                st.warning(f"Skipped {resume_file.name}: no text left after removing personal details.")
        progress.progress((idx + 1) / len(resume_files))

    progress.empty()

    if not resumes:
        st.error("No valid resumes found.")
        st.stop()

    note = ""
    with st.spinner("Loading Sentence-BERT model and screening applicants..."):
        try:
            results = {"keyword": match(criteria, resumes, "keyword")}
        except ValueError as exc:
            st.error(f"Screening could not run: {exc}")
            st.stop()
        try:
            results["semantic"] = match(criteria, resumes, "semantic", load_model())
        except Exception as exc:
            note = f"Sentence-BERT could not run ({exc}). Ranking uses TF-IDF only."

    st.session_state["run_seq"] = st.session_state.get("run_seq", 0) + 1
    st.session_state["run"] = dict(run_id=st.session_state["run_seq"], results=results, ranking=tabulate(results), criteria=criteria,
                                   resumes=resumes, files=files, note=note)
    st.success("Screening Completed")

# ===========================================
# Results
# ===========================================
run = st.session_state.get("run")

if run:
    results, ranking, files = run["results"], run["ranking"], run["files"]
    has_sem = "semantic" in results

    if run["note"]:
        st.warning(run["note"] + " Install `sentence-transformers` and allow the model download to use Sentence-BERT.")

    with st.expander(f"Job requirements used ({len(run['criteria'])}, equal weight)"):
        for i, c in enumerate(run["criteria"], 1):
            st.write(f"{i}. {c}")

    show_names = st.checkbox("Show candidate file names", value=False)

    by_id = {r["cid"]: r for r in ranking}

    def label(cid):
        return f"{cid} ({files[cid]})" if show_names else cid

    tab_rank, tab_detail, tab_eval = st.tabs(["Applicant Ranking", "Candidate Evidence", "Evaluate"])

    # ---------------------------------------
    # Ranking tab
    # ---------------------------------------
    with tab_rank:
        kw_top = min(ranking, key=lambda r: r["kw_rank"])

        st.subheader("🏆 Best Match")
        if has_sem:
            sem_top = min(ranking, key=lambda r: r["sem_rank"])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Top by BERT", label(sem_top["cid"]))
            c2.metric("BERT Score", f"{sem_top['sem_score']:.3f}")
            c3.metric("Top by TF-IDF", label(kw_top["cid"]))
            c4.metric("TF-IDF Score", f"{kw_top['kw_score']:.3f}")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Top by TF-IDF", label(kw_top["cid"]))
            c2.metric("TF-IDF Score", f"{kw_top['kw_score']:.3f}")
        if show_names:
            tops = [f"{kw_top['cid']} = {files[kw_top['cid']]}"]
            if has_sem:
                tops.insert(0, f"{sem_top['cid']} = {files[sem_top['cid']]}")
            st.caption("Best match file: " + " · ".join(dict.fromkeys(tops)))

        st.subheader("Applicant Ranking")
        rows = []
        for r in ranking:
            row = {"Candidate": r["cid"]}
            if show_names:
                row["Resume"] = files[r["cid"]]
            if has_sem:
                row["BERT Score"] = round(r["sem_score"], 3)
                row["BERT Rank"] = r["sem_rank"]
            row["TF-IDF Score"] = round(r["kw_score"], 3)
            row["TF-IDF Rank"] = r["kw_rank"]
            rows.append(row)
        results_df = pd.DataFrame(rows)
        
        st.dataframe(
            results_df,
            hide_index=True,
            column_config={
                "BERT Score": st.column_config.ProgressColumn("BERT Score", min_value=0, max_value=1, format="%.3f"),
                "TF-IDF Score": st.column_config.ProgressColumn("TF-IDF Score", min_value=0, max_value=1, format="%.3f"),
                "BERT Rank": st.column_config.NumberColumn("BERT Rank", format="%d"),
                "TF-IDF Rank": st.column_config.NumberColumn("TF-IDF Rank", format="%d"),
            },
            **FULL
        )

        if has_sem:
            if all(r["kw_rank"] == r["sem_rank"] for r in ranking):
                st.write("Both methods produced the same ranking.")
            else:
                st.write("The methods produced different rankings. Compare the rank columns; rank 1 is highest.")
        st.caption("Decision and Notes are editable: the recruiter has the final say. "
                   "Rows are ordered by BERT rank when available. Similarity is not a percentage of qualifications; "
                   "the two score scales differ, so compare ranks and evidence, and only compare candidates "
                   "screened against the same job description.")
        times = f"TF-IDF {results['keyword']['seconds']:.3f} s"
        if has_sem:
            times += f" · BERT {results['semantic']['seconds']:.3f} s"
        st.caption(f"Matching time for this run: {times} (excludes file reading and model loading).")

        with st.expander("How the ranking is computed"):
            st.markdown(
                "1. Each resume is split into passages of up to 80 words (15-word overlap on long lines).\n"
                "2. Every job requirement is compared with every passage of every resume. "
                "**TF-IDF** uses word and two-word-phrase overlap (cosine similarity); "
                "**BERT** uses Sentence-BERT (all-MiniLM-L6-v2) embeddings, which measure similarity in meaning.\n"
                "3. For each requirement, the **best-matching passage** in that resume sets the score.\n"
                "4. A candidate's score is the **average** of those best-passage scores across all requirements "
                "(each requirement counts equally).\n"
                "5. Candidates are ranked from highest to lowest score, separately for each method. "
                "Ties are broken by candidate ID.\n\n"
                "TF-IDF term weights are fitted on this batch of requirements and resumes, "
                "so scores depend on which resumes were uploaded together."
            )

        st.download_button(
            "Download Results",
            safe_csv(results_df),
            file_name="applicant_ranking.csv",
            mime="text/csv"
        )

    # ---------------------------------------
    # Evidence tab
    # ---------------------------------------
    with tab_detail:
        cid = st.selectbox("Candidate", list(by_id),
                           format_func=lambda c: f"{by_id[c].get('sem_rank', by_id[c]['kw_rank'])}. {label(c)}")
        r = by_id[cid]
        if show_names:
            st.markdown(f"**Resume file:** {files[cid]}")

        if has_sem:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("BERT Score", f"{r['sem_score']:.3f}")
            m2.metric("BERT Rank", r["sem_rank"])
            m3.metric("TF-IDF Score", f"{r['kw_score']:.3f}")
            m4.metric("TF-IDF Rank", r["kw_rank"])
        else:
            m1, m2 = st.columns(2)
            m1.metric("TF-IDF Score", f"{r['kw_score']:.3f}")
            m2.metric("TF-IDF Rank", r["kw_rank"])

        st.markdown("**Evidence for each job requirement**")
        kw_ev = [e for e in results["keyword"]["evidence"] if e["candidate_id"] == cid]
        sem_ev = ({e["criterion"]: e for e in results["semantic"]["evidence"] if e["candidate_id"] == cid}
                  if has_sem else {})
        table = []
        for e in kw_ev:
            row = {
                "Job requirement": e["criterion"],
                "TF-IDF Score": round(e["similarity"], 3),
                "TF-IDF best match": e["supporting_text"] if e["similarity"] > 0 else "(no shared words)",
            }
            if has_sem:
                s = sem_ev[e["criterion"]]
                row["BERT Score"] = round(s["similarity"], 3)
                row["BERT best match"] = s["supporting_text"]
            table.append(row)
        st.dataframe(
            pd.DataFrame(table),
            hide_index=True,
            column_config={
                "Job requirement": st.column_config.TextColumn(width="medium"),
                "TF-IDF best match": st.column_config.TextColumn(width="large"),
                "BERT best match": st.column_config.TextColumn(width="large"),
            },
            **FULL
        )
        st.caption("Each score compares one job requirement with the most similar passage (about 80 words) in the resume: "
                   "TF-IDF measures shared words, BERT measures similarity in meaning. "
                   "A candidate's score in the ranking is the average of that method's column. "
                   "The closest passage is not confirmed evidence of a qualification: check for negation and missing context.")

        with st.expander("Resume text that was actually scored (redacted)"):
            st.text(run["resumes"][cid])

    # ---------------------------------------
    # Evaluate tab
    # ---------------------------------------
    with tab_eval:
        st.subheader("Evaluate with reviewer labels")
        st.write("Label each resume against the same job requirements using your own judgment, "
                 "and record the reason. The labels are then used to check how well each method's ranking matches them.")
        st.markdown("**0:** inadequate evidence · **1:** partial or uncertain evidence · "
                    "**2:** explicit evidence for all essential requirements. Only **2** counts as relevant.")
        st.caption("Read each candidate's text in the Candidate Evidence tab (expander at the bottom). "
                   "Labelling before looking at the rankings reduces their influence on your judgment.")

        n = len(by_id)
        if show_names:
            st.dataframe(pd.DataFrame({"Candidate": list(by_id), "Resume": [files[c] for c in by_id]}),
                         hide_index=True, **FULL)
        editor = pd.DataFrame({
            "candidate_id": list(by_id),
            "label": pd.Series([None] * n, dtype="object"),
            "reason": [""] * n,
        })
        label_frame = st.data_editor(
            editor,
            hide_index=True,
            key=f"labels_{run['run_id']}",
            disabled=["candidate_id"],
            column_config={
                "candidate_id": st.column_config.TextColumn("Candidate"),
                "label": st.column_config.SelectboxColumn("Reviewer label", options=[0, 1, 2], required=True),
                "reason": st.column_config.TextColumn("Reason / evidence"),
            },
            **FULL
        )
        st.caption("Labels are judgments, not hiring truth.")

        k = st.number_input("Evaluate top k", min_value=1, max_value=n, value=min(5, n),
                            key=f"k_{run['run_id']}")
        if n < 5:
            st.caption("Fewer than five candidates: report the actual k shown, not Precision@5 or Recall@5.")

        try:
            labels = validate_labels(label_frame, list(by_id))
        except ValueError as exc:
            st.info(str(exc))
            labels = None

        if labels is not None:
            measures = evaluate_methods(results, labels, int(k))
            names = {"keyword": "TF-IDF", "semantic": "BERT"}
            for col, (_, m) in zip(st.columns(len(measures)), measures.iterrows()):
                with col:
                    st.markdown(f"**{names[m['method']]}**")
                    st.metric(f"Precision@{int(k)}", f"{m['precision']:.3f}")
                    st.metric(f"Recall@{int(k)}", "Undefined" if pd.isna(m["recall"]) else f"{m['recall']:.3f}")
                    st.caption(f"{int(m['hits'])} relevant in the top {int(k)}; "
                               f"{int(m['relevant_total'])} relevant in total.")
            st.caption("Precision = relevant among top k ÷ k. Recall = relevant among top k ÷ all relevant "
                       "(undefined if no candidate is labelled 2).")

            d1, d2 = st.columns([1, 1])
            d1.download_button("Download labels", safe_csv(label_frame), file_name="reviewer_labels.csv",
                               mime="text/csv", key=f"dl_labels_{run['run_id']}")
            d2.download_button("Download evaluation", safe_csv(measures), file_name="evaluation_metrics.csv",
                               mime="text/csv", key=f"dl_eval_{run['run_id']}")
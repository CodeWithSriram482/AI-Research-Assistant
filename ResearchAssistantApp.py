
# ResearchAssistantApp.py

# --- Imports ---
import re, time, requests

# Safe import for PyMuPDF
try:
    import fitz  # PyMuPDF
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

from urllib.parse import urlparse, urlunparse
from ddgs import DDGS
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st

# --- Constants ---
TOP_URLS = 6
PASSAGES_PER_PAGE = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARY_SENTENCES = 3
TIMEOUT = 8
MAX_WORDS_PER_CHUNK = 120
CROSSREF_API = "https://api.crossref.org/works"

# --- URL Cleaning ---
def unwrap_ddg(url):
    try:
        parsed = urlparse(url)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    except:
        return url

# --- Web Search ---
def search_web(query, max_results=TOP_URLS):
    urls, seen_urls = [], set()
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results*2):
            url = r.get("href") or r.get("uri")
            if not url: continue
            url = unwrap_ddg(url)
            if url not in seen_urls:
                urls.append(url)
                seen_urls.add(url)
            if len(urls) >= max_results:
                break
    return urls

# --- Fetch Text + Title from URL ---
def fetch_text(url, timeout=TIMEOUT):
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200 or "html" not in r.headers.get("content-type","").lower():
            return "", "No title", None
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title else "No title"
        meta_doi = soup.find("meta", attrs={"name":"citation_doi"}) or soup.find("meta", attrs={"name":"dc.identifier"})
        doi = meta_doi.get("content").strip() if meta_doi else None
        for tag in soup(["script","style","noscript","header","footer","svg","iframe","nav","aside"]):
            tag.extract()
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        text = re.sub(r"\s+", " ", text).strip()
        return text, title, doi
    except:
        return "", "No title", None

# --- PDF Parsing ---
def parse_pdf(file):
    try:
        doc = fitz.open(file)
        text = " ".join([page.get_text("text") for page in doc])
        return text
    except:
        return ""

# --- DOI Resolution ---
def resolve_doi(doi):
    try:
        r = requests.get(f"{CROSSREF_API}/{doi}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data["message"].get("title",[doi])[0], data["message"].get("URL", doi)
        return doi, doi
    except:
        return doi, doi

# --- Text Processing ---
def chunk_passages(text, max_words=MAX_WORDS_PER_CHUNK):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)] if words else []

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

# --- Research Agent ---
class ResearchAgent:
    def __init__(self, embed_model=EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embed_model)

    def run(self, query):
        start = time.time()
        urls = search_web(query, max_results=TOP_URLS)
        docs, titles, dois = [], {}, {}
        for u in urls:
            txt, title, doi = fetch_text(u)
            if not txt: txt = "No meaningful passage available."
            titles[u] = title
            dois[u] = doi
            chunks = chunk_passages(txt)
            if not chunks: chunks = [txt]
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": u, "passage": c})

        # --- Embeddings ---
        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]
        cosine = lambda a,b: np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10)
        sims = [cosine(e,q_emb) for e in emb_texts]

        # --- Metrics ---
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))

        # --- Top passages per URL ---
        top_passages, urls_added = [], set()
        for idx in np.argsort(sims)[::-1]:
            url = docs[idx]["url"]
            if url not in urls_added:
                top_passages.append({
                    "url": url,
                    "passage": docs[idx]["passage"],
                    "score": float(sims[idx]),
                    "title": titles.get(url,"No title"),
                    "doi": dois.get(url)
                })
                urls_added.add(url)
            if len(urls_added) >= TOP_URLS: break

        # --- Extractive summary ---
        sum_sentences, seen_sentences = [], set()
        for p in top_passages:
            sentences = split_sentences(p["passage"])
            emb_sentences = self.embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
            sent_sims = [cosine(e,q_emb) for e in emb_sentences]
            top_idx = np.argmax(sent_sims)
            p["top_sentence"] = sentences[top_idx] if sentences else ""
            for idx in np.argsort(sent_sims)[::-1]:
                s = sentences[idx]
                if s not in seen_sentences:
                    sum_sentences.append(s)
                    seen_sentences.add(s)
                    if len(sum_sentences) >= SUMMARY_SENTENCES: break
            if len(sum_sentences) >= SUMMARY_SENTENCES: break
        summary = " ".join(sum_sentences)
        end = time.time()

        # --- BibTeX & citations ---
        bib_entries, citation_urls = [], []
        for p in top_passages:
            if p['doi']:
                title, url_resolved = resolve_doi(p['doi'])
            else:
                title, url_resolved = p['title'], p['url']
            key = re.sub(r'[^a-zA-Z0-9]', '', title)[:40]
            bib_entries.append(rf"""@misc{{{key},
  title = {{{{ {title} }}}},
  howpublished = {{{{\url{{{url_resolved}}}}}}}, 
  year = {{2026}}
}}""")
            citation_urls.append(url_resolved)

        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": end-start,
            "num_passages": len(docs),
            "mean_similarity": mean_sim,
            "max_similarity": max_sim,
            "sources_used": len(top_passages),
            "bibtex": "\n\n".join(bib_entries),
            "citations": citation_urls
        }

# --- Streamlit Dashboard ---
def streamlit_dashboard():
    st.title("AI Research Assistant")
    query = st.text_input("Enter your research query:", "How does soil liquefaction affect building foundations during earthquakes?")
    uploaded_pdf = st.file_uploader("Upload PDF (optional):", type=["pdf"])
    if st.button("Run Research"):
        agent = ResearchAgent()
        if uploaded_pdf:
            pdf_text = parse_pdf(uploaded_pdf)
            st.write("PDF parsed, first 500 chars:")
            st.write(pdf_text[:500])
        out = agent.run(query)

        st.subheader("Top Sources")
        for i, p in enumerate(out["passages"],1):
            st.markdown(f"[{i}] {p['title']} ({p['score']:.3f}) - {p['url']}")
            highlighted = p["passage"].replace(p["top_sentence"], f"**{p['top_sentence']}**") if p["top_sentence"] else p["passage"]
            st.write(highlighted[:800]+"...")

        st.subheader("Extractive Summary")
        st.write(out["summary"])

        st.subheader("Metrics Evaluation")
        st.write({
            "Mean Similarity": out["mean_similarity"],
            "Max Similarity": out["max_similarity"],
            "Number of passages": out["num_passages"],
            "Sources used": out["sources_used"]
        })

        st.subheader("BibTeX Entries")
        st.code(out["bibtex"])

        st.subheader("Citation URLs")
        for c in out["citations"]:
            st.markdown(f"- {c}")

        st.write(f"Processing time: {out['time']:.2f}s")

# --- Run Streamlit ---
if __name__=="__main__":
    streamlit_dashboard()

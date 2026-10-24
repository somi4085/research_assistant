import json
import re
from pathlib import Path

import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

from pdf_ingestion import pdf_ingestion
from text_chunking import text_chunking
from main import (
    PDF_PATH,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    extract_claims_from_answer,
    verify_claim,
)

st.set_page_config(
    page_title="Research Assistant RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Research Assistant with Verifiable Answers")
st.caption("Ask questions about the research PDF and inspect the supporting pages.")


@st.cache_resource(show_spinner="Loading PDF, chunks, and embedding model...")
def load_rag_resources():
    """Load the PDF and embedding model once per Streamlit session cache."""
    total_pages, pages_data = pdf_ingestion(PDF_PATH)
    if not pages_data:
        raise ValueError("No readable text found in research.pdf.")

    chunks = []
    for page in pages_data:
        page_chunks = text_chunking(page["text"])
        for chunk in page_chunks:
            chunk_text = (
                " ".join(chunk).strip()
                if isinstance(chunk, list)
                else str(chunk).strip()
            )
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page_number": page["page_number"],
                })

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedding_model.encode(
        [chunk["text"] for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    page_lookup = {
        page["page_number"]: page["text"] for page in pages_data
    }
    return total_pages, chunks, embedding_model, page_lookup


def answer_question(question, chunks, embedding_model, page_lookup):
    query_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    ranked_chunks = sorted(
        chunks,
        key=lambda chunk: float(np.dot(query_embedding, chunk["embedding"])),
        reverse=True,
    )
    retrieved_chunks = ranked_chunks[:TOP_K]
    retrieved_pages = sorted({c["page_number"] for c in retrieved_chunks})
    context = "\n\n".join(
        f"[Page {c['page_number']}]\n{c['text']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a research assistant answering a question about a PDF.
Use only the supplied context. Never invent facts or evidence.
If the context does not contain enough information, say so clearly.

Available page numbers: {retrieved_pages}

Rules:
1. Every factual sentence must have exactly one page citation.
2. Use this exact format: (Page 4).
3. Cite only page numbers in the available page numbers list.
4. Put each independent factual claim in a separate sentence.
5. Do not combine claims supported by different pages in one sentence.
6. Do not guess page numbers.

CONTEXT:
{context}

QUESTION:
{question}

Give a concise, direct answer with page citations.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"].strip()

    cited_pages = [
        int(p) for p in re.findall(
            r"\(\s*Page\s+(\d+)\s*\)", answer, re.IGNORECASE
        )
    ]
    claims = extract_claims_from_answer(answer)

    results = []
    for item in claims:
        page = item["citation_page"]
        if page is None:
            result = {
                "classification": "INSUFFICIENT_EVIDENCE",
                "explanation": "The claim has no unambiguous page citation.",
            }
        elif page not in page_lookup:
            result = {
                "classification": "INVALID_CITATION",
                "explanation": "The cited page does not exist in the PDF.",
            }
        elif page not in retrieved_pages:
            result = {
                "classification": "INVALID_CITATION",
                "explanation": "The cited page was not among retrieved pages.",
            }
        else:
            try:
                result = verify_claim(item["claim"], page, page_lookup[page])
            except Exception as error:
                result = {
                    "classification": "VERIFICATION_ERROR",
                    "explanation": str(error),
                }
        results.append({**item, **result})

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "retrieved_pages": retrieved_pages,
        "cited_pages": cited_pages,
        "results": results,
    }


try:
    total_pages, chunks, embedding_model, page_lookup = load_rag_resources()
    st.success(f"PDF loaded successfully · {total_pages} pages · {len(chunks)} chunks")
except Exception as error:
    st.error(f"Could not load the PDF or embedding model: {error}")
    st.info("Check that research.pdf is in the project folder and your virtual environment is active.")
    st.stop()

question = st.text_input(
    "Your question",
    placeholder="e.g. What is the reason for using mobile devices?",
)

ask = st.button("Ask question", type="primary", disabled=not question.strip())

if ask:
    try:
        with st.spinner("Retrieving evidence, generating answer, and verifying claims..."):
            output = answer_question(question.strip(), chunks, embedding_model, page_lookup)

        st.subheader("Answer")
        st.write(output["answer"])

        st.subheader("Claim verification")
        if not output["results"]:
            st.warning("No claims were extracted from the answer.")
        else:
            for index, item in enumerate(output["results"], start=1):
                label = item["classification"]
                with st.expander(f"{index}. {label} — {item['claim']}", expanded=(label != "SUPPORTED")):
                    st.write("**Citation page:**", item["citation_page"] if item["citation_page"] is not None else "Missing")
                    st.write("**Explanation:**", item["explanation"])

        with st.expander("Retrieved evidence", expanded=False):
            st.write("**Retrieved pages:**", output["retrieved_pages"])
            for index, chunk in enumerate(output["retrieved_chunks"], start=1):
                st.markdown(f"**Chunk {index} · Page {chunk['page_number']} · Similarity {float(np.dot(embedding_model.encode(question.strip(), normalize_embeddings=True), chunk['embedding'])):.3f}**")
                st.write(chunk["text"])
                st.divider()

        evaluation_summary = {}
        for item in output["results"]:
            label = item["classification"]
            evaluation_summary[label] = evaluation_summary.get(label, 0) + 1

        result_json = {
            "question": question.strip(),
            "answer": output["answer"],
            "retrieved_pages": output["retrieved_pages"],
            "cited_pages": output["cited_pages"],
            "evaluation_summary": evaluation_summary,
            "results": output["results"],
        }
        st.download_button(
            "Download evaluation results (JSON)",
            data=json.dumps(result_json, indent=2, ensure_ascii=False),
            file_name="evaluation_results.json",
            mime="application/json",
        )

    except Exception as error:
        st.error(f"Question processing failed: {error}")
        st.caption("Check that Ollama is running and the configured model is available.")

st.divider()
st.caption("Note: automated claim labels are model-generated checks, not a guarantee of factual correctness.")

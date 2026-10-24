import json
import re
import numpy as np
import ollama

from pdf_ingestion import pdf_ingestion
from text_chunking import text_chunking
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

PDF_PATH = "research.pdf"
OLLAMA_MODEL = "llama3.2"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
OUTPUT_PATH = "evaluation_results.json"

ALLOWED_LABELS = {
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_SUPPORTED",
    "INSUFFICIENT_EVIDENCE",
}

VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": [
                "SUPPORTED",
                "CONTRADICTED",
                "NOT_SUPPORTED",
                "INSUFFICIENT_EVIDENCE",
            ],
        },
        "explanation": {"type": "string"},
    },
    "required": ["classification", "explanation"],
}


# ============================================================
# 2. JSON PARSING HELPER
# ============================================================

def parse_json_response(response_text):
    """Parse JSON returned by Ollama, including fenced JSON."""
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("Ollama returned an empty response.")

    cleaned_text = response_text.strip()
    cleaned_text = re.sub(
        r"\A```(?:json)?\s*", "", cleaned_text,
        count=1, flags=re.IGNORECASE
    )
    cleaned_text = re.sub(r"\s*```\s*\Z", "", cleaned_text, count=1)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "Ollama did not return valid JSON.\n"
            f"Response was:\n{cleaned_text}"
        ) from error


# ============================================================
# 3. CLAIM EXTRACTION
# ============================================================

def extract_claims_from_answer(answer):
    """
    Extract claims deterministically from explicit answer citations.

    Each factual statement should have one citation, e.g. (Page 4).
    This avoids asking an LLM to guess citation page numbers.
    """
    claims = []
    citation_pattern = re.compile(
        r"\(\s*Page\s+(\d+)\s*\)", re.IGNORECASE
    )
    matches = list(citation_pattern.finditer(answer))

    if not matches:
        # Preserve uncited answer text so it can be flagged as missing citation.
        text = re.sub(r"\s+", " ", answer).strip()
        if text:
            claims.append({"claim": text, "citation_page": None})
        return claims

    cursor = 0
    for match in matches:
        # Text between the previous citation and this citation is associated
        # with this citation. The answer prompt asks for one claim per sentence.
        segment = answer[cursor:match.start()]
        cursor = match.end()

        # Keep only the final sentence/statement in this segment, if there are
        # multiple sentences before the citation.
        sentence_parts = re.split(r"(?<=[.!?])\s+|\n+", segment.strip())
        claim_text = sentence_parts[-1].strip() if sentence_parts else ""
        claim_text = re.sub(r"^\s*[-*]\s+", "", claim_text)
        claim_text = claim_text.strip(" \t\r\n.;:")

        if claim_text:
            claims.append({
                "claim": claim_text,
                "citation_page": int(match.group(1)),
            })

    # Any factual text after the final citation is uncited and should not vanish.
    trailing_text = answer[cursor:].strip()
    if trailing_text:
        trailing_parts = re.split(
            r"(?<=[.!?])\s+|\n+", trailing_text
        )
        for part in trailing_parts:
            part = part.strip(" \t\r\n.;:")
            if part:
                claims.append({"claim": part, "citation_page": None})

    return claims


# ============================================================
# 4. CLAIM VERIFICATION
# ============================================================

def verify_claim(claim, citation_page, evidence_text):
    """Verify a claim against evidence from its cited PDF page."""
    prompt = f"""
You are a strict, evidence-grounded factual claim verifier.
Use ONLY the supplied evidence. Do not use outside knowledge.
Judge the meaning of the complete claim, not just keyword overlap.

CLAIM:
{claim}

CITATION PAGE:
{citation_page}

EVIDENCE:
{evidence_text}

Choose exactly ONE classification:
SUPPORTED, CONTRADICTED, NOT_SUPPORTED, INSUFFICIENT_EVIDENCE.

SUPPORTED:
The evidence explicitly states the claim or clearly expresses the same meaning.

CONTRADICTED:
The evidence explicitly states a fact that conflicts with the claim.

NOT_SUPPORTED:
The evidence is relevant, but does not establish the claim and does not directly contradict it.

INSUFFICIENT_EVIDENCE:
The evidence is missing, ambiguous, incomplete, or cannot reliably resolve the claim.
Use this for details such as an exact interview duration when no duration is reported.

Examples:
- Claim: interviews were conducted in English. Evidence: interviews were conducted in Polish.
  Label: CONTRADICTED.
- Claim: interviews were conducted in Polish. Evidence says they were conducted in Polish.
  Label: SUPPORTED.
- Claim: regression analysis was used. Evidence only describes counting responses and percentages.
  Label: NOT_SUPPORTED, unless evidence explicitly rules regression out.
- Claim: every interview lasted exactly 30 minutes. Evidence gives no duration.
  Label: INSUFFICIENT_EVIDENCE.

Do not treat an unmentioned detail as automatically false.
Return a JSON object with exactly these fields:
- classification: one allowed label
- explanation: a concise explanation grounded only in the evidence
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        format=VERIFICATION_SCHEMA,
        messages=[{"role": "user", "content": prompt}],
    )

    result = parse_json_response(response["message"]["content"])
    if not isinstance(result, dict):
        raise ValueError("Verification response must be a JSON object.")

    classification = result.get("classification")
    explanation = result.get("explanation")

    if not isinstance(classification, str):
        raise ValueError("Verification response has no valid classification.")

    classification = classification.strip().upper()
    if classification not in ALLOWED_LABELS:
        raise ValueError(f"Unexpected verification label: {classification}")
    if not isinstance(explanation, str):
        raise ValueError("Verification explanation must be a string.")

    return {
        "classification": classification,
        "explanation": explanation.strip(),
    }


# ============================================================
# 5. MAIN RAG PIPELINE
# ============================================================

def run_rag_pipeline():
    print("=" * 60)
    print("RESEARCH ASSISTANT WITH VERIFIABLE ANSWERS")
    print("=" * 60)

    total_pages, pages_data = pdf_ingestion(PDF_PATH)
    print("\nTotal pages:", total_pages)

    if not pages_data:
        raise ValueError("No readable text found in the PDF.")

    # --------------------------------------------------------
    # 6. TEXT CHUNKING
    # --------------------------------------------------------
    all_chunks = []
    for page in pages_data:
        page_chunks = text_chunking(page["text"])
        current_page = page["page_number"]

        for chunk in page_chunks:
            # text_chunking may return a list of strings or a string.
            chunk_text = (
                " ".join(chunk).strip()
                if isinstance(chunk, list)
                else str(chunk).strip()
            )
            if chunk_text:
                all_chunks.append({
                    "text": chunk_text,
                    "page_number": current_page,
                })

    if not all_chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print("\nTotal chunks:", len(all_chunks))

    # --------------------------------------------------------
    # 7. CREATE EMBEDDINGS
    # --------------------------------------------------------
    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    chunk_texts = [chunk["text"] for chunk in all_chunks]
    embeddings = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding

    # --------------------------------------------------------
    # 8. GET QUESTION
    # --------------------------------------------------------
    query = input(
        "\nAsk a question about the PDF "
        "(press Enter for the default question): "
    ).strip()
    if not query:
        query = "How were data collected and analyzed in this study?"

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # --------------------------------------------------------
    # 9. RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------------
    for chunk in all_chunks:
        chunk["similarity"] = float(
            np.dot(query_embedding, chunk["embedding"])
        )

    ranked_chunks = sorted(
        all_chunks,
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )
    retrieved_chunks = ranked_chunks[:TOP_K]
    retrieved_pages = sorted({
        chunk["page_number"] for chunk in retrieved_chunks
    })

    print("\nRETRIEVED CHUNKS")
    for chunk in retrieved_chunks:
        print("Page:", chunk["page_number"])
        print("Similarity:", round(chunk["similarity"], 4))
        print("Text:", chunk["text"][:200])
        print("-" * 50)
    print("\nRetrieved pages:", retrieved_pages)

    # --------------------------------------------------------
    # 10. BUILD CONTEXT
    # --------------------------------------------------------
    context_parts = [
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ]
    context = "\n\n".join(context_parts)

    # --------------------------------------------------------
    # 11. GENERATE ANSWER WITH CITATIONS
    # --------------------------------------------------------
    answer_prompt = f"""
You are a research assistant answering a question about a PDF.
Use only the provided context. Never invent facts or evidence.
If the context does not contain enough information, say so clearly.

Available page numbers: {retrieved_pages}

Rules:
1. Every factual sentence must have exactly one page citation.
2. Use this exact format: (Page 4).
3. Cite only page numbers in the available page numbers list above.
4. Put each independent factual claim in a separate sentence.
5. Do not combine claims supported by different pages in one sentence.
6. Do not guess page numbers.

CONTEXT:
{context}

QUESTION:
{query}

Give a concise, factual answer with page citations.
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
    )
    answer = response["message"]["content"].strip()

    print("\n" + "=" * 60)
    print("RAG ANSWER")
    print(answer)
    print("=" * 60)

    # --------------------------------------------------------
    # 12. CHECK CITATIONS
    # --------------------------------------------------------
    cited_pages = [
        int(page) for page in re.findall(
            r"\(\s*Page\s+(\d+)\s*\)", answer, re.IGNORECASE
        )
    ]

    if not cited_pages:
        print("\nWARNING: No page citations found in the answer.")

    for page in sorted(set(cited_pages)):
        if page in retrieved_pages:
            print(f"Page {page}: Retrieved-page reference")
        else:
            print(f"Page {page}: INVALID_CITATION (not among retrieved pages)")
    print("Cited pages:", cited_pages)

    # --------------------------------------------------------
    # 13. EXTRACT CLAIMS
    # --------------------------------------------------------
    claims = extract_claims_from_answer(answer)
    print("\nEXTRACTED CLAIMS")
    for index, item in enumerate(claims, start=1):
        print(f"{index}. {item}")

    # --------------------------------------------------------
    # 14. VERIFY CLAIMS
    # --------------------------------------------------------
    evaluation_summary = {
        "SUPPORTED": 0,
        "CONTRADICTED": 0,
        "NOT_SUPPORTED": 0,
        "INSUFFICIENT_EVIDENCE": 0,
        "INVALID_CITATION": 0,
        "VERIFICATION_ERROR": 0,
    }
    evaluation_results = []
    page_lookup = {
        page["page_number"]: page["text"] for page in pages_data
    }

    for item in claims:
        claim = item["claim"]
        citation_page = item["citation_page"]
        print("\n" + "=" * 60)
        print("CLAIM:", claim)
        print("CITATION PAGE:", citation_page)

        if citation_page is None:
            result = {
                "classification": "INSUFFICIENT_EVIDENCE",
                "explanation": "The claim has no unambiguous page citation.",
            }
        elif citation_page not in page_lookup:
            result = {
                "classification": "INVALID_CITATION",
                "explanation": "The cited page does not exist in the PDF.",
            }
        elif citation_page not in retrieved_pages:
            result = {
                "classification": "INVALID_CITATION",
                "explanation": "The cited page was not among the retrieved pages.",
            }
        else:
            try:
                result = verify_claim(
                    claim,
                    citation_page,
                    page_lookup[citation_page],
                )
            except (ValueError, KeyError, TypeError) as error:
                print("VERIFICATION ERROR:", error)
                result = {
                    "classification": "VERIFICATION_ERROR",
                    "explanation": str(error),
                }
            except Exception as error:
                print("UNEXPECTED VERIFICATION ERROR:", error)
                result = {
                    "classification": "VERIFICATION_ERROR",
                    "explanation": str(error),
                }

        evaluation_summary[result["classification"]] += 1
        print("CLASSIFICATION:", result["classification"])
        print("EXPLANATION:", result["explanation"])
        evaluation_results.append({
            "claim": claim,
            "citation_page": citation_page,
            "classification": result["classification"],
            "explanation": result["explanation"],
        })

    # --------------------------------------------------------
    # 15. SUMMARY AND SAVE RESULTS
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    for label, count in evaluation_summary.items():
        print(f"{label}: {count}")
    print("Total extracted claims:", len(claims))
    print("Total classified claims:", sum(evaluation_summary.values()))
    print("=" * 60)

    output = {
        "question": query,
        "answer": answer,
        "retrieved_pages": retrieved_pages,
        "cited_pages": cited_pages,
        "evaluation_summary": evaluation_summary,
        "results": evaluation_results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print(f"\nEvaluation results saved to {OUTPUT_PATH}")


# ============================================================
# 16. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        run_rag_pipeline()
    except FileNotFoundError as error:
        print("File not found:", error)
        print("Check that research.pdf and required project files exist.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as error:
        print("\nPipeline failed:", error)
        raise


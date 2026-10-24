# Research Assistant with Verifiable Answers

A local Retrieval-Augmented Generation (RAG) application that answers questions from a PDF document and provides supporting evidence, page citations, and claim verification.

## Features

- PDF ingestion and text extraction
- Text chunking for efficient retrieval
- Semantic search using sentence embeddings
- Local LLM-powered answer generation using Ollama
- Source page citations
- Claim verification against retrieved evidence
- Evidence display for answer validation
- Evaluation scripts for testing answer quality
- Interactive web interface built with Streamlit

## Tech Stack

- Python
- Streamlit
- Ollama
- Llama 3.2
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Retrieval-Augmented Generation (RAG)

## Project Structure

```text
research_assistant/
├── app.py
├── main.py
├── pdf_ingestion.py
├── text_chunking.py
├── test_ollama.py
├── research.pdf
├── evaluation_cases.json
├── evaluate.py
├── evaluation_results.json
├── answer_quality_cases.json
├── answer_quality_evaluation.py
└── answer_quality_results.json
```

## Prerequisites

- Python 3.10
- Ollama installed
- Llama 3.2 model downloaded
- Required Python packages installed

## Setup

1. Clone or download this repository.
2. Create and activate a Python virtual environment.
3. Install the Python packages required by the project.
4. Install Ollama and download the Llama 3.2 model:

   `ollama pull llama3.2`

5. Start Ollama.
6. Place the PDF document in the project directory.

## Run the Application

Start the Streamlit interface:

`streamlit run app.py --server.fileWatcherType none`

## Run Evaluation

Run the evaluation scripts using the project's configured Python environment:

- `evaluate.py` — retrieval and claim verification evaluation.
- `answer_quality_evaluation.py` — answer quality evaluation.

Check the generated JSON results for evaluation outcomes.

## How It Works

1. **Ingestion:** Extract text from the source PDF.
2. **Chunking:** Split extracted text into smaller passages.
3. **Embedding:** Convert passages into semantic vectors.
4. **Retrieval:** Find relevant passages for the user's question.
5. **Generation:** Generate an answer using the local LLM and retrieved context.
6. **Verification:** Compare claims against retrieved source evidence.
7. **Citation:** Display supporting evidence and source page numbers.
8. **Evaluation:** Run test cases to inspect system quality.

## Limitations

- Answer quality depends on PDF extraction, retrieval quality, and the language model.
- Automated claim verification can produce incorrect labels.
- Citations and supporting evidence should be checked against the original document.
- Response latency depends on local hardware and model configuration.

## Future Improvements

- Improve retrieval quality and evidence selection.
- Add automated regression tests.
- Measure retrieval accuracy, citation accuracy, and answer faithfulness.
- Add structured logging and performance monitoring.
- Support multiple PDF documents.

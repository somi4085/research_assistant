import pymupdf
import os

def pdf_ingestion(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("File not found")
    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
        pages_data = []
        for page_index, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                continue
            pages_data.append({
                "page_number" : page_index + 1,
                "text" : text
            })

    return total_pages, pages_data

def text_chunking(text):
    words = text.split()
    chunk_size = 100
    overlap = 20
    chunks = []
    for i in range(0, len(words), chunk_size-overlap):
        chunk = words[i:i+chunk_size]
        chunks.append(chunk) 

    return chunks

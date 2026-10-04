from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    return pages


def create_chunks(
    pages,
    chunk_size=1000,
    overlap=150
):

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

            start += chunk_size - overlap

    return chunks


def get_pdf_info(uploaded_file):

    reader = PdfReader(uploaded_file)

    return {
        "total_pages": len(reader.pages)
    }
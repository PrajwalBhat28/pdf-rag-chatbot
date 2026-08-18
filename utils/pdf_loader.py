import re
import pymupdf
from langchain_core.documents import Document


def extract_title_from_first_page(page) -> str:
    """
    Extract the likely paper title from the first page by selecting the
    largest text near the top of the page.
    """
    data = page.get_text("dict")
    candidates = []

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            line_text_parts = []
            line_sizes = []
            line_x = []
            line_y = []

            for span in line.get("spans", []):
                text = " ".join(span.get("text", "").split())
                if not text:
                    continue

                line_text_parts.append(text)
                line_sizes.append(float(span.get("size", 0)))
                x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                line_x.append(x0)
                line_y.append(y0)

            if not line_text_parts:
                continue

            text = " ".join(line_text_parts).strip()
            if len(text) < 8:
                continue

            y_pos = min(line_y) if line_y else 0
            if y_pos > page.rect.height * 0.45:
                continue

            candidates.append(
                {
                    "text": text,
                    "size": max(line_sizes) if line_sizes else 0,
                    "y": y_pos,
                    "x": min(line_x) if line_x else 0,
                }
            )

    if not candidates:
        return "Unknown Title"

    max_size = max(item["size"] for item in candidates)
    title_lines = [
        item for item in candidates
        if item["size"] >= max_size - 1.0
    ]

    title_lines = sorted(title_lines, key=lambda item: (item["y"], item["x"]))

    title = " ".join(item["text"] for item in title_lines)
    title = re.sub(r"\s+", " ", title).strip(" -–—|")

    return title[:250] if title else "Unknown Title"


def extract_text_from_pdf(uploaded_file):
    """
    Extract full text from the PDF and create one LangChain Document per page
    with page metadata. Also extract the paper title from the first page.
    """
    pdf = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")

    full_text = []
    documents = []
    paper_title = "Unknown Title"

    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text()
        full_text.append(page_text)

        if page_number == 1:
            paper_title = extract_title_from_first_page(page)

        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "page": page_number,
                    "source": uploaded_file.name,
                },
            )
        )

    pdf.close()

    return "\n".join(full_text), documents, paper_title
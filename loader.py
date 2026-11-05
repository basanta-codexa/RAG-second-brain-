import re
from pathlib import Path
from pypdf import PdfReader


def load_pdf(path: Path) -> list[dict]:
    """Read a PDF and return one record per page."""
    reader = PdfReader(path)
    records = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip empty pages
            records.append(
                {"text": text, "source": path.name, "page": page_number})
    return records


def load_text(path: Path) -> list[dict]:
    """Read a .txt or .md file. If it has <<<PAGE n>>> markers, split into pages."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"<<<PAGE (\d+)>>>", text)
    if len(parts) == 1:  # no markers: whole file is one record
        return [{"text": text, "source": path.name, "page": 1}]
    records = []
    # parts looks like [before, "1", text1, "2", text2, ...]
    for num, body in zip(parts[1::2], parts[2::2]):
        if body.strip():
            records.append(
                {"text": body, "source": path.name, "page": int(num)})
    return records


def load_folder(folder: str) -> list[dict]:
    """Load every supported file in a folder."""
    records = []
    for path in sorted(Path(folder).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            records.extend(load_pdf(path))
        elif suffix in (".txt", ".md"):
            records.extend(load_text(path))
        else:
            print(f"Skipping unsupported file: {path.name}")
    return records


if __name__ == "__main__":
    docs = load_folder("data")
    print(f"Loaded {len(docs)} pieces\n")
    for d in docs[:3]:
        print(f"[{d['source']} | page {d['page']}]")
        print(d["text"][:200].replace("\n", " "))
        print("-" * 40)

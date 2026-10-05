"""Audit published PDFs and embedded listings; optionally render every page.

Requires PyMuPDF. Run after both document builders, from any directory.
Visual inspection is a separate step; rendering alone does not certify layout.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    "main": "Compiladores_ML_Desde_Cero_2026",
    "cuda": "Tutorial_CUDA_Desde_Cero_2026",
    "tinygrad": "Tutorial_Tinygrad_Desde_Cero_2026",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def embedded_files(name):
    if name == "main":
        return sorted(p for folder, pattern in [
            ("code/lumbre", "*.py"), ("code/examples", "*.py"),
            ("code/kernels", "*"), ("code/tests", "*.py")
        ] for p in (ROOT / folder).glob(pattern) if p.is_file())
    if name == "cuda":
        return [ROOT / "tutorials/cuda/common.cuh", *sorted(
            (ROOT / "tutorials/cuda").glob("*.cu")), ROOT / "tutorials/validate_cuda.py"]
    return [ROOT / "tutorials/tinygrad/labs.py"]


def audit(render):
    reports = ROOT / "tutorials/reports"
    source_audit = {}
    for name, stem in DOCUMENTS.items():
        source, pdf = ROOT / (stem + ".tex"), ROOT / (stem + ".pdf")
        tex = source.read_text(encoding="utf-8")
        build = json.loads((reports / f"{name}_build.json").read_text(encoding="utf-8"))
        assert build["source_sha256"] == digest(source), f"Stale source: {stem}"
        assert build["pdf_sha256"] == digest(pdf), f"Stale PDF: {stem}"
        listings = re.findall(r"\\begin\{lstlisting\}(?:\[[^\n]*\])?\n(.*?)\n\\end\{lstlisting\}", tex, re.S)
        files = embedded_files(name)
        for path in files:
            content = path.read_text(encoding="utf-8")
            assert content.rstrip("\n") in {s.rstrip("\n") for s in listings}, path
            if path.suffix == ".py":
                compile(content, str(path), "exec")
        cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", tex)
                 for key in group.split(",")}
        bibliography = re.findall(r"\\bibitem\{([^}]+)\}", tex)
        assert cited <= set(bibliography), cited - set(bibliography)
        assert len(bibliography) == len(set(bibliography)), "Duplicate bibliography keys"
        labels = re.findall(r"\\label\{([^}]+)\}", tex)
        assert len(labels) == len(set(labels)), "Duplicate labels"
        source_audit[name] = {
            "embedded_files": [p.relative_to(ROOT).as_posix() for p in files],
            "exact_match": True, "python_sources_compile": True,
            "citation_keys": len(cited), "bibliography_entries": len(bibliography),
            "listing_blocks": len(listings), "source_sha256": digest(source),
            "pdf_sha256": digest(pdf),
        }
        with fitz.open(pdf) as doc:
            font_refs = {font[0] for page in doc for font in page.get_fonts()}
            missing_fonts = [doc.extract_font(ref)[0] for ref in font_refs
                             if not doc.extract_font(ref)[3]]
            assert not missing_fonts, f"Fonts not embedded: {missing_fonts}"
            blank, outside, replacement = [], [], []
            words_total = 0
            if render:
                directory = ROOT / "tmp/pdfs" / name
                directory.mkdir(parents=True, exist_ok=True)
            for index, page in enumerate(doc):
                words = page.get_text("words")
                words_total += len(words)
                if not words:
                    blank.append(index + 1)
                for word in words:
                    if not (page.rect + (-1, -1, 1, 1)).contains(fitz.Rect(word[:4])):
                        outside.append({"page": index + 1, "word": word[4]})
                if "\ufffd" in page.get_text():
                    replacement.append(index + 1)
                if render:
                    page.get_pixmap(dpi=60).save(directory / f"page-{index + 1:03}.png")
            result = {
                "file": pdf.name, "pages": len(doc), "bookmarks": len(doc.get_toc()),
                "text_words": words_total, "blank_pages": blank,
                "outside_page_words": outside, "replacement_character_pages": replacement,
                "rendered_pages": len(doc) if render else 0,
                "embedded_fonts": len(font_refs), "fonts_not_embedded": missing_fonts,
                "visual_sample_pages": [], "visual_review": "Pending separate visual inspection",
                "pdf_sha256": digest(pdf),
            }
            write_json(reports / f"{name}_pdf_qa.json", result)
            assert not (blank or outside or replacement), result
            if name == "main":
                write_json(ROOT / "pdf_qa.json", {
                    **result, "toc_items": result["bookmarks"],
                    "references_converged": build["references_converged"],
                    "embedded_source_files": len(files),
                    "validation_report": "tutorials/reports/main_pdf_qa.json",
                    "runtime_validation_report": "code/reports/validation_20261001/validation.json",
                })
            print(f"{name}: {len(doc)} pages, {len(files)} exact embedded files", flush=True)
    write_json(reports / "source_audit.json", source_audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render every page under tmp/pdfs/")
    audit(parser.parse_args().render)

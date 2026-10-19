"""Registry of raw source documents.

The raw files keep the names WHO and NCBI distribute them with, which are not
self-explanatory. Everything else in the code base refers to sources through the
logical IDs defined here, so renaming a file only requires a change in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
CHUNKS_DIR: Path = PROJECT_ROOT / "data" / "chunks"

LICENCE: str = "CC BY-NC-SA 3.0 IGO"


@dataclass(frozen=True)
class Source:
    """Metadata for one raw source file."""

    source_id: str
    filename: str
    title: str
    year: int
    isbn: str

    @property
    def path(self) -> Path:
        """Absolute path to the raw file in ``data/raw/``."""
        return RAW_DIR / self.filename


SOURCES: dict[str, Source] = {
    "aware_epub": Source(
        source_id="aware_epub",
        filename="9789240069374-eng.epub",
        title="The WHO AWaRe (Access, Watch, Reserve) antibiotic book",
        year=2022,
        isbn="978-92-4-006937-4",
    ),
    "aware_pdf": Source(
        source_id="aware_pdf",
        filename="9789240062382-eng.pdf",
        title="The WHO AWaRe (Access, Watch, Reserve) antibiotic book",
        year=2022,
        isbn="978-92-4-006238-2",
    ),
    "hypertension_pdf": Source(
        source_id="hypertension_pdf",
        filename="Bookshelf_NBK573631.pdf",
        title="Guideline for the pharmacological treatment of hypertension in adults",
        year=2021,
        isbn="978-92-4-003398-6",
    ),
}


def get_source(source_id: str) -> Source:
    """Return the registered source, raising a clear error if its file is missing."""
    source = SOURCES[source_id]
    if not source.path.exists():
        raise FileNotFoundError(
            f"Raw file for {source_id!r} not found at {source.path}. "
            "Place the WHO source files in data/raw/ (see README)."
        )
    return source

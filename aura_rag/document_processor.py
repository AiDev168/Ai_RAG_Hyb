"""Document loading (incl. OCR) and text splitting."""

import logging
from pathlib import Path
from typing import List

from .config import Config
from .dependencies import (
    ALL_KNOWN_EXTENSIONS,
    Document,
    OCR_AVAILABLE,
    OCR_IMAGE_EXTENSIONS,
    PyMuPDFModule,
    Pytesseract,
    RecursiveCharacterTextSplitter,
    SUPPORTED_EXTENSIONS,
    _OCRImageLoader,
)
from .ocr import (
    OCR_LANG_INSTALL_HINT,
    _clean_ocr_text,
    _prepare_ocr_image,
    _resolve_ocr_languages,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles document loading and processing"""

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, config: Config):
        self.config = config
        self._build_splitter()

    def _build_splitter(self):
        if RecursiveCharacterTextSplitter is None:
            raise RuntimeError(
                "langchain-text-splitters is not installed. "
                "Run: pip install langchain-text-splitters --break-system-packages"
            )
        strategy = self.config.config.get("chunking_strategy", "fixed")
        if strategy == "structural":
            # Structure-aware: prefer splitting on headings and paragraph
            # boundaries before falling back to sentence/word splits, so a
            # chunk is more likely to be a coherent section rather than an
            # arbitrary character-count slice through the middle of a
            # thought. Best for well-formatted Markdown/docx/PDF text with
            # real headings; for unstructured plain text it behaves the
            # same as the fixed strategy since those separators just never
            # match.
            separators = [
                "\n# ", "\n## ", "\n### ", "\n#### ",
                "\n\n\n", "\n\n", "\n",
                "。", "؟", "!", "؛", ". ", " ", "",
            ]
        else:
            separators = ["\n\n", "\n", "。", "؟", "!", "؛", " ", ""]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.config["chunk_size"],
            chunk_overlap=self.config.config["chunk_overlap"],
            length_function=len,
            separators=separators,
        )

    def get_loader(self, file_path: Path):
        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            if ext in ALL_KNOWN_EXTENSIONS:
                return None, (
                    f"'{ext}' support isn't loaded (its package failed to import — "
                    f"see the import errors banner at the top of the app for the fix)."
                )
            return None, f"'{ext}' is not a supported file type."

        loader_class = self.SUPPORTED_EXTENSIONS[ext]
        if loader_class is _OCRImageLoader:
            return None, "Image files are read through the OCR pipeline, not a document loader."
        try:
            if ext == ".json":
                return loader_class(str(file_path), jq_schema=".", text_content=False), None
            else:
                return loader_class(str(file_path)), None
        except Exception as e:
            logger.error(f"Error creating loader for {file_path}: {e}")
            return None, f"Could not open file: {e}"

    def _ocr_languages(self):
        """Resolve the configured OCR languages against installed packs.
        Returns (languages, error). A missing pack is skipped with a warning;
        only when none of the requested packs exist is an error returned."""
        requested = self.config.config.get("ocr_languages", "fas+eng")
        languages, missing = _resolve_ocr_languages(requested)
        if missing:
            logger.warning(
                f"OCR language pack(s) not installed and skipped: {', '.join(missing)}. "
                f"{OCR_LANG_INSTALL_HINT}"
            )
        if not languages:
            return None, (
                f"OCR was attempted but none of the requested language packs "
                f"('{requested}') are installed for tesseract. {OCR_LANG_INSTALL_HINT}"
            )
        return languages, None

    def _ocr_pil_image(self, image, languages: str):
        """OCR one PIL image. Returns (text, error); error is set only when the
        OCR engine itself could not run."""
        persian = "fas" in languages.split("+")
        try:
            text = Pytesseract.image_to_string(
                _prepare_ocr_image(image), lang=languages, config="--oem 1 --psm 3"
            )
        except Pytesseract.pytesseract.TesseractNotFoundError:
            return "", (
                "OCR was attempted but the tesseract-ocr engine isn't installed "
                "(or isn't on PATH) on this machine. Install the system package — "
                "e.g. 'apt install tesseract-ocr' on Debian/Ubuntu, 'brew install "
                "tesseract' on macOS — then restart the app. This is separate from "
                "the pytesseract Python package, which can import fine without it."
            )
        except Pytesseract.pytesseract.TesseractError as e:
            msg = str(e)
            if "Failed loading language" in msg or "language" in msg.lower():
                return "", (
                    f"OCR was attempted but failed loading language pack '{languages}'. "
                    f"{OCR_LANG_INSTALL_HINT} Or change the OCR language in Settings. "
                    f"Detail: {msg}"
                )
            return "", f"OCR engine error: {msg}"
        return _clean_ocr_text(text, persian), None

    def _ocr_pdf(self, file_path: Path):
        """Render each page of a scanned/image-only PDF and run OCR on it.
        Used as a fallback only when the normal text-layer extraction
        (PyPDFLoader) comes back empty — most PDFs have a real text layer
        and never touch this path, so it costs nothing for the common case.
        Returns (documents, error_reason). error_reason is None on success
        (even if it just found no text) — it's only set when OCR itself
        could not run at all, so the real cause (most commonly: the
        tesseract-ocr *system* binary isn't installed, even though the
        pytesseract *Python package* imported fine) reaches the UI instead
        of being indistinguishable from 'this file has no text'."""
        if not OCR_AVAILABLE or Document is None:
            return [], None
        languages, lang_error = self._ocr_languages()
        if lang_error:
            return [], lang_error
        try:
            dpi = max(100, min(400, int(self.config.config.get("ocr_dpi", 250))))
        except (TypeError, ValueError):
            dpi = 250
        zoom = dpi / 72.0
        docs = []
        try:
            pdf = PyMuPDFModule.open(str(file_path))
            try:
                from io import BytesIO
                from PIL import Image
                for page_index in range(pdf.page_count):
                    page = pdf.load_page(page_index)
                    pix = page.get_pixmap(
                        matrix=PyMuPDFModule.Matrix(zoom, zoom),
                        colorspace=PyMuPDFModule.csGRAY,
                    )
                    image = Image.open(BytesIO(pix.tobytes("png")))
                    text, error = self._ocr_pil_image(image, languages)
                    if error:
                        return [], error
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"page": page_index, "ocr": True},
                            )
                        )
            finally:
                pdf.close()
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}")
            return [], f"OCR failed: {type(e).__name__}: {e}"
        return docs, None

    def _ocr_image_file(self, file_path: Path):
        """OCR an image file (PNG/JPG/TIFF/...). Multi-page TIFFs yield one
        document per frame. Returns (documents, error_reason) like _ocr_pdf."""
        if not OCR_AVAILABLE or Document is None:
            return [], "OCR isn't available (install pymupdf + pytesseract + the tesseract-ocr system package)."
        languages, lang_error = self._ocr_languages()
        if lang_error:
            return [], lang_error
        docs = []
        try:
            from PIL import Image, ImageSequence
            with Image.open(str(file_path)) as img:
                for index, frame in enumerate(ImageSequence.Iterator(img)):
                    text, error = self._ocr_pil_image(frame.copy(), languages)
                    if error:
                        return [], error
                    if text:
                        docs.append(
                            Document(page_content=text, metadata={"page": index, "ocr": True})
                        )
        except Exception as e:
            logger.error(f"Image OCR failed for {file_path}: {e}")
            return [], f"OCR failed: {type(e).__name__}: {e}"
        return docs, None

    def load_documents(self, file_paths: List[Path], progress_callback=None):
        all_documents = []
        failures = []
        total_files = len(file_paths)

        for idx, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(idx, total_files, f"Loading: {file_path.name}")

            try:
                if file_path.suffix.lower() in OCR_IMAGE_EXTENSIONS:
                    documents, ocr_error = self._ocr_image_file(file_path)
                    if not documents:
                        failures.append((
                            file_path.name,
                            ocr_error or "OCR ran but found no readable text in this image.",
                        ))
                        continue
                    for doc in documents:
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["file_name"] = file_path.name
                    all_documents.extend(documents)
                    logger.info(f"OCR read {len(documents)} page(s) from {file_path.name}")
                    continue

                loader, error_reason = self.get_loader(file_path)
                if loader:
                    # Prefer the generator-based lazy_load() when the loader
                    # supports it (most langchain_community loaders do) —
                    # it streams pages/records instead of materializing the
                    # whole parsed document in memory at once, which matters
                    # for very large individual files (e.g. a multi-GB PDF).
                    extraction_error = None
                    try:
                        if hasattr(loader, "lazy_load"):
                            documents = list(loader.lazy_load())
                        else:
                            documents = loader.load()
                        documents = [d for d in documents if d.page_content and d.page_content.strip()]
                    except Exception as e:
                        # pypdf can outright crash on certain malformed PDFs —
                        # most commonly a KeyError on a font table field (e.g.
                        # '/DescendantFonts') in PDFs with broken embedded
                        # composite fonts. This is a parser bug, not something
                        # fixable by reconfiguring anything — but OCR renders
                        # the page as a flat image and never touches font
                        # tables at all, so it's still a valid recovery path
                        # for a PDF, same as a plain empty-text extraction.
                        documents = []
                        extraction_error = f"{type(e).__name__}: {e}"
                        logger.warning(f"Text extraction raised an error for {file_path.name}: {extraction_error}")

                    if not documents:
                        ocr_used = False
                        ocr_error = None
                        if (
                            file_path.suffix.lower() == ".pdf"
                            and OCR_AVAILABLE
                            and self.config.config.get("ocr_enabled", True)
                        ):
                            documents, ocr_error = self._ocr_pdf(file_path)
                            ocr_used = bool(documents)
                        if not documents:
                            if ocr_error:
                                reason = ocr_error
                            elif extraction_error:
                                reason = f"Text extraction failed ({extraction_error})."
                                if file_path.suffix.lower() == ".pdf" and not OCR_AVAILABLE:
                                    reason += (
                                        " This is a known pypdf limitation with certain PDFs' "
                                        "embedded fonts, not a problem with the file itself — "
                                        "install pymupdf + pytesseract (and the tesseract-ocr "
                                        "system package) so OCR can recover this file instead."
                                    )
                                elif file_path.suffix.lower() == ".pdf" and OCR_AVAILABLE:
                                    reason += " OCR fallback also could not recover readable text."
                            else:
                                reason = "Loader ran but returned no usable text (file may be empty, image-only, or corrupted)."
                                if file_path.suffix.lower() == ".pdf" and not OCR_AVAILABLE:
                                    reason += " This looks like it may be a scanned PDF — install pymupdf + pytesseract (and the tesseract-ocr system package) to enable OCR fallback."
                                elif file_path.suffix.lower() == ".pdf" and OCR_AVAILABLE:
                                    reason += " OCR ran but found no readable text on any page."
                            failures.append((file_path.name, reason))
                            continue
                        if ocr_used:
                            reason_note = " (recovered via OCR after normal text extraction failed)" if extraction_error else ""
                            logger.info(f"OCR fallback recovered text from {file_path.name}{reason_note}")
                    for doc in documents:
                        doc.metadata["source_file"] = str(file_path)
                        doc.metadata["file_name"] = file_path.name
                    all_documents.extend(documents)
                    logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
                else:
                    failures.append((file_path.name, error_reason or "Unknown reason"))
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                failures.append((file_path.name, f"{type(e).__name__}: {e}"))
                continue

        return all_documents, failures

    def split_documents(self, documents):
        return self.text_splitter.split_documents(documents)

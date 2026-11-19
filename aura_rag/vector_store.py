"""Chroma vector store management."""

import logging
import shutil
from typing import List, Optional

from .compute import available_gpu
from .config import Config, resolve_local_model
from .dependencies import Chroma, HuggingFaceEmbeddings
from .network import is_offline_mode

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages the vector store operations."""

    def __init__(self, config: Config):
        self.config = config
        self.embeddings = None
        self.vectorstore = None
        self.embedding_error = None
        self.compute_warning = None
        self.active_device = "cpu"
        self.active_device_name = "CPU"
        self._initialize_embeddings()

    def _initialize_embeddings(self, model_name: str = None):
        if HuggingFaceEmbeddings is None:
            self.embeddings = None
            self.embedding_error = "langchain-huggingface is not installed. Run: uv sync --locked"
            logger.error(self.embedding_error)
            return False

        name = model_name or self.config.config["embedding_model"]
        resolved = resolve_local_model(name, self.config.models_dir, "embeddings")
        self.embeddings = None
        self.vectorstore = None
        self.compute_warning = None

        mode = self.config.config.get("embedding_compute_mode", "auto")
        gpu = available_gpu() if mode != "cpu" else None
        if gpu:
            target_device, target_name = gpu
        else:
            target_device, target_name = "cpu", "CPU"
            if mode == "gpu":
                self.compute_warning = (
                    "GPU mode was selected, but PyTorch cannot access a supported GPU. "
                    "For NVIDIA CUDA, install the GPU build with `uv sync --locked --extra cuda`, "
                    "then restart the app. Using CPU for now."
                )

        devices_to_try = [(target_device, target_name)]
        if target_device != "cpu":
            devices_to_try.append(("cpu", "CPU"))

        last_error = None
        for device, device_name in devices_to_try:
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=resolved,
                    model_kwargs={"device": device},
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.active_device = device
                self.active_device_name = device_name
                self.embedding_error = None
                if device == "cpu" and target_device != "cpu":
                    self.compute_warning = (
                        f"GPU model initialization failed ({type(last_error).__name__}: {last_error}); "
                        "embeddings are running on CPU."
                    )
                logger.info("Embeddings initialized successfully (%s) on %s", resolved, device_name)
                return True
            except Exception as exc:
                last_error = exc
                logger.error("Error initializing embeddings on %s: %s", device_name, exc)

        self.active_device = "cpu"
        self.active_device_name = "CPU"
        if is_offline_mode():
            self.embedding_error = (
                f"Embedding model '{name}' is unavailable locally while Offline mode is active. "
                f"Copy the complete model folder into {self.config.models_dir / 'embeddings'}, "
                "choose a local model in Settings, or switch to Online mode to download it. "
                f"Original error: {type(last_error).__name__}: {last_error}"
            )
        else:
            self.embedding_error = f"Could not load embedding model '{name}': {type(last_error).__name__}: {last_error}"
        return False

    def create_or_update_vectorstore(self, documents, progress_callback=None):
        if self.embeddings is None:
            logger.error("Cannot index documents: %s", self.embedding_error or "embedding model is not loaded")
            return False
        if Chroma is None:
            logger.error("langchain-chroma is not installed.")
            return False
        try:
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=str(self.config.db_dir),
                    collection_name="documents",
                )
                logger.info("Created new vector store with %s documents", len(documents))
            else:
                self.vectorstore.add_documents(documents)
                logger.info("Added %s documents to existing vector store", len(documents))
            return True
        except Exception:
            logger.exception("Error creating/updating vector store")
            return False

    def load_vectorstore(self):
        if self.embeddings is None:
            self.vectorstore = None
            return False
        if Chroma is None:
            return False
        try:
            if self.config.db_dir.exists():
                self.vectorstore = Chroma(
                    persist_directory=str(self.config.db_dir),
                    embedding_function=self.embeddings,
                    collection_name="documents",
                )
                count = self.vectorstore._collection.count()
                logger.info("Loaded vector store with %s documents", count)
                return True
        except Exception:
            logger.exception("Error loading vector store")
        return False

    def get_retriever(self, k=5, file_names: Optional[List[str]] = None):
        """Return a vector retriever, optionally restricted to filenames."""
        if self.vectorstore:
            search_kwargs = {"k": k}
            if file_names:
                search_kwargs["filter"] = {"file_name": {"$in": list(file_names)}}
            return self.vectorstore.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        return None

    def delete_documents_by_filename(self, file_name: str) -> bool:
        if not self.vectorstore:
            return False
        try:
            self.vectorstore._collection.delete(where={"file_name": file_name})
            return True
        except Exception:
            logger.exception("Error deleting chunks for %s", file_name)
            return False

    def clear_vectorstore(self):
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
            if self.config.db_dir.exists():
                shutil.rmtree(self.config.db_dir)
                self.config.db_dir.mkdir(exist_ok=True)
            self.vectorstore = None
            return True
        except Exception:
            logger.exception("Error clearing vector store")
            return False

    def get_document_count(self):
        if self.vectorstore:
            try:
                return self.vectorstore._collection.count()
            except Exception:
                pass
        return 0

from pathlib import Path
from typing import List, Union
from tqdm.auto import tqdm
import fitz
from langchain.text_splitter import CharacterTextSplitter
from loguru import logger


class PDFSplitter:
    def __init__(self, chunk_overlap: int = 200) -> None:
        self.chunk_overlap = chunk_overlap

    def split_document(
        self, document_path: Union[str, Path], max_size: int, **kwargs
    ) -> List[dict]:

        logger.info(f"Partitioning document: {document_path}")

        all_chunks = []
        splitter = CharacterTextSplitter(
            separator="\n",
            keep_separator=True,
            chunk_size=max_size,
            chunk_overlap=self.chunk_overlap,
        )

        doc = fitz.open(document_path)
        current_text = ""
        for page in doc:
            text = page.get_text("block")

            if len(text) > max_size:
                all_chunks.append(
                    {"text": current_text, "metadata": {"page": page.number}}
                )
                chunks = splitter.split_text(text)
                for chunk in chunks:
                    logger.info(
                        f"Flushing chunk. Length: {len(chunk)}, page: {page.number}"
                    )
                    all_chunks.append(
                        {"text": chunk, "metadata": {"page": page.number}}
                    )
                current_text = ""

            elif len(current_text + text) >= max_size:
                if current_text != "":
                    all_chunks.append(
                        {"text": current_text, "metadata": {"page": page.number}}
                    )
                logger.info(
                    f"Flushing chunk. Length: {len(current_text)}, page: {page.number}"
                )
                current_text = text

            else:
                current_text += text

        # Filter out empty docs
        all_chunks = [
            chunk for chunk in all_chunks if chunk["text"].strip().replace(" ", "")
        ]
        return all_chunks
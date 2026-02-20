import logging
import os
import sys
from typing import List, Dict, Any, Optional
import PyPDF2 # type: ignore
from functools import lru_cache
import asyncio
from llama_index.core import VectorStoreIndex, StorageContext, LlamaSettings # pyright: ignore[reportMissingImports]
from llama_index.core import SimpleDirectoryReader, ServiceContext # pyright: ignore[reportMissingImports]
from llama_index.core import VectorStoreIndex, Settings as LlamaSettings # type: ignore
from llama_index.vector_stores.chroma import ChromaVectorStore # pyright: ignore[reportMissingImports]
from fastapi import FastAPI, HTTPException # type: ignore
from pydantic import BaseModel # type: ignore

LOG_LEVEL = logging.INFO
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHUNK_SIZE = 256
MAX_DOCUMENTS = 100


logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

#defining supported file extensions
SUPPORTED_FILE_EXTENSIONS = {'.txt', '.pdf', '.docx', '.xlsx', '.pptx', '.csv', '.json', '.xml', '.html', '.md'}

def is_supported_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in SUPPORTED_FILE_EXTENSIONS

def scan_data_directory(data_dir: str) -> List[Dict[str, Any]]:
    documents = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if is_supported_file(file):
                file_path = os.path.join(root, file)
                documents.append({
                    "file_name": file,
                    "file_path": file_path
                })
            else:
                logger.warning(f"Unsupported file type skipped: {file}")
    return documents

##loading the docs
def load_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    loaded_docs = []
    for doc in documents:
        try:
            with open(doc["file_path"], 'r', encoding='utf-8') as f:
                content = f.read()
                loaded_docs.append({
                    "file_name": doc["file_name"],
                    "content": content
                })
        except Exception as e:
            logger.error(f"Error loading document {doc['file_name']}: {e}")
    return loaded_docs

##index and batching
MAX_DOCUMENTS_PER_BATCH = 50

async def add_documents_to_index_in_batches(documents, storage_context: StorageContext):
    batch_size = MAX_DOCUMENTS_PER_BATCH
    index = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if index is None:
            index = VectorStoreIndex.from_documents(
                batch,
                storage_context=storage_context,
                embed_model=LlamaSettings.embed_model
            )
        else:
            for doc in batch:
                index.insert(doc)
        await asyncio.sleep(1)  # Prevent overload or rate limiting
    return index


##naive query engine setup
class GetOrCreateCollectionAsync:
    def __init__(self, client, collection_name):
        self.client = client
        self.collection_name = collection_name


class GetChromaClientAsync:
    def __init__(self):
        pass


async def get_chroma_client_async():
    # Simulated async client creation
    return "chroma_client"


async def get_or_create_collection_async_fn(client, collection_name):
    return GetOrCreateCollectionAsync(client, collection_name)


async def get_naive_query_engine_async(collection_name: str, similarity_top_k: int = 6):
    client = await get_chroma_client_async()  # Assume defined elsewhere
    collection = await get_or_create_collection_async_fn(client, collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=LlamaSettings.embed_model
    )
    return index.as_query_engine(
        similarity_top_k=similarity_top_k,
        response_mode="compact",  # Condenses response for efficiency
        verbose=True  # Logs retrieval details
    )
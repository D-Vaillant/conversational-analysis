# reads the friends corpus and returns it as nodes.
import json
import os
import logging

from pathlib import Path
from typing import Dict, List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, Document

from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_integrations import UtteranceNodeParser, UtteranceReader

USING_OPENAI = True
i = 2

if USING_OPENAI:
    COLLECTION_NAME = f'openai_0{i}'
    DIM = 1536
    OPENAI_MODEL="text-embedding-3-small"
    embedding_model = OpenAIEmbedding(model=OPENAI_MODEL)
else:
    # WARNING: Untested, because I couldn't get it to use my GPU.
    COLLECTION_NAME = f'ollama_0{i}'
    DIM = 1024
    embedding_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        dimensionality=512
    )


utterances = Path('./data/friends-corpus/friends-corpus/utterances.jsonl')
conversations = Path("./data/friends-corpus/friends-corpus/conversations.json")

# Can be replaced with some other embedding model.


if __name__ == "__main__":
    logger = logging.Logger(name="ingestion", level=logging.DEBUG)
    # Loads 67k nodes in a few seconds.
    # The embedding takes minutes, because embedding takes a while.
    convos = UtteranceReader().load_data(utterances)
    pipeline = IngestionPipeline(
        transformations=[UtteranceNodeParser(),
                         embedding_model]
    )
    if os.path.exists("./pipeline_storage"):
        pipeline.load("./pipeline_storage")
    nodes = pipeline.run(documents=convos, show_progress=True)
    pipeline.persist("./pipeline_storage")
    
    vector_store = MilvusVectorStore(collection_name=COLLECTION_NAME,
        uri="http://localhost:19530", dim=1536, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    logger.info("We are now storing all our nodes into Milvus. This may take some time.")
    index = VectorStoreIndex(nodes, storage_context=storage_context)
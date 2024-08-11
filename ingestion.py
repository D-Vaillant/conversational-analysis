# reads the friends corpus and returns it as nodes.
import json
from pathlib import Path
from typing import Dict, List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, Document

from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore

from llama_integrations import UtteranceNodeParser, UtteranceReader


OPENAI_MODEL="text-embedding-3-small"

# We rely on LlamaIndex to be fast here.
utterances = Path('./data/friends-corpus/friends-corpus/utterances.jsonl')
conversations = Path("./data/friends-corpus/friends-corpus/conversations.json")


if __name__ == "__main__":
    # Loads 67k nodes in a few seconds.
    convos = UtteranceReader().load_data(utterances)
    pipeline = IngestionPipeline(
        transformations=[UtteranceNodeParser(),
                         OpenAIEmbedding(model=OPENAI_MODEL)]
    )
    nodes = pipeline.run(documents=convos, show_progress=True)
    
    vector_store = MilvusVectorStore(
        uri="http://localhost:19530", dim=1536, overwrite=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
# reads the friends corpus and returns it as nodes.
from collections import deque
from pathlib import Path
import json
from typing import Dict, List, Optional
from fsspec import AbstractFileSystem

from llama_index.core import SimpleDirectoryReader
# from llama_index.core.readers import JSONReader
from llama_index.core.node_parser import JSONNodeParser
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, Document
from llama_index.core.readers.base import BaseReader

from llama_index.core.ingestion import IngestionPipeline
from llama_index.readers.json import JSONReader
from llama_index.vector_stores.milvus import MilvusVectorStore

# We rely on LlamaIndex to be fast here.
utterances = Path('./data/friends-corpus/friends-corpus/utterances.jsonl')
conversations = Path("./data/friends-corpus/friends-corpus/conversations.json")
# json_parser = JSONNodeParser()
# pipeline = IngestionPipeline(
#     transformations=[json_parser]
# )

class UtteranceReader(BaseReader):
    def __init__(self, *args,
                 ignore_transcript_notes: bool=False,
                 ignore_alls: bool=True,
                 ignore_other: bool=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._ignore_transcript_notes = ignore_transcript_notes
        self._ignore_alls = ignore_alls
        self._ignore_other = ignore_other
        
    def create_document(self, id_, context, metadata):
        return Document(id_=id_,
                        text='\n'.join(context),
                        metadata=metadata)
        
    def load_data(self, file: Path,
                  extra_info: Optional[Dict] = None,
                  fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse the jsonl file into separate Documents for each conversation."""
        # NOTE: This part doesn't scale particularly well, since it loads all of the conversations
        # into our memory all at once. Unfortunately this is also what most LlamaIndex Readers tend to do!
        all_convos = []

        current_conversation_id = None
        current_conversation_speakers = set()
        current_conversation_content = []
        with open(file) as fileobj:
            for file_line in fileobj:
                record = json.loads(file_line)
                if current_conversation_id is None:   # Only done on the first line.
                    current_conversation_id = record['conversation_id']

                if current_conversation_id != record['conversation_id']:  # New conversation started.
                    metadata = {"speakers": list(current_conversation_speakers)}  # NOTE: Could add season, episode information here.
                    all_convos.append(
                        self.create_document(id_=current_conversation_id,
                                             context=current_conversation_content,
                                             metadata=metadata)
                    )

                    # Reset information.
                    current_conversation_id = record['conversation_id']
                    current_conversation_speakers = set()
                    current_conversation_content = []

                speaker = record['speaker']
                # Handling various special speaker cases.
                # On ignores, we just skip the line altogether.
                # Otherwise: for All and Other we give them lines but don't add them to the cast.
                # For transcript notes we record the line differently.
                if speaker == 'TRANSCRIPT_NOTE':
                    if self._ignore_transcript_notes:
                        continue
                    else:
                        line = f"(NOTE: {record['text']})"
                else:
                    if speaker == '#ALL#':
                        if self._ignore_alls:
                            continue
                        else:
                            speaker = "ALL"
                    elif speaker == '#OTHER#':
                        if self._ignore_other:
                            continue
                        else:
                            speaker = "Other"
                    else:  # Default case.
                        current_conversation_speakers.add(speaker)
                    line = f"{speaker}: {record['text']}"

                if line is not None:
                    current_conversation_content.append(line)

            # End of loop. Clean up with the last conversation.
            metadata = {"speakers": list(current_conversation_speakers)}
            all_convos.append(
                self.create_document(id_=current_conversation_id,
                                        context=current_conversation_content,
                                        metadata=metadata)
            )
        return all_convos
                    

def create_utterance_nodes(convofile: Path):
    previous_node = None
    with convofile.open() as f:
        for line in f:
            u = json.loads(line)
            node_id = u['id']
            conversation_id = u['conversation_id']
            speaker_name = u['speaker']
            content = u['text']
            text = f"{speaker_name}: {content}"
            node = TextNode(text=text, id_=node_id)
                            
            if previous_node is not None:
                node.relationships[NodeRelationship.PREVIOUS] = previous_node.node_id
                previous_node.relationships[NodeRelationship.NEXT] = node.node_id

# convo = UtteranceReader().load_data(input_file=conversations)

# def create_sliding_window_nodes(convofile: Path, window_size: int = 3):
#     window = deque(maxlen=window_size)
#     current_conversation_id = None
#     with convofile.open() as f:
#         for line in f:
#             u = json.loads(line)
#             # new conversation
#             if u['conversation_id'] != current_conversation_id:
#                 window.clear()
#                 current_conversation_id = u['conversation_id']
#             else:
#             # add to existing conversation
#                 pass
            
            
# nodes = pipeline.run(
#     documents=convo,
#     show_progress=True
# )

# TODO: Adjust this URI so it works on a Docker container.
# Milvus Lite doesn't even work on anything that's not Mac or Ubuntu. Booo!!!
# vector_store = MilvusVectorStore(
#     uri="http://localhost:19530", dim=1536, overwrite=True
# )
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex(nodes, storage_context=storage_context)

if __name__ == "__main__":
    pass
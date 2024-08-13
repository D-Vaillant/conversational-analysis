"""Parses `utterances.jsonl` from the Cornell Friends dialogue dataset.
In theory, works for any other JSONL file in a similar format.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import BaseNode, MetadataMode, TextNode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable

from llama_index.core.schema import TextNode, Document
from llama_index.core.readers.base import BaseReader


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
 
    def load_data(self, file: Path,
                  extra_info: Optional[Dict] = None
    ) -> List[Document]:
        """Parse the jsonl file into separate Documents for each conversation."""
        # NOTE: This part doesn't scale particularly well, since it loads all of the conversations
        # into our memory all at once. Unfortunately this is also what most LlamaIndex file Readers tend to do!
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
                    season, episode, scene, _ = record['conversation_id'].split('_')
                    metadata = {"speakers": list(current_conversation_speakers),
                                "season": season, "episode": episode, "scene": scene}
                    all_convos.append(
                        Document(id_=current_conversation_id,
                                             text=json.dumps(current_conversation_content),
                                             metadata=metadata,
                                             excluded_embed_metadata_keys=['speakers'])
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
                if speaker == 'TRANSCRIPT_NOTE' and self._ignore_transcript_notes:
                    continue
                else:
                    if speaker == '#ALL#' and self._ignore_alls:
                        continue
                    elif speaker == '#OTHER#' and self._ignore_other:
                        continue
                    else:  # Default case.
                        current_conversation_speakers.add(speaker)
                    current_conversation_content.append(record)

            # End of loop. Clean up with the last conversation.
            season, episode, scene, _ = record['conversation_id'].split('_')
            season = int(season[1:])
            episode = int(episode[1:])
            scene = int(scene[1:])
            metadata = {"speakers": list(current_conversation_speakers),
                        "season": season, "episode": episode, "scene": scene}

            all_convos.append(
                Document(id_=current_conversation_id,
                        context=json.dumps(current_conversation_content),
                        metadata=metadata,
                        excluded_embed_metadata_keys=['speakers']))
        return all_convos
                    

class UtteranceNodeParser(NodeParser):
    """JSON node parser.

    Splits a document into Nodes using custom JSON splitting logic.

    Args:
        include_metadata (bool): whether to include metadata in nodes [nonfunc]
        include_prev_next_rel (bool): whether to include prev/next relationships

    """
    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "UtteranceNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "UtteranceNodeParser"

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.get_nodes_from_node(node)
            all_nodes.extend(nodes)

        return all_nodes

    def make_id_func(self, id_arr: List):
        def id_func(i: int, doc: BaseNode) -> str:
            return id_arr[i]['id']
        return id_func
        
    def get_nodes_from_node(self, node: BaseNode) -> List[TextNode]:
        """Get nodes from document."""
        toplevel_metadata = node.metadata
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Handle invalid JSON input here
            return []

        json_nodes = []
        
        if isinstance(data, list):
            previous_node = None
            for ix, json_object in enumerate(data):
                metadata_fields = ["season", "episode", "scene"]
                inherited_metadata = {field: toplevel_metadata[field] for field in metadata_fields}
                node = TextNode(id_=json_object['id'], text=json_object['text'],
                                metadata={'speaker': json_object['speaker'],
                                          'conversation_id': json_object['conversation_id'],
                                          **inherited_metadata})
                node.excluded_embed_metadata_keys = ['speaker']
            if previous_node is not None:
                node.relationships[NodeRelationship.PREVIOUS] = previous_node.node_id
                data[ix-1].relationships[NodeRelationship.NEXT] = node.node_id

            json_nodes.append(node)
            node = previous_node
                    
        else:
            raise ValueError("JSON is invalid")

        return json_nodes
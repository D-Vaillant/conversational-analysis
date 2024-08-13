# convokit_parser
import logging
import os

# from llama_index.core import JSONNodeParser
from dotenv import load_dotenv

# from engine import ContextChatEngine
from llama_index.core import Settings, load_index_from_storage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from tests import load_trivia_qa_pairs

load_dotenv()
logger = logging.getLogger(__name__)


vector_store = MilvusVectorStore(
    uri="http://localhost:19530", dim=1536, overwrite=False
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = load_index_from_storage(storage_context)


def doublecheck_answer(answer: str, response: str):
    # TODO: Could add some more complicated logic here.
    # Lowhanging fruit, but it costs some money, is checking with an LLM.
    pass


class OpenAIActor():
    def __init__(self):
        try:
            import openai
        except ImportError as e:
            logging.fatal("OpenAI not installed!")
            raise(e)
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def ask(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are one of the world's foremost experts on TV trivia. You will be asked questions about the television program, 'Friends'. Answer as briefly as possible."},
            {"role": "user", "content": question}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        llm_answer = response.choices[0].message.content
        return llm_answer


if __name__ == "__main__":
    actor = index.as_chat_engine()
    # actor = OpenAIActor()
    logging.basicConfig(level=logging.INFO)
    qa_pairs = load_trivia_qa_pairs()
    assert len(qa_pairs) > 0
    # verify keys
    for q, a in qa_pairs:
        print(f"Asking: {q}")
        llm_response = actor.ask(f"{q}")
        is_correct = (a == llm_response)
        if is_correct:
            print(f"CORRECT ANSWER: {a}")
        else:
            print(f"CORRECT: {a}")
            print(f"LLM: {llm_response}")
        print()
        # logger.warning(f"Answer: {a}")
    print("Done!")
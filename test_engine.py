# convokit_parser
import logging
import os

# from llama_index.core import JSONNodeParser
from dotenv import load_dotenv

# from engine import ContextChatEngine
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from openai import OpenAI


from tests import load_trivia_qa_pairs

load_dotenv()
logger = logging.getLogger(__name__)


# Conversational search.
# friends_knower = friends_index.as_chat_engine()


def doublecheck_answer(answer: str, response: str):
    # TODO: Could add some more complicated logic here.
    # Lowhanging fruit, but it costs some money, is checking with an LLM.
    pass


def ask_openai(client: OpenAI, question: str) -> str:
    try:
        import openai
    except ImportError as e:
        print("OpenAI not installed!")
        raise(e)

    messages = [
        {"role": "system", "content": "You are one of the world's foremost experts on TV trivia. You will be asked questions about the television program, 'Friends'. Answer as briefly as possible."},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    llm_answer = response.choices[0].message.content
    return llm_answer
    
#  is_correct = (answer == llm_answer) or doublecheck_answer(answer=answer, response=llm_answer)

class OpenAIActor():
    def __init__(self):
        self.provider = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
    def ask(self, question: str) -> str:
        return ask_openai(self.provider, question)


if __name__ == "__main__":
    # actor = ContextChatEngine.from_defaults()
    actor = OpenAIActor()
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
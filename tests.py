# tests.py
import logging
from collections import namedtuple
from typing import List

logger = logging.getLogger(__name__)
QAPair = namedtuple("QAPair", field_names=["question", "answer"])

# Loads from a text file sourced from https://parade.com/1061827/alexandra-hurtado/friends-trivia-questions/.
def load_trivia_qa_pairs() -> List[QAPair]:
    with open("./data/friends_trivia.txt") as trivia:
        # NOTE: Not scalable! But does for our purposes.
        # The "right way" to do this would be some kind of streaming thing,
        # but this just reads everything into memory.
        trivia_lines = trivia.readlines()
        qa_pairs = []
        found_question = None
        found_answer = None
        for ix, line in enumerate(trivia_lines):
            seps = line.split(':')[1:]
            if line.startswith("Question"):
                found_question = ':'.join(seps).strip()
            elif line.startswith("Answer"):
                found_answer = ':'.join(seps).strip()
                if found_question is not None:
                    if found_answer.strip():  # falsy, i.e. nonwhitespace
                        qa_pairs.append(QAPair(**{"question": found_question, "answer": found_answer}))
                        found_question = None
                        found_answer = None
                    else:
                        logger.warning(f"Question with blank answer on line {ix}. Question was: `{found_question}`, answer was: `{found_answer}`.")
                        logger.info("Skipping the question.")
                        found_question = None
                        found_answer = None
                else:
                    logger.warning(f"Answer without question on line {ix}. Answer: `{found_answer}`")
                    logger.info("Throwing it out.")
                    found_answer = None
            else:
                continue
    return qa_pairs
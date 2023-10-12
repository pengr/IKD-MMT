from typing import Iterable, Tuple, List, Dict, Pattern
import re
import stanza
from stanza.models.common.doc import Document, Sentence
import torch


NEWLINE_WHITESPACE_RE = re.compile(r'\n\s*\n')


def _stanza_batch(
    data: Iterable[str], batch_size: int = 32
) -> Iterable[Tuple[str, List[int], List[int]]]:
    """
    Batches text together so that Stanza can process them quicker, rather than
    giving Stanza one text at a time to process. The way this batches text
    together is by joining the texts together with `\n\n` as suggested in the
    Stanza documentation:
    https://stanfordnlp.github.io/stanza/pipeline.html#usage

    However it will split a given document into smaller documents within the
    batch using the following regular expression: re.compile('\\n\\s*\\n')
    Thus if your single document is `hello\n \nhow are you` this will be
    processed as two separate paragraphs e.g. `hello` and `how are you`. The
    list of document indexes that are produced as output will allow you to
    know if one of your documents has been split into two or more pieces e.g.
    in the last case the returned document indexes will be [0,0] as the two
    separate documents have come from the same one document that was the input.

    :param data: A list/iterable of texts you want tagging.
    :param batch_size: The number of texts to process at one time.
    :returns: The Tuple of length 3 where the first items is the batched up
              texts, the second are the character offsets that denoted the end
              of a text/paragraph within the batch and the last the list of
              document indexes.
    :raises ValueError: If a sample in the data contains no text after being
                        split using `re.compile('\\n\\s*\\n')` regular expression.
    """
    batch_str = ""
    current_batch_size = 0
    for sample in data:
        batch_str += f"{sample}"
        current_batch_size += 1
        if current_batch_size == batch_size:
            batch_str += "\n\n"
            yield batch_str
            batch_str = ""
            current_batch_size = 0
        else:
            batch_str += "\n\n"

    if batch_str:
        yield batch_str



def batch(
    data: Iterable[str],
    stanza_pipeline: stanza.Pipeline,
    batch_size: int = 32,
    clear_cache: bool = True,
    num_sent: int = 0,
) -> Iterable[Document]:

    for batch_str in _stanza_batch(
        data, batch_size=batch_size
    ):
        stanza_document = stanza_pipeline(batch_str)
        num_sent += len(stanza_document.sentences)
        print(f'Process {num_sent} sent')

        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        yield stanza_document

    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
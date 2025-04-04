# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing chunk strategies."""

from collections.abc import Iterable

import nltk
import tiktoken

from graphrag.config.models.chunking_config import ChunkingConfig
from graphrag.index.operations.chunk_text.typing import TextChunk
from graphrag.index.text_splitting.text_splitting import (
    Tokenizer,
    split_multiple_texts_on_tokens,
)
from graphrag.index.text_splitting.chinese_recursive_text_splitter import ChineseRecursiveTextSplitter
from graphrag.logger.progress import ProgressTicker


def get_encoding_fn(encoding_name):
    """Get the encoding model."""
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return encode, decode


def run_tokens(
    input: list[str],
    config: ChunkingConfig,
    tick: ProgressTicker,
) -> Iterable[TextChunk]:
    """Chunks text into chunks based on encoding tokens."""
    tokens_per_chunk = config.size
    chunk_overlap = config.overlap
    encoding_name = config.encoding_model

    encode, decode = get_encoding_fn(encoding_name)
    return split_multiple_texts_on_tokens(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


def run_sentences(
    input: list[str], _config: ChunkingConfig, tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into multiple parts by sentence."""
    for doc_idx, text in enumerate(input):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            yield TextChunk(
                text_chunk=sentence,
                source_doc_indices=[doc_idx],
            )
        tick(1)


def run_chinese(
    input: list[str], config: ChunkingConfig, tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into chunks based on encoding tokens."""
    tokens_per_chunk = config.size
    chunk_overlap = config.overlap
    encoding_name = config.encoding_model
    enc = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)

    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)

    return _split_text_to_chinese(
        input,
        Tokenizer(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=tokens_per_chunk,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


def _split_text_to_chinese(
    texts: list[str], enc: Tokenizer, tick: ProgressTicker
) -> list[TextChunk]:
    result = []
    for source_doc_idx, text in enumerate(texts):
        doc_indices = [source_doc_idx]
        
        text_splitter = ChineseRecursiveTextSplitter(
            keep_separator=True, is_separator_regex=True,
            chunk_size=enc.tokens_per_chunk, chunk_overlap=enc.chunk_overlap
        )

        chunks = text_splitter.split_text(text)
        for _, chunk in enumerate(chunks):
            # tick(1)

            encoded = enc.encode(chunk)
            n_tokens = len(encoded)
            # print(f"\n\n[_split_text_to_chinese] doc_indices:{doc_indices} "
            #     f"n_tokens:{n_tokens}\n{chunk}\n\n")
            result.append(
                TextChunk(
                    text_chunk=chunk,
                    source_doc_indices=doc_indices,
                    n_tokens=n_tokens,
                )
            )
    
    return result
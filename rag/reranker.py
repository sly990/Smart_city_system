# rag/reranker.py
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

import torch
from sentence_transformers import SentenceTransformer, util

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # 兼容旧版 langchain


RERANK_MODEL_PATH = os.getenv(
    "RERANK_MODEL_PATH",
    r"G:\BiShe\cyber\models\all-MiniLM-L6-v2"
)


@lru_cache(maxsize=1)
def get_rerank_model() -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(RERANK_MODEL_PATH, device=device)


def rerank_docs(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """
    对召回结果重排，返回 top_k。
    """
    print("进入重排序函数！")
    if not docs:
        return []

    model = get_rerank_model()
    doc_texts = [doc.page_content for doc in docs]

    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    sorted_indices = cosine_scores.argsort(descending=True).tolist()

    sorted_docs = [docs[i] for i in sorted_indices]
    return sorted_docs[:top_k]
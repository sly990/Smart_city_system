# rag/retriever.py
# 该模块负责构建、保存、加载向量存储和Elasticsearch索引，并提供混合检索功能（向量+关键词）

from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except Exception:
    from langchain.schema import Document  # 兼容旧版 langchain


# =========================
# 配置
# =========================
# 项目根目录：当前文件的上两级目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 向量存储的保存路径（默认为项目根目录下的vector_store/vector_store.pkl）
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", PROJECT_ROOT / "vector_store" / "vector_store.pkl"))
# Elasticsearch 索引名称（可通过环境变量 ES_INDEX_NAME 配置）
ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "rag_docs")
# Elasticsearch 连接地址（默认本地9200端口）
ES_URL = os.getenv("ES_URL", "http://localhost:9200")

# 嵌入模型路径（可通过环境变量 EMBED_MODEL_PATH 配置）
EMBED_MODEL_PATH = os.getenv(
    "EMBED_MODEL_PATH",
    r"G:\BiShe\cyber\models\xiaobu-embedding-v2"
)

# 如果你有自己的文档构建脚本，可以在这里接入
DEFAULT_DOC_PATH = os.getenv("RAG_DOC_PATH", "")


# =========================
# 简单向量存储兜底实现
# =========================
# 当 FAISS 无法使用时，使用内存中的简单向量存储作为备选方案
class InMemorySimpleVectorStore:
    """
    内存中的简单向量存储，用于在 FAISS 不可用时作为降级方案。
    实现了 similarity_search 方法，通过余弦相似度计算相似文档。
    """
    def __init__(self, embedding_function, texts, metadatas=None):
        """
        初始化内存向量存储。

        Args:
            embedding_function: 嵌入函数（通常为 HuggingFaceEmbeddings 实例）
            texts: 文本列表
            metadatas: 元数据列表，每个元素为字典
        """
        self.embedding_function = embedding_function
        self.texts = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.doc_embeddings = []  # 存储所有文档的嵌入向量

        # 对每个文本生成嵌入向量
        for text in texts:
            try:
                emb = embedding_function.embed_query(text)
                self.doc_embeddings.append(emb)
            except Exception:
                # 如果嵌入失败，用零向量填充（维度为768，可根据模型调整）
                self.doc_embeddings.append(np.zeros(768, dtype=np.float32))

    def similarity_search(self, query: str, k: int = 4):
        """
        基于余弦相似度检索最相似的 k 个文档。

        Args:
            query: 查询字符串
            k: 返回结果数量

        Returns:
            List[Document]: 相似文档列表
        """
        # 生成查询向量
        query_embedding = self.embedding_function.embed_query(query)
        query_vec = np.array(query_embedding, dtype=np.float32)

        # 计算每个文档与查询的余弦相似度
        similarities = []
        for doc_embedding in self.doc_embeddings:
            doc_vec = np.array(doc_embedding, dtype=np.float32)
            dot_product = float(np.dot(query_vec, doc_vec))
            query_norm = float(np.linalg.norm(query_vec))
            doc_norm = float(np.linalg.norm(doc_vec))
            similarity = dot_product / (query_norm * doc_norm) if query_norm > 0 and doc_norm > 0 else 0.0
            similarities.append(similarity)

        # 获取相似度最高的 k 个索引（降序）
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [
            Document(page_content=self.texts[i], metadata=self.metadatas[i])
            for i in top_indices
        ]


# =========================
# 基础对象
# =========================
@lru_cache(maxsize=1)
def get_embeddings():
    """
    获取嵌入模型实例（使用 LRU 缓存，确保单例）。
    根据 CUDA 环境变量决定是否使用 GPU。
    """
    # 判断设备：如果 CUDA_VISIBLE_DEVICES 有值则使用 GPU，否则 CPU
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    model_kwargs = {"device": device}
    # 返回 HuggingFaceEmbeddings 实例，启用向量归一化
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_PATH,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def _ensure_parent_dir(path: Path):
    """
    确保路径的父目录存在，若不存在则递归创建。
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def load_vector_store() -> Optional[Any]:
    """
    从本地文件加载向量存储。

    Returns:
        向量存储对象（FAISS 或 InMemorySimpleVectorStore），如果文件不存在或加载失败则返回 None。
    """
    if not VECTOR_STORE_PATH.exists():
        return None

    try:
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_vector_store(vector_store: Any) -> None:
    """
    将向量存储保存到本地文件。
    """
    _ensure_parent_dir(VECTOR_STORE_PATH)
    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)


def connect_es() -> Optional[Elasticsearch]:
    """
    连接 Elasticsearch 实例。

    Returns:
        Elasticsearch 客户端实例，若连接失败则返回 None。
    """
    try:
        es = Elasticsearch(ES_URL)
        es.info()  # 测试连接
        return es
    except Exception:
        return None


# =========================
# 构建索引
# =========================
def build_vector_store_from_documents(docs: List[Document]) -> Tuple[Optional[Any], Optional[Elasticsearch]]:
    """
    用 Document 列表构建本地向量库和 ES 索引。

    Args:
        docs: Document 对象列表，每个包含 page_content 和 metadata。

    Returns:
        tuple: (vector_store, es_client) 若成功则返回对象，否则 (None, None)
    """
    # 过滤掉空内容的文档
    if not docs:
        return None, None

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        return None, None

    embedder = get_embeddings()

    try:
        # 优先尝试使用 FAISS 构建向量库
        try:
            vector_store = FAISS.from_documents(docs, embedder)
        except Exception:
            # 如果 FAISS 失败，使用内存简单向量库作为降级方案
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            vector_store = InMemorySimpleVectorStore(embedder, texts, metadatas)

        # 保存向量库到本地
        save_vector_store(vector_store)

        # 尝试连接 ES 并构建索引（ES 不可用时不阻断主流程）
        es = connect_es()
        if es:
            _build_es_index(es, docs)

        return vector_store, es
    except Exception:
        # 任何异常都返回 None，避免影响上层调用
        return None, None


def _build_es_index(es: Elasticsearch, docs: List[Document]) -> None:
    """
    在 Elasticsearch 中构建索引，批量索引文档。

    Args:
        es: Elasticsearch 客户端
        docs: Document 对象列表
    """
    try:
        # 如果索引不存在则创建
        if not es.indices.exists(index=ES_INDEX_NAME):
            es.indices.create(index=ES_INDEX_NAME)

        actions = []
        for i, doc in enumerate(docs):
            actions.append(
                {
                    "_index": ES_INDEX_NAME,
                    "_id": i,
                    "_source": {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "filename": doc.metadata.get("filename", ""),
                        "chunk_id": doc.metadata.get("chunk_id", -1),
                    },
                }
            )

        # 批量索引
        if actions:
            bulk(es, actions)
    except Exception:
        # ES 不可用时不影响主流程
        pass


def load_or_build_resources(docs: Optional[List[Document]] = None) -> Tuple[Optional[Any], Optional[Elasticsearch]]:
    """
    优先加载已存在的向量库；如果没有，则用提供的 docs 构建。

    Args:
        docs: 可选，用于构建的文档列表

    Returns:
        tuple: (vector_store, es_client)
    """
    # 尝试加载已有的向量库
    vector_store = load_vector_store()
    es = connect_es()

    # 如果向量库已存在，直接返回
    if vector_store is not None:
        return vector_store, es

    # 否则尝试用文档构建
    if docs:
        return build_vector_store_from_documents(docs)

    # 无可用资源
    return None, es


# =========================
# 检索
# =========================
def keyword_search(es: Optional[Elasticsearch], query: str, top_k: int = 5) -> List[Document]:
    """
    基于 Elasticsearch 的关键词检索（match 查询）。

    Args:
        es: Elasticsearch 客户端
        query: 查询字符串
        top_k: 返回结果数量

    Returns:
        List[Document]: 检索到的文档列表，如果 ES 不可用或出错则返回空列表
    """
    if not es:
        return []

    try:
        res = es.search(
            index=ES_INDEX_NAME,
            query={"match": {"content": query}},
            size=top_k,
        )
        return [
            Document(
                page_content=hit["_source"]["content"],
                metadata={
                    "source": hit["_source"].get("source", "unknown"),
                    "filename": hit["_source"].get("filename", ""),
                    "chunk_id": hit["_source"].get("chunk_id", -1),
                    "retriever": "keyword",  # 标记来源为关键词检索
                },
            )
            for hit in res["hits"]["hits"]
        ]
    except Exception:
        return []


def vector_search(vector_store: Optional[Any], query: str, k: int = 5) -> List[Document]:
    """
    基于向量存储的相似性检索。

    Args:
        vector_store: 向量存储对象（FAISS 或 InMemorySimpleVectorStore）
        query: 查询字符串
        k: 返回结果数量

    Returns:
        List[Document]: 检索到的文档列表，如果向量库不可用则返回空列表
    """
    if not vector_store:
        return []

    try:
        return vector_store.similarity_search(query, k=k)
    except Exception:
        return []


def deduplicate_docs(docs: List[Document]) -> List[Document]:
    """
    对文档列表进行去重，基于 source, filename, chunk_id 和内容前200字符的组合作为唯一键。

    Args:
        docs: 文档列表

    Returns:
        List[Document]: 去重后的文档列表
    """
    seen = set()
    unique_docs = []

    for doc in docs:
        md = doc.metadata or {}
        # 构造唯一键
        key = (
            md.get("source", ""),
            md.get("filename", ""),
            md.get("chunk_id", -1),
            doc.page_content[:200],
        )
        if key in seen:
            continue
        seen.add(key)
        unique_docs.append(doc)

    return unique_docs


def local_hybrid_search(
    query: str,
    vector_store: Optional[Any],
    es: Optional[Elasticsearch],
    k: int = 5,
) -> List[Document]:
    """
    混合召回：向量召回 + 关键词召回，然后去重。

    Args:
        query: 查询字符串
        vector_store: 向量存储对象
        es: Elasticsearch 客户端
        k: 每种召回方式返回的结果数量（最终结果会合并去重）

    Returns:
        List[Document]: 合并去重后的文档列表
    """
    local_docs: List[Document] = []

    # 向量召回
    local_docs.extend(vector_search(vector_store, query, k=k))
    print("向量召回成功！")
    # 关键词召回
    local_docs.extend(keyword_search(es, query, top_k=k))
    print("关键词召回成功！")

    return deduplicate_docs(local_docs)


def hybrid_retrieve(
    query: str,
    vector_store: Optional[Any] = None,
    es: Optional[Elasticsearch] = None,
    k: int = 10,
    docs_for_build: Optional[List[Document]] = None,
) -> List[Document]:
    """
    标准检索入口：
    1. 如果未提供 vector_store 和 es，则尝试加载或构建资源
    2. 执行混合召回

    Args:
        query: 查询字符串
        vector_store: 可选，外部传入的向量存储
        es: 可选，外部传入的 ES 客户端
        k: 每种召回返回的结果数（最终混合结果会多于 k，由去重决定）
        docs_for_build: 可选，若需要构建资源时使用的文档列表

    Returns:
        List[Document]: 检索结果列表
    """
    # 如果没有提供资源，尝试加载或构建
    if vector_store is None and es is None:
        vector_store, es = load_or_build_resources(docs_for_build)

    # 执行混合搜索
    return local_hybrid_search(query, vector_store, es, k=k)
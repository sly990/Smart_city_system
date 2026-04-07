import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer


def load_txt_files(txt_dir: str) -> List[Document]:
    """
    读取目录下所有 txt 文件，返回 LangChain Document 列表
    """
    txt_path = Path(txt_dir)
    if not txt_path.exists():
        raise FileNotFoundError(f"目录不存在: {txt_dir}")

    documents: List[Document] = []
    for file_path in sorted(txt_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                },
            )
        )
    return documents


def split_to_paragraphs(text: str) -> List[str]:
    """
    先按空行切段；如果没有空行，再按中文句号/分号等做兜底切分
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paras) >= 2:
        return paras

    # 兜底：按中文/英文句末标点切句
    sentences = re.split(r"(?<=[。！？!?；;])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦相似度
    """
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def semantic_chunk_text(
    text: str,
    model: SentenceTransformer,
    similarity_threshold: float = 0.55,
    max_chars: int = 800,
    min_chars: int = 120,
) -> List[str]:
    """
    基于 embedding 的语义分块：
    - 先切成段落/句子
    - 计算每个单元的向量
    - 依次尝试与当前块合并
    - 若语义相似度低于阈值，或块太长，则开启新块
    """
    units = split_to_paragraphs(text)
    if not units:
        return []

    # 用模型生成向量，normalize 后更适合直接算余弦
    emb = model.encode(units, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    chunks: List[str] = []
    current_units: List[str] = [units[0]]
    current_vec = emb[0].copy()
    current_len = len(units[0])

    for i in range(1, len(units)):
        candidate = units[i]
        candidate_vec = emb[i]

        # 用“当前块中心向量”与“候选单元向量”比较
        sim = cosine_sim(current_vec, candidate_vec)
        would_be_too_long = current_len + len(candidate) > max_chars

        if sim >= similarity_threshold and not would_be_too_long:
            current_units.append(candidate)

            # 更新当前块中心向量（简单平均）
            current_vec = np.mean(np.vstack([current_vec, candidate_vec]), axis=0)
            current_len += len(candidate)
        else:
            chunk_text = "\n".join(current_units).strip()
            if chunk_text:
                chunks.append(chunk_text)

            current_units = [candidate]
            current_vec = candidate_vec.copy()
            current_len = len(candidate)

    last_chunk = "\n".join(current_units).strip()
    if last_chunk:
        chunks.append(last_chunk)

    # 对太短的块做一次轻微合并，避免碎片化
    merged: List[str] = []
    for ch in chunks:
        if merged and len(ch) < min_chars:
            merged[-1] = merged[-1] + "\n" + ch
        else:
            merged.append(ch)

    return merged


def split_documents_semantic(
    documents: List[Document],
    model_name: str = "BAAI/bge-small-zh-v1.5",
    similarity_threshold: float = 0.55,
    max_chars: int = 800,
    min_chars: int = 120,
) -> List[Document]:
    """
    对多个 txt 文档做语义分块，保留源文件信息
    """
    model = SentenceTransformer(model_name)

    all_chunks: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source", "")
        filename = doc.metadata.get("filename", os.path.basename(source))

        chunks = semantic_chunk_text(
            doc.page_content,
            model=model,
            similarity_threshold=similarity_threshold,
            max_chars=max_chars,
            min_chars=min_chars,
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "filename": filename,
                        "chunk_id": idx,
                        "chunk_count": len(chunks),
                    },
                )
            )

    return all_chunks


def load_and_split_txt_dir(
    txt_dir: str,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    similarity_threshold: float = 0.55,
    max_chars: int = 800,
    min_chars: int = 120,
) -> List[Document]:
    """
    主入口：读取目录下所有 txt 文件并语义分块
    """
    docs = load_txt_files(txt_dir)
    if not docs:
        print(f"没有找到可处理的 txt 文件: {txt_dir}")
        return []

    print(f"读取到 {len(docs)} 个 txt 文件")
    chunks = split_documents_semantic(
        docs,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    print(f"语义分块完成，共生成 {len(chunks)} 个 chunk")
    return chunks


if __name__ == "__main__":
    txt_dir = r"G:\Agent\customer-service-ai-agent\Smart_City"

    chunks = load_and_split_txt_dir(
        txt_dir=txt_dir,
        model_name="BAAI/bge-small-zh-v1.5",   # 中文语义向量模型
        similarity_threshold=0.55,             # 越大越保守，块越碎
        max_chars=800,                         # 单块最大长度
        min_chars=120,                         # 太短的块尽量合并
    )

    print("\n前几个块示例：")
    for i in range(min(5, len(chunks))):
        print(f"\n--- chunk {i} ---")
        print(chunks[i].metadata)
        print(chunks[i].page_content[:500])
        
        
# python chunk_smart_city.py
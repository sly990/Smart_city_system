# rag/ingest.py
# 该模块用于将文本目录中的文档进行分块并构建向量存储

from pathlib import Path
from typing import Optional

# 导入自定义的分块函数
from chunk_smart_city import load_and_split_txt_dir
# 导入构建向量存储的函数
from retriever import build_vector_store_from_documents


def ingest_txt_directory(
    txt_dir: str,
    model_name: str = "BAAI/bge-small-zh-v1.5",
    similarity_threshold: float = 0.55,
    max_chars: int = 800,
    min_chars: int = 120,
):
    """
    将指定目录下的所有txt文件进行智能分块，并构建向量存储和Elasticsearch索引。

    Args:
        txt_dir (str): 包含txt文件的目录路径。
        model_name (str): 用于分块的嵌入模型名称，默认为"BAAI/bge-small-zh-v1.5"。
        similarity_threshold (float): 分块时的相似度阈值，控制合并的严格程度。
        max_chars (int): 分块的最大字符数。
        min_chars (int): 分块的最小字符数。

    Returns:
        tuple: 包含两个元素的元组：
            - vector_store (Optional[Any]): 构建的向量存储对象，如果无文档则返回None。
            - es (Optional[Any]): Elasticsearch客户端对象，如果无文档则返回None。
    """
    # 调用分块函数，加载目录下的txt文件并进行智能分块
    docs = load_and_split_txt_dir(
        txt_dir=txt_dir,
        model_name=model_name,
        similarity_threshold=similarity_threshold,
        max_chars=max_chars,
        min_chars=min_chars,
    )

    # 如果没有分块结果，则提前返回
    if not docs:
        print("没有可入库的文档")
        return None, None

    # 使用分块后的文档构建向量存储和Elasticsearch索引
    vector_store, es = build_vector_store_from_documents(docs)
    print(f"入库完成，共 {len(docs)} 个 chunk")
    return vector_store, es


if __name__ == "__main__":
    # 当脚本作为主程序运行时，执行入库操作
    txt_dir = r"G:\Agent\customer-service-ai-agent\Smart_City"  # 指定txt文件所在的目录
    ingest_txt_directory(txt_dir)  # 调用入库函数
    

# python G:\Agent\customer-service-ai-agent\rag\ingest.py
# 运行这段代码删除ES旧索引（也可以用ES客户端）,知识库内容更新后，要运行这段代码
from elasticsearch import Elasticsearch
es = Elasticsearch("http://localhost:9200")
es.indices.delete(index="rag_docs")  # 删除包含旧文档的索引
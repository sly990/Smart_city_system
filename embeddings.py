from langchain_openai import OpenAIEmbeddings

# LangGraph 会自动 import 这个变量
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
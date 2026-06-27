"""两步 RAG：检索 + 基于 context 生成答案（供 eval 使用）"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

load_dotenv(Path(__file__).parent.parent / ".env")

from src.prompts import GENERATE_RAG_ANSWER_PROMPT
from src.Rag.indexer import VectorIndexer
from src.Rag.retriever import advanced_retrieve
from langchain_core.prompts import ChatPromptTemplate

K_DEFAULT = 3

def _get_llm():
    return ChatDeepSeek(model="deepseek-chat", temperature=0.1, api_key=os.getenv("DEEPSEEK_API_KEY", ""))

def retrieve_docs(question: str, mode: str = "baseline", k: int = K_DEFAULT):
    if mode == "advanced":
        bundle = advanced_retrieve(question, k=k)
        return bundle.chunks
    retriever = VectorIndexer(persist_directory="db").get_retriever(k=k)
    return retriever.invoke(question)

def retrieve_chunk_ids(question: str, mode: str = "baseline", k: int = K_DEFAULT) -> list[str]:
    return [d.metadata.get("chunk_id", "") for d in retrieve_docs(question, mode, k) if d.metadata.get("chunk_id")]

def run_rag(question: str, retriever_mode: str = "baseline", k: int = K_DEFAULT) -> dict:
    docs = retrieve_docs(question, retriever_mode, k)
    contexts = [d.page_content for d in docs]
    context_text = "\n\n".join(contexts)
    qa_prompt = ChatPromptTemplate.from_template(GENERATE_RAG_ANSWER_PROMPT)
    chain = (
        {"context": lambda _: context_text, "question": RunnablePassthrough()}
        | qa_prompt
        | _get_llm()
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return {"question": question, "contexts": contexts, "answer": answer, "chunk_ids": [d.metadata.get("chunk_id", "") for d in docs]}

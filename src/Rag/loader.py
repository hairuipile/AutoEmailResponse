"""文档加载和分割模块"""
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_documents():
    """加载所有知识文档"""
    docs_dir = PROJECT_ROOT / "context"
    documents = []
    extensions = [".txt", ".md"]

    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in extensions:
            print(f"加载文档: {file_path}")
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = str(file_path)
            documents.extend(docs)

    if not documents:
        print(f"警告: 未找到任何文档，{docs_dir} 目录为空或没有支持的文档格式")
        return []

    print(f"共加载 {len(documents)} 个文档")
    return documents


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """将文档分割成块"""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"分割为 {len(chunks)} 个文本块")
    return chunks

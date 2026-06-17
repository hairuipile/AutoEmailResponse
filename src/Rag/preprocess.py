#!/usr/bin/env python3
"""RAG 预处理入口脚本"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

# Load .env
project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

# Import from module
from src.Rag import (
    check_knowledge_changes,
    get_documents,
    split_documents,
    VectorIndexer,
    save_manifest,
)


def run(force_rebuild: bool = False) -> bool:
    """运行预处理流程"""
    print("=" * 50)
    print("RAG 知识库预处理")
    print("=" * 50)

    # 0. 检查变化
    print("\n[0/6] 检查知识库变化...")
    has_changes, new_manifest = check_knowledge_changes()

    indexer = VectorIndexer(persist_directory="db")

    # 0.1 同步被删除的源文章（清理孤立块）
    print("\n[0/6] 同步被删除的源文章...")
    removed = indexer.sync_deleted_articles()

    if not force_rebuild and not has_changes and removed == 0:
        print("[+] 知识库无变化，跳过预处理")
        return False

    if has_changes and not force_rebuild:
        print("[+] 检测到知识库有变化，开始重建向量库...")

    if removed > 0:
        print(f"[+] 已清理 {removed} 个孤立块")

    # 1. 加载文档
    print("\n[1/6] 加载知识文档...")
    documents = get_documents()

    # 2. 分割文档
    print("\n[2/6] 分割文档为文本块...")
    chunks = split_documents(documents)

    # 3. 构建向量索引（同时建立文章↔块关联）
    print("\n[3/6] 构建向量索引并建立文章↔块关联...")
    vectorstore = indexer.build_index(chunks)

    # 4. 保存清单
    print("\n[4/6] 保存源文件清单...")
    save_manifest(new_manifest)

    # 5. 输出统计
    print("\n[5/6] 索引统计...")
    stats = indexer.get_collection_stats()
    print(f"    文档块数量: {stats.get('count', 0)}")
    print(f"    存储路径: {stats.get('persist_directory', 'N/A')}")

    # 6. 最终校验：再次同步确保无遗漏
    print("\n[6/6] 最终一致性校验...")
    final_removed = indexer.sync_deleted_articles()
    if final_removed > 0:
        print(f"[!] 校验发现 {final_removed} 个额外孤立块，已清理")

    if vectorstore:
        print("\n" + "=" * 50)
        print("预处理完成！向量库已就绪。")
        print("=" * 50)
        return True
    else:
        print("\n预处理未完成，请检查文档目录和配置。")
        return False


def check_and_update() -> bool:
    """检查知识库变化并在需要时更新向量库。供 main.py 调用"""
    has_changes, _ = check_knowledge_changes()
    indexer = VectorIndexer(persist_directory="db")

    # 即使内容没变，也要先同步被删除的文件
    removed = indexer.sync_deleted_articles()

    if not has_changes and removed == 0:
        print("[+] 知识库无变化，跳过向量库更新")
        return False

    if removed > 0:
        print(f"[+] 已清理 {removed} 个孤立块")
        return run(force_rebuild=False)

    print("[+] 检测到知识库变化，开始更新向量库...")
    return run(force_rebuild=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 知识库预处理")
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重建向量库（忽略变化检测）"
    )
    args = parser.parse_args()
    run(force_rebuild=args.force)

"""清单数据库管理：检测知识库变化，建立文章↔块关联，支持源文章删除时清理孤立块"""
import hashlib
import sqlite3
from pathlib import Path

MANIFEST_DB = "db/knowledge_manifest.db"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_file_hash(file_path: Path) -> str:
    """计算文件的 MD5 哈希值"""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_db_connection():
    """获取数据库连接"""
    db_path = PROJECT_ROOT / MANIFEST_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_manifest_table():
    """初始化清单表（articles + article_chunks）"""
    conn = get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                modified_time REAL NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON articles(file_path)")

        # 文章↔块关联表：源文章被删除时，按 article_id 找到所有 chroma_id 批量删除
        conn.execute("""
            CREATE TABLE IF NOT EXISTS article_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                chroma_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                UNIQUE (article_id, chunk_index)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_article_id ON article_chunks(article_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chroma_id ON article_chunks(chroma_id)")

        conn.commit()
    finally:
        conn.close()


def load_manifest() -> dict:
    """从数据库加载源文件清单（按 file_path 索引）"""
    manifest = {}
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.execute(
            "SELECT id, file_path, file_hash, modified_time, chunk_count FROM articles"
        )
        for row in cursor.fetchall():
            article_id, file_path, file_hash, modified_time, chunk_count = row
            manifest[file_path] = {
                "id": article_id,
                "hash": file_hash,
                "modified": str(modified_time),
                "chunk_count": chunk_count,
            }
    except sqlite3.OperationalError:
        manifest = {}
    finally:
        if conn is not None:
            conn.close()
    return manifest


def save_manifest(manifest: dict) -> None:
    """保存源文件清单到数据库（替换式写入）"""
    conn = get_db_connection()
    try:
        init_manifest_table()
        conn.execute("DELETE FROM articles")
        for file_path, info in manifest.items():
            conn.execute(
                """INSERT INTO articles (file_path, file_hash, modified_time, chunk_count)
                   VALUES (?, ?, ?, ?)""",
                (
                    file_path,
                    info["hash"],
                    float(info["modified"]),
                    int(info.get("chunk_count", 0)),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def record_article_chunks(file_path: str, chroma_ids: list[str]) -> int:
    """记录一篇文章对应的所有 chroma chunk id

    Returns:
        article_id
    """
    conn = get_db_connection()
    try:
        init_manifest_table()

        # 1. 删除该文章已有的关联（重建场景：先清空旧关联）
        article_id = conn.execute(
            "SELECT id FROM articles WHERE file_path = ?", (file_path,)
        ).fetchone()
        if article_id:
            article_id = article_id[0]
            conn.execute("DELETE FROM article_chunks WHERE article_id = ?", (article_id,))
        else:
            cursor = conn.execute(
                "INSERT INTO articles (file_path, file_hash, modified_time, chunk_count) "
                "VALUES (?, '', 0, 0)",
                (file_path,),
            )
            article_id = cursor.lastrowid

        # 2. 批量插入 chunk 关联
        for idx, cid in enumerate(chroma_ids):
            conn.execute(
                "INSERT INTO article_chunks (article_id, chroma_id, chunk_index) VALUES (?, ?, ?)",
                (article_id, cid, idx),
            )

        # 3. 更新文章的 chunk_count
        conn.execute(
            "UPDATE articles SET chunk_count = ? WHERE id = ?",
            (len(chroma_ids), article_id),
        )

        conn.commit()
        return article_id
    finally:
        conn.close()


def get_deleted_files(current_files: set[str]) -> list[dict]:
    """对比当前磁盘文件与数据库，找出被删除的文件

    Returns:
        [{"file_path": ..., "chroma_ids": [...], "article_id": ...}, ...]
    """
    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT id, file_path FROM articles").fetchall()
        deleted = []
        for article_id, file_path in rows:
            if file_path not in current_files:
                chroma_rows = conn.execute(
                    "SELECT chroma_id FROM article_chunks WHERE article_id = ?",
                    (article_id,),
                ).fetchall()
                deleted.append({
                    "file_path": file_path,
                    "article_id": article_id,
                    "chroma_ids": [r[0] for r in chroma_rows],
                })
        return deleted
    finally:
        conn.close()


def check_knowledge_changes() -> tuple[bool, dict]:
    """检查知识库是否有变化"""
    init_manifest_table()

    docs_dir = PROJECT_ROOT / "context"
    extensions = [".txt", ".md"]

    new_manifest = {}

    for file_path in docs_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix in extensions:
            file_hash = get_file_hash(file_path)
            relative_path = str(file_path.relative_to(PROJECT_ROOT))
            new_manifest[relative_path] = {
                "hash": file_hash,
                "modified": str(file_path.stat().st_mtime),
                "chunk_count": 0,  # 初始为 0，重建向量后更新
            }

    old_manifest = load_manifest()

    if old_manifest != new_manifest:
        return True, new_manifest

    return False, new_manifest

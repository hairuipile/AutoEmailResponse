"""清单数据库管理：检测知识库变化，管理文章级账本（UUID、物理路径、Hash指纹）"""
import hashlib
import uuid as uuid_lib
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
    """初始化清单表（只记录文章级信息）"""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                uuid TEXT PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                file_hash TEXT NOT NULL,
                description TEXT,
                modified_time REAL NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON articles(file_path)")
        conn.commit()


def load_manifest() -> dict:
    """从数据库加载文章清单（按 file_path 索引）"""
    manifest = {}
    try:
        with get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT uuid, file_path, file_hash, modified_time, description FROM articles"
            )
            for article_uuid, file_path, file_hash, modified_time, description in cursor:
                manifest[file_path] = {
                    "uuid": article_uuid,
                    "hash": file_hash,
                    "modified": str(modified_time),
                    "description": description or "",
                }
    except sqlite3.OperationalError:
        manifest = {}
    return manifest


def save_manifest(manifest: dict) -> None:
    """保存文章清单到数据库"""
    with get_db_connection() as conn:
        init_manifest_table()
        conn.execute("DELETE FROM articles")
        for file_path, info in manifest.items():
            conn.execute(
                """INSERT INTO articles (uuid, file_path, file_hash, description, modified_time)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    info["uuid"],
                    file_path,
                    info["hash"],
                    info.get("description", ""),
                    float(info["modified"]),
                ),
            )
        conn.commit()


def check_knowledge_changes() -> tuple[bool, dict]:
    """检查知识库是否有变化，返回 (是否有变化, 新清单)"""
    init_manifest_table()

    docs_dir = PROJECT_ROOT / "context"
    extensions = [".txt", ".md", ".json"]

    old_manifest = load_manifest()

    new_manifest = {}
    for file_path in docs_dir.rglob("*"):
        if not file_path.is_file():
            continue
        # 跳过清洗输出目录，避免循环扫描
        rel = file_path.relative_to(docs_dir)
        if rel.parts[0] == "clean_context":
            continue
        if file_path.suffix not in extensions:
            continue

        file_hash = get_file_hash(file_path)
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        file_uuid = old_manifest.get(relative_path, {}).get("uuid") or str(uuid_lib.uuid4())
        description = old_manifest.get(relative_path, {}).get("description") or file_path.stem
        new_manifest[relative_path] = {
            "uuid": file_uuid,
            "hash": file_hash,
            "modified": str(file_path.stat().st_mtime),
            "description": description,
        }

    if old_manifest != new_manifest:
        return True, new_manifest

    return False, new_manifest


def get_deleted_articles() -> list[dict]:
    """对比当前磁盘文件与数据库，返回被删除的文章列表

    Returns:
        [{"uuid": ..., "file_path": ...}, ...]
    """
    with get_db_connection() as conn:
        rows = conn.execute("SELECT uuid, file_path FROM articles").fetchall()

    docs_dir = PROJECT_ROOT / "context"
    current_files = set(str(fp.relative_to(PROJECT_ROOT)) for fp in docs_dir.rglob("*") if fp.is_file())

    deleted = []
    for article_uuid, file_path in rows:
        if file_path not in current_files:
            deleted.append({
                "uuid": article_uuid,
                "file_path": file_path,
            })
    return deleted

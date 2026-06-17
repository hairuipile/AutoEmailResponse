"""数据清洗模块：把大文件按 H1 拆分为多个小文件

设计原则：
- 不破坏 loader.py / manifest.py / indexer.py 的现有契约
- 拆分后落盘到 context/clean_context/，loader 改为读这个目录
- 小文件（< 小文件阈值）原样复制，不做无谓拆分
- 拆分后每个文件都保留 H1 标题作为正文开头（与原 loader 行为一致）

拆分粒度：H1
- 一个文件如果有 N 个 H1 → 拆为 N 个小文件
- 没有 H1 → 整个文件视为一个 H1 处理
- 文件总字数 < MIN_SPLIT_LENGTH → 不拆，原样复制
"""
import re
import shutil
from pathlib import Path
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter

# 拆分阈值：文件总字数小于该值时不做拆分
MIN_SPLIT_LENGTH = 4000

# 标题安全化：去掉文件名不允许的字符
_SAFE_INVALID = re.compile(r"[\\/:*?\"<>|]+")
_SAFE_SPACE = re.compile(r"\s+")


def _safe_filename(title: str) -> str:
    """把 H1 标题转成安全的文件名片段"""
    safe = _SAFE_INVALID.sub("_", title)
    safe = _SAFE_SPACE.sub("_", safe)
    return safe.strip("_")[:60] or "untitled"


def _split_by_h1(content: str) -> List[tuple[str, str]]:
    """按 H1 切分 Markdown，返回 [(h1_title, h1_block), ...]

    - 第一个块可能没有 H1（开头的引言段），会归入下一个 H1
    - 整段只有一个 H1 → 返回 [(title, full_content)]
    - 完全没 H1 → 返回 [("", full_content)]
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header_1")],
        strip_headers=False,
    )
    chunks = splitter.split_text(content)
    return [(c.metadata.get("Header_1", ""), c.page_content) for c in chunks]


def clean_knowledge(
    source_dir: str | Path = "context",
    target_dir: str | Path = "context/clean_context",
    min_split_length: int = MIN_SPLIT_LENGTH,
) -> List[Path]:
    """把 source_dir 下所有 .md / .txt 文件清洗后写入 target_dir

    处理规则：
    1. 每个文件读出来后按 H1 切分
    2. 文件总字数 < min_split_length → 原样复制（不拆）
    3. 切分后有 N 个 H1 块 → 写入 N 个文件：{stem}__{h1_safe}.md
    4. 切分后只有 1 个块（无 H1 或 H1=1）→ 原样复制
    5. 保留 Markdown front-matter，注入 original_source 注释

    Args:
        source_dir: 原始知识库目录
        target_dir: 清洗后输出目录
        min_split_length: 拆分阈值

    Returns:
        写入的所有文件路径列表
    """
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    extensions = [".md", ".txt", ".json"]
    # 跳过子目录（如之前清洗残留的 clean_context/），避免循环扫描
    skip_dirs = {"clean_context"}

    for src_file in sorted(source.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(source)
        if rel.parts[0] in skip_dirs:
            continue
        if src_file.suffix not in extensions:
            continue

        content = src_file.read_text(encoding="utf-8")
        total_len = len(content)

        # 规则 1 + 规则 4：小文件 / 单 H1 块 → 原样复制
        if total_len < min_split_length:
            dst = target / src_file.name
            _write_with_header(dst, content, src_file)
            written.append(dst)
            print(f"  [复制] {src_file.name} ({total_len}字, 太小不拆)")
            continue

        # 规则 2：按 H1 切分
        h1_blocks = _split_by_h1(content)
        if len(h1_blocks) <= 1:
            dst = target / src_file.name
            _write_with_header(dst, content, src_file)
            written.append(dst)
            print(f"  [复制] {src_file.name} ({total_len}字, 仅 1 个 H1)")
            continue

        # 规则 3：拆分为多个文件
        stem = src_file.stem
        suffix = src_file.suffix
        for idx, (h1_title, h1_block) in enumerate(h1_blocks, start=1):
            if not h1_title:
                # 没有 H1 标题的前置块（引言），归入第一个有 H1 的文件
                # 这里为了简化直接单独写一份
                fname = f"{stem}__intro{suffix}"
            else:
                fname = f"{stem}__{_safe_filename(h1_title)}{suffix}"
            dst = target / fname
            _write_with_header(dst, h1_block, src_file, h1_title)
            written.append(dst)
            print(f"  [拆分] {src_file.name} → {fname} ({len(h1_block)}字)")

    print(f"\n✅ 清洗完成: {len(written)} 个文件已写入 {target.resolve()}")
    return written


def _write_with_header(dst: Path, content: str, src_file: Path, h1_title: str = "") -> None:
    """写入文件，注入溯源 front-matter（Markdown 注释形式，不影响解析）"""
    header_lines = [
        "<!--",
        f"  original_source: {src_file.name}",
    ]
    if h1_title:
        header_lines.append(f"  h1_title: {h1_title}")
    header_lines.append("-->")
    header_lines.append("")

    full_content = "\n".join(header_lines) + content
    dst.write_text(full_content, encoding="utf-8")


def clean_knowledge_idempotent(
    source_dir: str | Path = "context",
    target_dir: str | Path = "context/clean_context",
    min_split_length: int = MIN_SPLIT_LENGTH,
) -> List[Path]:
    """幂等版本：先清空 target_dir 再清洗

    适用于开发/调试阶段。生产环境建议改成增量。
    """
    target = Path(target_dir)
    if target.exists():
        shutil.rmtree(target)
    return clean_knowledge(source_dir, target_dir, min_split_length)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--incremental":
        clean_knowledge()
    else:
        clean_knowledge_idempotent()

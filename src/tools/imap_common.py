def append_to_drafts(conn, raw: bytes, folders: list[str]) -> str:
    for folder in folders:
        try:
            if conn.append(folder, '\\Draft', None, raw)[0] == 'OK':
                return folder
        except Exception:
            continue
    raise RuntimeError(f"无法写入草稿箱，已尝试: {folders}")

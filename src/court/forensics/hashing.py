import hashlib
from pathlib import Path


def path_sha256(path: Path) -> str:
    """Content hash of a file or directory tree.

    Directories hash every file in sorted order, mixing in each file's
    POSIX-relative path so renames change the digest. The training side records
    this at save time and the runtime registry recomputes it at load time to
    verify a model has not changed; both sides must stay byte-for-byte identical,
    which is why this lives in one place.
    """
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()

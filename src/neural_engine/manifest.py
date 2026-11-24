"""Integrity of the trusted checkout, not an authenticity signature."""
import hashlib
import re
from pathlib import Path, PurePosixPath

def read_pins(root):
    root=Path(root).resolve()
    result={}
    for line in (root/"SHA256SUMS").read_text("utf-8").splitlines():
        m=re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)",line)
        if m is None:raise ValueError("malformed digest entry")
        rel=PurePosixPath(m[2]); p=root/rel
        if rel.is_absolute() or ".." in rel.parts or p.resolve()!=p or p in result or rel.as_posix()!=m[2]:
            raise ValueError("unsafe or duplicate release path")
        result[p]=m[1]
    if not result:raise ValueError("empty release manifest")
    return result

def verify(pins):
    for p,d in pins.items():
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=d:
            raise ValueError("changed or missing: "+p.name)

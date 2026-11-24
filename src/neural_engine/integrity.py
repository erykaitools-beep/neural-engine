import hashlib, json

def need(ok, reason):
    if not ok: raise ValueError(reason)

def digest(b): return hashlib.sha256(b).hexdigest()

def sealed_bytes(path, expected):
    data=path.read_bytes()
    need(not path.is_symlink() and digest(data)==expected, 'integrity: '+path.name)
    return data

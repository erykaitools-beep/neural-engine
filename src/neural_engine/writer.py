"""Isolated synthetic candidate. State is encoded in 3/5 persistent bytes.

Bank.get deliberately does not read pending evidence. Writing always requires
an explicit arm; a neutral checkpoint alone cannot identify the arm.
"""
import struct

MASKS = tuple(n for n in range(1, 256) if n.bit_count() == 2)
CODES = tuple(n for n in range(1, 256) if n.bit_count() == 3)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def key(k):
    return type(k) is int and 0 < k < 256 and k.bit_count() == 2


def pack(a, x, p=0, b=0, y=0):
    return struct.pack("=BbB", a, x, p) + (struct.pack("=Bb", b, y) if b else b"")


def pending_byte(k, f, diverse):
    return CODES[2 * MASKS.index(k) + int(f > 0)] if diverse else k


def pending_info(p):
    if not p:
        return 0, 0
    if key(p):
        return p, 0
    need(p in CODES, "pending code")
    i = CODES.index(p)
    return MASKS[i // 2], (1 if i % 2 else -1)


class Bank:
    __slots__ = ("blob",)

    def __init__(self, blob, *, diverse):
        need(type(diverse) is bool, "arm")
        need(type(blob) is bytes and len(blob) in (3, 5), "bank length")
        a, x, p = struct.unpack("=BbB", blob[:3])
        need((key(a) or a == x == 0) and abs(x) <= 31, "primary")
        pk, pf = pending_info(p)
        need(not pk or (a and pk != a), "pending marker")
        if len(blob) == 5:
            b, y = struct.unpack("=Bb", blob[3:])
            need(a and key(b) and b != a and not p and abs(y) <= 31, "allocated")
        need(not pk or bool(pf) == diverse, "pending policy")
        self.blob = blob

    def __contains__(self, k):
        return key(k)

    def get(self, k, default=0):
        if not key(k):
            return default
        off = 4 if len(self.blob) == 5 and self.blob[3] == k else 1
        return struct.unpack_from("=b", self.blob, off)[0]


def deliver(blob, ticket, target, *, diverse, defer):
    need(type(diverse) is bool and type(defer) is bool, "arm")
    view = Bank(blob, diverse=diverse)
    a, x, p = struct.unpack("=BbB", blob[:3])
    pk, pf = pending_info(p)
    need(type(ticket) is int and 0 <= ticket < 1 << 24 and
         (target is None or type(target) is int and target in (-1, 1)), "format")
    if not ticket or target is None:
        return blob
    k, cue, v = ticket >> 16, (ticket >> 8) & 255, ticket & 255
    need(k in view and cue.bit_count() == 3 and v in (0, 2, 8, 10), "invalid physical token")
    first, last = (v >> 2) - 1, (v & 3) - 1
    delta = target * ((first > last) - (first < last))
    if not delta:
        return blob
    clip = lambda z: max(-31, min(31, z))
    if len(blob) == 5:
        b, y = struct.unpack("=Bb", blob[3:])
        return pack(a, x, 0, b, clip(y + delta)) if k == b else pack(a, clip(x + delta), 0, b, y)
    if not a:
        return pack(k, delta)
    if k != a and x * delta < 0:
        if pk == k and (not diverse or first != pf):
            return pack(a, x, 0, k, delta)
        return pack(a, x if defer else clip(x + delta), pending_byte(k, first, diverse))
    return pack(a, clip(x + delta))

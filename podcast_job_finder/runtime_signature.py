from __future__ import annotations

import hashlib
import json


def build_runtime_signature_hash(payload: object) -> str:
    serialized_payload = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized_payload.encode("utf-8")).hexdigest()

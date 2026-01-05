from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict


@dataclass(frozen=True)
class ParamSpace:
    """
    Minimal parameter container for MC experiments.

    Keep it stable & serializable. Avoid lambdas / dynamic code.
    """
    params: Dict[str, Any]

    def fingerprint(self) -> str:
        payload = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

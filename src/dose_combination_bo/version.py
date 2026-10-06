"""Version and source fingerprint for maintained-release checkpoint identity."""
from functools import lru_cache
from pathlib import Path
import hashlib
VERSION = "0.2.1rc5"

@lru_cache(maxsize=1)
def implementation_fingerprint():
    """Hash installed core Python sources; does not authenticate external plug-ins."""
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(b"\0"); h.update(p.read_bytes())
    return h.hexdigest()

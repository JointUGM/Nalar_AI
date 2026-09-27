from pathlib import Path

import yaml

from nalar_ai.shared.enums import ChunkKind
from nalar_ai.subsystems.s1_knowledge_base.domain.kinds import KindLexicon, KindMarker, RegionScope

DEFAULT_LEXICON_PATH = Path(__file__).resolve().parents[1] / "config" / "chunk_kinds.yaml"


def load_kind_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> KindLexicon:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    markers = [
        KindMarker(ChunkKind(kind), RegionScope(spec["scope"]), str(marker))
        for kind, spec in raw["kinds"].items()
        for marker in spec["markers"]
    ]
    return KindLexicon.build(int(raw["version"]), markers)

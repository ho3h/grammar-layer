"""Centralised configuration constants for Neograph."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class ModelConfig:
    """Gemma 2 2B + Gemma Scope SAE selection."""

    name: str = "gemma-2-2b"
    family: str = "gemma-2"
    d_model: int = 2304
    n_layers: int = 26
    vocab_size: int = 256000
    hf_repo: str = "google/gemma-2-2b"


@dataclass(frozen=True)
class SAEConfig:
    """Selected Gemma Scope release/layer/width."""

    release: str = "gemma-scope-2b-pt-res-canonical"
    sae_id: str = "layer_20/width_16k/canonical"
    layer: int = 20
    site: str = "resid_post"
    hook_name: str = "blocks.20.hook_resid_post"
    d_in: int = 2304
    d_sae: int = 16384
    architecture: str = "jumprelu"
    activation_threshold: float = 1e-3

    @property
    def neograph_id(self) -> str:
        return f"{self.release}/L{self.layer}/16k/canonical"


@dataclass(frozen=True)
class Neo4jConfig:
    """Neo4j connection details (loaded from .env)."""

    uri: str = os.environ.get("NEO4J_URI", "bolt://localhost:7693")
    user: str = os.environ.get("NEO4J_USER", "neo4j")
    password: str = os.environ.get("NEO4J_PASSWORD", "neograph_local_dev")
    database: str = os.environ.get("NEO4J_DATABASE", "neo4j")


@dataclass(frozen=True)
class Paths:
    """Filesystem layout."""

    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    staging: Path = PROJECT_ROOT / "data" / "staging"
    synthetic: Path = PROJECT_ROOT / "data" / "synthetic"
    labels_cache: Path = PROJECT_ROOT / "data" / "labels_cache.json"
    reports: Path = PROJECT_ROOT / "reports"
    cypher_dir: Path = PROJECT_ROOT / "cypher"
    bloom_dir: Path = PROJECT_ROOT / "bloom"

    def ensure(self) -> None:
        for p in (self.data, self.staging, self.synthetic, self.reports):
            p.mkdir(parents=True, exist_ok=True)


MODEL = ModelConfig()
SAE = SAEConfig()
NEO4J = Neo4jConfig()
PATHS = Paths()
PATHS.ensure()


# === Embeddings ===
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# === Corpus sizing ===
PILE_PROMPTS = 10_000
PILE_TOKENS_PER_PROMPT = 128
RHYME_PROMPTS = 1_000
WEEKDAY_PROMPTS = 500
TOP_K_TOKENS = 32

# === Relation building ===
KNN_K = 32
COACTIVATION_PMI_MIN = 1.0
COACTIVATION_NMIN = 5

# === Manifold fitting ===
PCA_VARIANCE_TARGET = 0.95
PCA_DIM_FLOOR = 8
PCA_DIM_CEIL = 32
N_WAYPOINTS = 16
COMMUNITY_SIZE_MIN = 8

# === Neuronpedia ===
NEURONPEDIA_BASE = "https://www.neuronpedia.org/api"
NEURONPEDIA_MODEL_SLUG = "gemma-2-2b"
NEURONPEDIA_SAE_SLUG = "20-gemmascope-res-16k"


def neuronpedia_feature_url(idx: int) -> str:
    return f"{NEURONPEDIA_BASE}/feature/{NEURONPEDIA_MODEL_SLUG}/{NEURONPEDIA_SAE_SLUG}/{idx}"

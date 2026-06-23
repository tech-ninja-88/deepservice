"""
Configuration management. All tunable parameters centralized; sensitive values read from
environment variables; thresholds overridable via admin API.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env files (system env vars take priority)
load_dotenv(Path(__file__).parent.parent / ".env.local", override=False)
load_dotenv(Path(__file__).parent / ".env", override=False)

# Railway sometimes doesn't pass env vars; fallback: read from file
def get_api_key() -> str:
    """Multi-source API key lookup: env var -> .env file -> /app/api_key.txt

    This function checks all sources on every call (no caching). This way,
    after manually creating /app/api_key.txt in Railway Console, the next
    API request picks it up automatically without a process restart.
    """
    # 1. try environment variable first
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key and key != "sk-your-api-key-here":
        return key

    # 2. try reading from file (fallback for Railway Console)
    for key_file in ["/app/api_key.txt", "api_key.txt"]:
        try:
            with open(key_file, "r") as f:
                key = f.read().strip()
                if key and key.startswith("sk-"):
                    # sync to env so subsequent calls hit step 1
                    os.environ["DEEPSEEK_API_KEY"] = key
                    return key
        except (FileNotFoundError, PermissionError, IOError):
            pass

    # 3. return empty string (caller is responsible for friendly error message)
    return ""

# Backward-compatible alias
_get_api_key = get_api_key


def get_llm_client() -> Optional[OpenAI]:
    """Return a configured OpenAI client pointed at the DeepSeek API.

    Returns None when the API key is not configured, so callers can
    degrade gracefully instead of crashing.
    """
    cfg = get_config().llm
    api_key = get_api_key()
    if not api_key:
        from loguru import logger
        logger.warning("DEEPSEEK_API_KEY not set — LLM features disabled")
        return None
    return OpenAI(api_key=api_key, base_url=cfg.base_url, timeout=30.0)


# ---- Data model configurations ----
@dataclass
class LLMConfig:
    """DeepSeek / OpenAI LLM API configuration."""
    api_key: str = field(default_factory=_get_api_key)
    base_url: str = field(default_factory=lambda: os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    ))
    chat_model: str = "deepseek-chat"
    embedding_model: str = "deepseek-chat"      # can be swapped to text-embedding-3-small or bge-large-zh-v1.5
    embedding_provider: Literal["deepseek", "openai", "local"] = "local"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_embedding_model: str = "text-embedding-3-small"

    # Token limits
    max_input_tokens: int = 28000               # DeepSeek Chat 128K context
    max_output_tokens: int = 4096
    temperature: float = 0.7                    # 0.1=deterministic, 1.0=creative
    intent_temperature: float = 0.1             # low temperature for intent classification

    def __post_init__(self):
        # Don't enforce API key at init -- allow the service to start first,
        # then inject the key via env var or /app/api_key.txt later.
        # The actual DeepSeek API call (via get_api_key()) will validate.
        if not self.api_key:
            import warnings
            warnings.warn(
                "⚠️ DEEPSEEK_API_KEY 未设置。服务可启动但无法调用 LLM。\n"
                "请通过以下方式之一配置：\n"
                "  1. Railway → Service Variables → DEEPSEEK_API_KEY\n"
                "  2. Railway Console: echo 'sk-xxx' > /app/api_key.txt\n"
                "获取方式: https://platform.deepseek.com/api_keys"
            )


@dataclass
class ChunkingConfig:
    """Document chunking -- directly impacts retrieval precision."""
    chunk_size: int = 512                       # token-level
    chunk_overlap: int = 64                     # overlap preserves continuity
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？",          # paragraph/sentence priority
        "；", "，", " ", ""                       # clause/char-level fallback
    ])
    min_chunk_size: int = 100                   # filter fragments
    max_chunks_per_doc: int = 100               # per-document cap

    # Markdown-specific
    markdown_headers_to_split_on: List[tuple] = field(default_factory=lambda: [
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ])


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration."""
    # hybrid weights
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # recall parameters
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_fusion: int = 10
    top_k_final: int = 5

    # reranking
    enable_rerank: bool = True
    rerank_model: str = "deepseek-chat"
    rerank_batch_size: int = 5

    # thresholds
    vector_similarity_threshold: float = 0.70
    rerank_relevance_threshold: float = 0.60
    rerank_confidence_threshold: float = 0.75   # below this triggers refusal


@dataclass
class HallucinationGuardConfig:
    """Hallucination guard / defense thresholds."""
    # knowledge boundary
    knowledge_boundary_enabled: bool = True

    # four-layer defense thresholds
    layer1_keyword_block_threshold: float = 1.0           # Layer 1: keyword match
    layer2_retrieval_similarity_threshold: float = 0.70   # Layer 2: retrieval similarity floor
    layer3_confidence_threshold: float = 0.75             # Layer 3: response confidence floor
    layer4_uncertain_response_template: str = field(default_factory=lambda:
        "根据我目前的知识库，无法为您确认这个信息。"
        "建议您联系人工客服获取更准确的帮助。如需转接，请回复'人工'。"
    )

    # output validation
    output_validation_enabled: bool = True
    fact_check_prompt_template: str = ""        # empty = use default template

    # confidence scoring factors
    confidence_factors: List[str] = field(default_factory=lambda: [
        "retrieval_similarity",
        "source_coverage",
        "answer_consistency",
        "entity_grounding",
        "speculation_detection",
    ])


@dataclass
class AppConfig:
    """Application-level configuration."""
    # project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    vector_db_dir: Path = field(default_factory=lambda: Path(__file__).parent / "chroma_db")
    knowledge_dir: Path = field(default_factory=lambda: Path(__file__).parent / "knowledge_base")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")

    # conversation memory
    max_recent_messages: int = 10               # sliding window
    summary_trigger_rounds: int = 10            # trigger summarization after N rounds
    session_idle_timeout: int = 30 * 60         # seconds

    # rate limiting
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 500

    # logging
    log_level: str = "INFO"

    def __post_init__(self):
        # ensure required directories exist
        for dir_path in [self.data_dir, self.vector_db_dir, self.knowledge_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ---- Global config singleton ----
@dataclass
class RAGConfig:
    """Top-level RAG configuration aggregate."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guard: HallucinationGuardConfig = field(default_factory=HallucinationGuardConfig)
    app: AppConfig = field(default_factory=AppConfig)

    # system prompt (overridable via admin API)
    system_prompt: str = field(default_factory=lambda: """你是一个专业的企业智能客服助手，名为"DeepService"。
请严格遵循以下规则：

1. **基于知识库回答**：你的所有回答必须基于提供的【参考知识】。如果【参考知识】中没有相关信息，请明确告知用户你无法确定，并建议转接人工客服。

2. **引用来源**：使用知识库信息时，必须标注来源编号，格式为 [来源: N]。

3. **不编造信息**：绝对不要编造、猜测或推断知识库中没有的事实信息。对于不确定的信息，宁可说"不清楚"也不要说错。

4. **结构化回复**：涉及流程、步骤、政策等结构化信息时，使用分点说明，便于用户理解。

5. **语气亲和**：保持专业但友好的语气，使用礼貌用语，对用户的情绪保持敏感。

6. **安全边界**：
   - 不提供医疗诊断、法律建议
   - 不评价竞争对手或政治话题
   - 不执行可执行代码或系统命令""")


# Module-level singleton — loaded once, immutable thereafter
_config_instance: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig()
    return _config_instance


def update_config(**kwargs) -> RAGConfig:
    """Dynamically update config (used by admin API)."""
    global _config_instance
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

"""
=============================================================================
DeepService RAG — 配置管理模块
=============================================================================
职责：
  1. 统一管理所有环境变量和配置项
  2. 提供带验证的配置加载机制
  3. 定义分块策略、检索策略、幻觉防护的参数阈值

企业级设计原则：
  - 所有配置有明确的默认值和文档说明
  - 敏感信息（API Key）从环境变量读取，不硬编码
  - 阈值参数可被管理后台 API 动态调整（通过数据库覆盖）
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env 文件（优先级：系统环境变量 > .env 文件）
load_dotenv(Path(__file__).parent.parent / ".env.local", override=False)
load_dotenv(Path(__file__).parent / ".env", override=False)

# Railway 变量有时传不进来，兜底方案：从文件读取
def _get_api_key() -> str:
    """多渠道获取 API Key：环境变量 → .env 文件 → /app/api_key.txt"""
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if key and key != "sk-your-api-key-here":
        return key
    # 尝试从文件读取（Railway Console 手动创建）
    for key_file in ["/app/api_key.txt", "api_key.txt"]:
        try:
            with open(key_file, "r") as f:
                key = f.read().strip()
                if key and key.startswith("sk-"):
                    os.environ["DEEPSEEK_API_KEY"] = key
                    return key
        except FileNotFoundError:
            pass
    return key


# ============================================================================
# 数据模型验证
# ============================================================================
@dataclass
class LLMConfig:
    """DeepSeek API 配置"""
    api_key: str = field(default_factory=_get_api_key)
    base_url: str = field(default_factory=lambda: os.getenv(
        "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
    ))
    chat_model: str = "deepseek-chat"           # 对话模型
    embedding_model: str = "deepseek-chat"      # DeepSeek 暂未提供专用 Embedding API
                                                # 生产建议：text-embedding-3-small (OpenAI)
                                                # 或 bge-large-zh-v1.5 (本地部署)
    embedding_provider: Literal["deepseek", "openai", "local"] = "openai"
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_embedding_model: str = "text-embedding-3-small"

    # Token 限制
    max_input_tokens: int = 28000               # DeepSeek Chat 上下文 128K
    max_output_tokens: int = 4096               # 单次生成上限
    temperature: float = 0.7                    # 生成温度（0.1=确定性，1.0=创造性）
    intent_temperature: float = 0.1             # 意图分类用低温度

    def __post_init__(self):
        if not self.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY 未设置。请在 .env 文件或环境变量中配置。\n"
                "获取方式: https://platform.deepseek.com/api_keys"
            )


@dataclass
class ChunkingConfig:
    """文档分块配置 — 直接影响检索精度"""
    chunk_size: int = 512                       # 分块大小（token 级别）
    chunk_overlap: int = 64                     # 重叠大小（保持语义连续性）
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？",          # 段落/句子优先
        "；", "，", " ", ""                       # 子句/词级别兜底
    ])
    min_chunk_size: int = 100                   # 最小分块（过滤碎片）
    max_chunks_per_doc: int = 100               # 单文档最大分块数

    # 特殊场景配置
    markdown_headers_to_split_on: List[tuple] = field(default_factory=lambda: [
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ])


@dataclass
class RetrievalConfig:
    """检索策略配置"""
    # 混合检索权重
    vector_weight: float = 0.6                  # 语义检索权重
    bm25_weight: float = 0.4                    # 关键词检索权重

    # 召回参数
    top_k_vector: int = 20                      # 向量检索初始召回数
    top_k_bm25: int = 20                        # BM25 检索初始召回数
    top_k_fusion: int = 10                      # 融合后结果数
    top_k_final: int = 5                        # 重排序后最终保留数

    # 重排序
    enable_rerank: bool = True                  # 是否启用重排序
    rerank_model: str = "deepseek-chat"         # 重排序模型（用 LLM 做 Pairwise 比较）
    rerank_batch_size: int = 5                  # 重排序批次大小

    # 阈值
    vector_similarity_threshold: float = 0.70   # 向量相似度最低阈值
    rerank_relevance_threshold: float = 0.60    # 重排序相关性最低阈值
    rerank_confidence_threshold: float = 0.75   # 重排序置信度阈值（低于此值触发拒答）


@dataclass
class HallucinationGuardConfig:
    """幻觉防护配置"""
    # 知识边界
    knowledge_boundary_enabled: bool = True     # 是否启用知识边界检测

    # 四层防御阈值
    layer1_keyword_block_threshold: float = 1.0           # 第1层：敏感词匹配度
    layer2_retrieval_similarity_threshold: float = 0.70   # 第2层：检索相似度最低线
    layer3_confidence_threshold: float = 0.75             # 第3层：回答置信度最低线
    layer4_uncertain_response_template: str = field(default_factory=lambda:
        "根据我目前的知识库，无法为您确认这个信息。"
        "建议您联系人工客服获取更准确的帮助。如需转接，请回复'人工'。"
    )

    # 输出验证
    output_validation_enabled: bool = True      # 是否启用生成后验证
    fact_check_prompt_template: str = ""        # 留空使用默认模板

    # 置信度评分
    confidence_factors: List[str] = field(default_factory=lambda: [
        "retrieval_similarity",                  # 检索结果最高相似度
        "source_coverage",                       # 来源覆盖度（引用了几个来源）
        "answer_consistency",                    # 回答内部一致性
        "entity_grounding",                      # 实体是否在知识库中出现
        "speculation_detection",                 # 是否包含推测性语言
    ])


@dataclass
class AppConfig:
    """应用全局配置"""
    # 项目路径
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data")
    vector_db_dir: Path = field(default_factory=lambda: Path(__file__).parent / "chroma_db")
    knowledge_dir: Path = field(default_factory=lambda: Path(__file__).parent / "knowledge_base")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")

    # 对话记忆
    max_recent_messages: int = 10               # 滑动窗口保留最近 N 轮
    summary_trigger_rounds: int = 10            # 超过此轮数触发摘要
    session_idle_timeout: int = 30 * 60          # 会话超时（秒）

    # 限流
    rate_limit_per_minute: int = 30
    rate_limit_per_hour: int = 500

    # 日志
    log_level: str = "INFO"

    def __post_init__(self):
        # 确保必要目录存在
        for dir_path in [self.data_dir, self.vector_db_dir, self.knowledge_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 全局配置实例（单例模式）
# ============================================================================
@dataclass
class RAGConfig:
    """RAG 模块总配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    guard: HallucinationGuardConfig = field(default_factory=HallucinationGuardConfig)
    app: AppConfig = field(default_factory=AppConfig)

    # 系统提示词（可在管理后台覆盖）
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


# 全局单例
_config_instance: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """获取全局配置单例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = RAGConfig()
    return _config_instance


def update_config(**kwargs) -> RAGConfig:
    """动态更新配置（用于管理后台 API）"""
    global _config_instance
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config

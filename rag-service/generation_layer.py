"""
=============================================================================
DeepService RAG — RAG 生成模块 (Generation Layer)
=============================================================================
职责：
  1. Prompt 模板设计（含严格的输出约束）
  2. 源引用标注（回答可追溯到具体知识来源）
  3. 拒答规则（检索相关度低时触发拒答）
  4. 流式输出（SSE 打字机效果）

企业级设计原则：
  - Prompt 是 LLM 应用的"源代码"，需版本化管理
  - 输出约束防止模型自由发挥导致幻觉
  - 引用来源是建立用户信任的关键
  - 流式输出提升感知性能

参考：
  [reference:2] — 解决幻觉需要知识边界控制和输出约束等机制
=============================================================================
"""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, AsyncGenerator, Generator, Tuple
from enum import Enum

from openai import OpenAI
from loguru import logger

from config import get_config
from retrieval_layer import SearchResult, RetrievalResult


# ============================================================================
# 数据结构
# ============================================================================
class ResponseType(str, Enum):
    """回答类型"""
    KNOWLEDGE_BASED = "knowledge_based"       # 基于知识库回答
    UNCERTAIN = "uncertain"                    # 无法确定
    CLARIFICATION = "clarification"            # 需要澄清
    HANDOFF = "handoff"                        # 建议转人工
    GREETING = "greeting"                      # 寒暄
    CHITCHAT = "chitchat"                      # 闲聊


@dataclass
class TokenStream:
    """流式 Token"""
    content: str
    finish_reason: Optional[str] = None


@dataclass
class Citation:
    """来源引用"""
    index: int                                # 引用编号 [来源: 1]
    document_title: str                       # 文档标题
    content_snippet: str                      # 引用内容摘要
    relevance_score: float                    # 相关性得分


@dataclass
class RAGResponse:
    """RAG 生成的完整响应"""
    content: str                              # 回答正文
    response_type: ResponseType               # 回答类型
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0                   # 置信度 (0-1)
    metadata: Dict = field(default_factory=dict)
    usage: Dict = field(default_factory=dict)  # token 使用量

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "response_type": self.response_type.value,
            "citations": [asdict(c) for c in self.citations],
            "confidence": round(self.confidence, 4),
            "metadata": self.metadata,
            "usage": self.usage,
        }

    def to_markdown_with_citations(self) -> str:
        """生成带来源标注的 Markdown 回答（展示用）"""
        md = self.content
        if self.citations:
            md += "\n\n---\n**参考来源：**\n"
            for c in self.citations:
                md += (
                    f"- [来源: {c.index}] **{c.document_title}** "
                    f"(相关度: {c.relevance_score:.2f})\n"
                    f"  > {c.content_snippet[:100]}...\n"
                )
        return md


# ============================================================================
# Prompt 模板管理
# ============================================================================
class PromptTemplates:
    """
    Prompt 模板库 — LLM 应用的"源代码"

    版本化管理，可在管理后台动态切换。
    每个模板包含：系统提示词、用户消息模板、输出约束。

    设计要点：
      - 角色设定 + 行为约束 + 输出格式 = 三层结构
      - 使用少数示例（Few-shot）锚定输出格式
      - 输出约束用 <指令> 标签明确标注
    """

    # === System Prompt（系统提示词）===
    SYSTEM_PROMPT_RAG = """你是一个专业的企业智能客服助手，名为"DeepService"。

<核心规则>
1. **知识库优先**：你的所有回答必须严格基于【参考知识】中提供的信息。
2. **拒答规则**：如果【参考知识】中不包含回答所需的直接信息，你必须明确表示"无法确定"并建议联系人工客服。**绝对不要编造或推测**。
3. **来源标注**：使用参考知识时，在回答末尾标注来源编号（如 [来源: 1]）。
4. **语气要求**：专业、亲和、简洁。使用礼貌用语，对用户情绪保持敏感。
</核心规则>

<能力边界>
你能做的：
- 基于知识库回答用户关于产品、政策、流程的问题
- 解释和总结知识库中的信息
- 在信息不完整时请求用户补充说明

你不能做的：
- 编造知识库中没有的事实信息
- 提供医疗诊断、法律建议
- 评价竞争对手
- 执行代码或系统命令
</能力边界>"""

    SYSTEM_PROMPT_RAG_SALES = """你是一个专业的电商售后客服助手。

<核心规则>
1. 友好的语气，主动帮助用户解决问题。
2. 基于知识库回答退换货政策、物流查询、售后服务等问题。
3. 对于订单状态查询（含订单号），回复查询方法并建议用户提供更多信息。
4. 遇到投诉或强烈负面情绪时，表达理解并主动建议转接人工客服。
</核心规则>"""

    SYSTEM_PROMPT_RAG_IT = """你是一个专业的IT技术支持助手（Help Desk）。

<核心规则>
1. 专业、准确、步骤清晰。
2. 基于知识库提供技术问题的解决方案。
3. 如果用户问题不在你的知识范围内，明确说明并建议提交工单。
4. 涉及系统权限、密码重置等敏感操作，必须转接人工坐席。
</核心规则>"""

    # === User Message 模板 ===
    USER_MESSAGE_WITH_CONTEXT = """<用户问题>
{user_query}
</用户问题>

<参考知识>
{retrieved_context}
</参考知识>

请根据上述参考知识回答用户问题。请：
1. 如果参考知识足够回答，直接给出准确答案并标注来源编号
2. 如果参考知识不足，明确表示"根据现有知识库我无法确定"，建议联系人工客服
3. 保持回答简洁、有条理"""

    USER_MESSAGE_WITH_CONTEXT_AND_HISTORY = """<对话历史>
{conversation_summary}

{recent_messages}
</对话历史>

<用户当前问题>
{user_query}
</用户当前问题>

<参考知识>
{retrieved_context}
</参考知识>

请结合对话历史和参考知识回答。如果在对话历史中提到过相关信息，请关联上下文。"""

    # === 意图分类 Prompt ===
    INTENT_CLASSIFICATION_PROMPT = """请将以下用户消息分类到对应的意图类别。

用户消息："{user_query}"

类别定义：
- order_status：查询订单状态、物流信息
- return_exchange：退换货、退款咨询
- product_inquiry：产品信息、规格、价格咨询
- complaint：投诉、不满表达
- technical_issue：技术问题、使用问题
- account_issue：账号、密码、权限问题
- greeting：问候、寒暄
- unknown：无法归类

请以 JSON 格式输出：
```json
{{"intent": "类别", "confidence": 0.95, "reason": "简短分类理由"}}
```"""

    # === 拒答模板 ===
    UNCERTAIN_RESPONSE_TEMPLATE = (
        "根据我目前的知识库，无法为您确认这个问题的答案。"
        "我建议您：\n"
        "1. 换个方式重新描述您的问题\n"
        "2. 联系人工客服获取更准确的帮助（回复"人工"即可转接）\n"
        "3. 查看我们的帮助中心获取更多信息"
    )

    # === 澄清追问模板 ===
    CLARIFICATION_TEMPLATE = (
        "为了更好地帮助您，我需要确认一下：\n{clarification_questions}\n"
        "请提供更多信息，我会为您更准确地解答。"
    )

    @classmethod
    def get_system_prompt(cls, scenario: str = "general") -> str:
        """按场景获取系统提示词"""
        prompts = {
            "general": cls.SYSTEM_PROMPT_RAG,
            "sales": cls.SYSTEM_PROMPT_RAG_SALES,
            "it": cls.SYSTEM_PROMPT_RAG_IT,
        }
        return prompts.get(scenario, cls.SYSTEM_PROMPT_RAG)

    @classmethod
    def build_rag_messages(
        cls,
        user_query: str,
        retrieved_context: str,
        conversation_history: Optional[str] = None,
        recent_messages: Optional[str] = None,
        scenario: str = "general",
    ) -> List[Dict[str, str]]:
        """
        构建完整的 RAG 消息列表

        参数:
          user_query: 用户当前问题
          retrieved_context: 检索到的参考知识（已格式化）
          conversation_history: 对话摘要（长期记忆）
          recent_messages: 最近消息文本（短期记忆）
          scenario: 场景标识

        返回:
          OpenAI 格式的 messages 列表
        """
        system_prompt = cls.get_system_prompt(scenario)

        if conversation_history or recent_messages:
            user_message = cls.USER_MESSAGE_WITH_CONTEXT_AND_HISTORY.format(
                conversation_summary=conversation_history or "（无历史对话）",
                recent_messages=recent_messages or "",
                user_query=user_query,
                retrieved_context=retrieved_context,
            )
        else:
            user_message = cls.USER_MESSAGE_WITH_CONTEXT.format(
                user_query=user_query,
                retrieved_context=retrieved_context,
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]


# ============================================================================
# DeepSeek API 客户端封装
# ============================================================================
class DeepSeekClient:
    """
    DeepSeek API 客户端

    封装了 OpenAI 兼容接口，提供：
      - 同步/流式调用
      - Token 计数
      - 自动重试
      - 结构化输出日志
    """

    def __init__(self):
        config = get_config().llm
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.chat_model = config.chat_model
        self.max_output_tokens = config.max_output_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict:
        """
        同步对话调用

        返回: {content, usage, finish_reason, model}
        """
        max_tokens = max_tokens or self.max_output_tokens

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )

            choice = response.choices[0]
            return {
                "content": choice.message.content or "",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                "finish_reason": choice.finish_reason,
                "model": response.model,
            }

        except Exception as e:
            logger.error(f"[DeepSeekClient] API 调用失败: {e}")
            raise

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[TokenStream, None, Dict]:
        """
        流式对话调用

        使用 Generator 模式，逐步 yield 每个 token。

        用法:
            client = DeepSeekClient()
            for token in client.chat_stream(messages):
                print(token.content, end="", flush=True)
        """
        max_tokens = max_tokens or self.max_output_tokens
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            stream = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None

                if delta and delta.content:
                    yield TokenStream(
                        content=delta.content,
                        finish_reason=None,
                    )

                if chunk.choices and chunk.choices[0].finish_reason:
                    # 记录 usage（在最后一个 chunk 中）
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens or 0,
                            "completion_tokens": chunk.usage.completion_tokens or 0,
                            "total_tokens": chunk.usage.total_tokens or 0,
                        }
                    yield TokenStream(
                        content="",
                        finish_reason=chunk.choices[0].finish_reason,
                    )

        except Exception as e:
            logger.error(f"[DeepSeekClient] 流式调用失败: {e}")
            raise

        return usage


# ============================================================================
# RAG 生成引擎
# ============================================================================
class RAGGenerator:
    """
    RAG 生成引擎 — 将检索结果转化为最终回答

    核心流程：
      1. 构建检索上下文（格式化来源内容）
      2. 组装 Prompt + 检索结果
      3. 调用 DeepSeek 生成回答
      4. 解析来源引用
      5. 判断是否需要拒答

    输出约束策略：
      - System Prompt 明确"只能基于参考知识回答"
      - User Message 格式化的上下文注入
      - 后处理检查是否有来源标注
    """

    def __init__(self, scenario: str = "general"):
        self.scenario = scenario
        self.llm = DeepSeekClient()
        self.config = get_config()

    def generate(
        self,
        user_query: str,
        search_result: SearchResult,
        conversation_summary: Optional[str] = None,
        recent_messages: Optional[str] = None,
        stream: bool = False,
    ) -> RAGResponse:
        """
        生成 RAG 回答

        参数:
          user_query: 用户问题
          search_result: 检索结果
          conversation_summary: 对话摘要
          recent_messages: 最近消息
          stream: 是否流式输出

        返回:
          RAGResponse
        """
        logger.info(
            f"[RAGGenerator] 生成回答: query='{user_query[:50]}...', "
            f"retrieved={search_result.result_count} chunks, "
            f"top_similarity={search_result.top_similarity:.3f}"
        )

        # Step 1: 知识边界预检
        if not search_result.is_reliable:
            logger.info(f"[RAGGenerator] 检索结果不可靠，触发拒答")
            return RAGResponse(
                content=PromptTemplates.UNCERTAIN_RESPONSE_TEMPLATE,
                response_type=ResponseType.UNCERTAIN,
                confidence=search_result.top_similarity,
                metadata={
                    "reason": "retrieval_below_threshold",
                    "top_similarity": search_result.top_similarity,
                },
            )

        # Step 2: 构建检索上下文
        retrieved_context = self._build_context(search_result)

        # Step 3: 构建消息
        messages = PromptTemplates.build_rag_messages(
            user_query=user_query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_summary,
            recent_messages=recent_messages,
            scenario=self.scenario,
        )

        # Step 4: 调用 LLM 生成
        if stream:
            # 流式模式：返回生成器和元数据，由上层处理
            return self._generate_stream(messages, search_result, user_query)
        else:
            return self._generate_sync(messages, search_result, user_query)

    def _build_context(self, search_result: SearchResult) -> str:
        """构建检索上下文文本"""
        if not search_result.results:
            return "（未找到相关参考知识）"

        parts = []
        for i, result in enumerate(search_result.results):
            parts.append(result.to_context_string(i + 1))

        return "\n---\n" + "\n---\n".join(parts)

    def _generate_sync(
        self,
        messages: List[Dict[str, str]],
        search_result: SearchResult,
        user_query: str,
    ) -> RAGResponse:
        """同步生成"""
        response = self.llm.chat(
            messages=messages,
            temperature=self.config.llm.temperature,
        )

        # 解析来源引用
        citations = self._extract_citations(
            response["content"],
            search_result.results,
        )

        # 判断回答类型
        response_type = self._classify_response(
            response["content"],
            search_result,
        )

        # 计算置信度
        confidence = self._calculate_confidence(
            search_result,
            response["content"],
            citations,
        )

        return RAGResponse(
            content=response["content"],
            response_type=response_type,
            citations=citations,
            confidence=confidence,
            metadata={
                "model": response.get("model", ""),
                "finish_reason": response.get("finish_reason", ""),
                "retrieved_count": search_result.result_count,
                "top_similarity": search_result.top_similarity,
            },
            usage=response.get("usage", {}),
        )

    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        search_result: SearchResult,
        user_query: str,
    ) -> RAGResponse:
        """
        流式生成

        返回的 RAGResponse.content 是空的，实际内容通过 TokenStream 流式输出。
        此处仅返回元数据。调用方通过 generate_stream() 方法获取 TokenStream。
        """
        # 返回占位响应，实际流通过单独方法处理
        return RAGResponse(
            content="",
            response_type=ResponseType.KNOWLEDGE_BASED,
            confidence=search_result.top_similarity,
            metadata={"stream": True},
        )

    def generate_stream(
        self,
        user_query: str,
        search_result: SearchResult,
        conversation_summary: Optional[str] = None,
        recent_messages: Optional[str] = None,
    ) -> Generator[Tuple[str, Dict], None, RAGResponse]:
        """
        流式生成器方法

        每次 yield (content_chunk, metadata_dict)
        最终返回完整的 RAGResponse

        用法（SSE 模式）:
            gen = RAGGenerator()
            for chunk, meta in gen.generate_stream(query, result):
                # 发送 SSE event
                yield f"data: {json.dumps({'content': chunk})}\\n\\n"
        """
        if not search_result.is_reliable:
            content = PromptTemplates.UNCERTAIN_RESPONSE_TEMPLATE
            word_by_word = list(content)
            for i in range(0, len(word_by_word), 3):
                chunk = "".join(word_by_word[i:i + 3])
                yield chunk, {"type": "token"}
            return RAGResponse(
                content=content,
                response_type=ResponseType.UNCERTAIN,
                confidence=search_result.top_similarity,
            )

        # 构建上下文和消息
        retrieved_context = self._build_context(search_result)
        messages = PromptTemplates.build_rag_messages(
            user_query=user_query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_summary,
            recent_messages=recent_messages,
            scenario=self.scenario,
        )

        # 流式调用 LLM
        full_content = ""
        usage = {}

        try:
            stream_gen = self.llm.chat_stream(
                messages=messages,
                temperature=self.config.llm.temperature,
            )

            for token in stream_gen:
                if token.content:
                    full_content += token.content
                    yield token.content, {"type": "token"}
                if token.finish_reason:
                    yield "", {"type": "done", "finish_reason": token.finish_reason}

        except Exception as e:
            logger.error(f"[RAGGenerator] 流式生成失败: {e}")
            error_msg = f"抱歉，生成回答时出现错误，请稍后重试。"
            yield error_msg, {"type": "error", "error": str(e)}
            full_content = error_msg

        # 构建最终响应
        citations = self._extract_citations(full_content, search_result.results)
        response_type = self._classify_response(full_content, search_result)
        confidence = self._calculate_confidence(search_result, full_content, citations)

        return RAGResponse(
            content=full_content,
            response_type=response_type,
            citations=citations,
            confidence=confidence,
            metadata={
                "retrieved_count": search_result.result_count,
                "top_similarity": search_result.top_similarity,
            },
            usage=usage,
        )

    def _extract_citations(
        self,
        content: str,
        results: List[RetrievalResult],
    ) -> List[Citation]:
        """
        从生成内容中提取来源引用

        匹配模式：[来源: 1], [来源: 2,3], [来源:1]
        """
        citations = []
        # 匹配所有来源引用标记
        pattern = r"\[来源:\s*([\d,\s]+)\]"
        matches = re.findall(pattern, content)

        cited_indices = set()
        for match in matches:
            for num_str in match.split(","):
                try:
                    idx = int(num_str.strip()) - 1  # 转换为 0-based
                    if 0 <= idx < len(results):
                        cited_indices.add(idx)
                except ValueError:
                    continue

        # 构建 Citation 对象
        for idx in sorted(cited_indices):
            result = results[idx]
            citations.append(Citation(
                index=idx + 1,
                document_title=result.metadata.get("title", "未知文档"),
                content_snippet=result.content[:200],
                relevance_score=result.final_score,
            ))

        return citations

    def _classify_response(
        self,
        content: str,
        search_result: SearchResult,
    ) -> ResponseType:
        """判断回答类型"""
        content_lower = content.lower()

        # 拒答信号
        uncertain_signals = [
            "无法确定", "无法确认", "不确定", "无法回答",
            "没有相关信息", "知识库中没有", "无法为您确认",
            "i cannot", "i don't know", "unable to",
        ]
        if any(sig in content_lower for sig in uncertain_signals):
            return ResponseType.UNCERTAIN

        # 转人工信号
        handoff_signals = [
            "转接人工", "联系人工客服", "人工客服", "提交工单",
            "转人工", "人工坐席",
        ]
        if any(sig in content_lower for sig in handoff_signals):
            return ResponseType.HANDOFF

        # 澄清信号
        clarification_signals = [
            "请问", "能否提供", "请提供", "具体是哪", "您是指",
            "需要确认", "进一步说明",
        ]
        if any(sig in content_lower for sig in clarification_signals):
            return ResponseType.CLARIFICATION

        # 基于知识的回答
        if search_result.is_reliable:
            return ResponseType.KNOWLEDGE_BASED

        return ResponseType.UNCERTAIN

    def _calculate_confidence(
        self,
        search_result: SearchResult,
        content: str,
        citations: List[Citation],
    ) -> float:
        """
        计算回答置信度

        综合考虑：
          1. 检索最高相似度 (40% 权重)
          2. 是否有引用来源 (30% 权重)
          3. 检索结果数量可靠性 (20% 权重)
          4. 内容长度合理性 (10% 权重)
        """
        scores = []

        # 1. 检索相似度得分
        retrieval_score = min(search_result.top_similarity / 0.9, 1.0) if search_result.top_similarity > 0 else 0.0
        scores.append(("retrieval_similarity", retrieval_score, 0.4))

        # 2. 来源引用得分
        citation_score = min(len(citations) / 3.0, 1.0) if citations else 0.0
        scores.append(("source_coverage", citation_score, 0.3))

        # 3. 结果数量得分
        count_score = min(search_result.result_count / 5.0, 1.0)
        scores.append(("result_count", count_score, 0.2))

        # 4. 内容长度得分（太短可能不完整）
        content_len = len(content)
        if content_len < 20:
            length_score = 0.2
        elif content_len < 50:
            length_score = 0.5
        elif content_len < 100:
            length_score = 0.8
        else:
            length_score = 1.0
        scores.append(("content_length", length_score, 0.1))

        # 加权求和
        total = sum(score * weight for _, score, weight in scores)
        return round(total, 4)


# ============================================================================
# 意图识别器
# ============================================================================
class IntentClassifier:
    """
    意图识别器

    使用 DeepSeek 做 Few-shot 意图分类。
    生产环境可在前面加一层规则匹配作为快速通道。
    """

    INTENTS = [
        "order_status",
        "return_exchange",
        "product_inquiry",
        "complaint",
        "technical_issue",
        "account_issue",
        "greeting",
        "unknown",
    ]

    def __init__(self):
        self.llm = DeepSeekClient()
        self.config = get_config()

    def classify(self, user_message: str) -> Dict:
        """
        分类用户意图

        返回: {intent, confidence, reason}
        """
        # 快速规则匹配（60% 场景命中，节省 API 调用）
        quick_match = self._rule_based_classify(user_message)
        if quick_match and quick_match["confidence"] > 0.9:
            logger.debug(f"[IntentClassifier] 规则命中: {quick_match['intent']}")
            return quick_match

        # LLM 分类（35% 场景）
        try:
            prompt = PromptTemplates.INTENT_CLASSIFICATION_PROMPT.format(
                user_query=user_message
            )
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个准确的用户意图分类专家。只输出JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.llm.intent_temperature,
                max_tokens=150,
            )

            result_text = response["content"]
            # 提取 JSON
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                intent = result.get("intent", "unknown")
                confidence = float(result.get("confidence", 0.5))
                reason = result.get("reason", "")

                # 验证意图值
                if intent not in self.INTENTS:
                    intent = "unknown"
                    confidence = 0.3

                return {
                    "intent": intent,
                    "confidence": confidence,
                    "reason": reason,
                }

        except Exception as e:
            logger.error(f"[IntentClassifier] LLM 分类失败: {e}")

        # 降级
        return {"intent": "unknown", "confidence": 0.1, "reason": "classification_failed"}

    def _rule_based_classify(self, message: str) -> Optional[Dict]:
        """
        基于规则的快速意图匹配

        适合高频、确定性强的场景。
        """
        message_lower = message.lower().strip()

        # 定义规则
        rules = [
            (["你好", "嗨", "hello", "hi", "在吗", "您好"], "greeting", 0.95),
            (["订单", "物流", "发货", "快递", "运单", "tracking"], "order_status", 0.90),
            (["退货", "退款", "换货", "退换", "return", "refund"], "return_exchange", 0.92),
            (["投诉", "举报", "差评", "态度差", "太差"], "complaint", 0.95),
            (["产品", "规格", "价格", "多少钱", "参数"], "product_inquiry", 0.85),
            (["密码", "登录", "账号", "注册", "绑定"], "account_issue", 0.90),
            (["报错", "打不开", "闪退", "安装", "配置", "bug"], "technical_issue", 0.85),
        ]

        for keywords, intent, confidence in rules:
            if any(kw in message_lower for kw in keywords):
                return {
                    "intent": intent,
                    "confidence": confidence,
                    "reason": f"rule_match: {keywords[0]}",
                }

        return None


# ============================================================================
# 独立测试入口
# ============================================================================
if __name__ == "__main__":
    """
    快速验证生成层功能：

        python generation_layer.py
    """
    logger.info("=" * 60)
    logger.info("DeepService Generation Layer — 独立测试")
    logger.info("=" * 60)

    # 测试意图识别
    classifier = IntentClassifier()
    test_messages = [
        "我的订单12345发货了吗？",
        "你好",
        "我要退货，质量太差了！",
    ]
    for msg in test_messages:
        result = classifier.classify(msg)
        logger.info(f"意图: {msg} → {result['intent']} (置信度: {result['confidence']})")

    # 测试 Prompt 构建
    from retrieval_layer import RetrievalResult

    mock_results = [
        RetrievalResult(
            chunk_id="1",
            content="自签收之日起7天内可申请退货。质量问题退换货运费由商家承担。",
            metadata={"title": "退换货政策", "section": "退换货条件"},
            final_score=0.92,
            source_label="[来源: 1]",
        ),
    ]

    search_result = SearchResult(
        query="如何退货？",
        results=mock_results,
        top_similarity=0.92,
        result_count=1,
    )

    generator = RAGGenerator()
    ctx = generator._build_context(search_result)
    logger.info(f"检索上下文:\n{ctx}")

    logger.info("=" * 60)
    logger.info("生成层测试完成 ✓")

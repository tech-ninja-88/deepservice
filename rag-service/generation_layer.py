"""RAG generation — prompt templates, citation extraction, streaming, and response classification."""

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Generator, Tuple
from enum import Enum

from openai import OpenAI
from loguru import logger

from config import get_config
from retrieval_layer import SearchResult, RetrievalResult


# /// Data structures
class ResponseType(str, Enum):
    """Response classification enum."""
    KNOWLEDGE_BASED = "knowledge_based"       # knowledge-base answer
    UNCERTAIN = "uncertain"                    # cannot determine
    CLARIFICATION = "clarification"            # needs clarification
    HANDOFF = "handoff"                        # suggest human handoff
    GREETING = "greeting"                      # greeting
    CHITCHAT = "chitchat"                      # casual chat


@dataclass
class TokenStream:
    """Single streaming token chunk."""
    content: str
    finish_reason: Optional[str] = None


@dataclass
class Citation:
    """A cited source extracted from the generated response."""
    index: int                                # citation number [source: 1]
    document_title: str                       # document title
    content_snippet: str                      # cited content snippet
    relevance_score: float                    # relevance score


@dataclass
class RAGResponse:
    """Full RAG response including content, type, citations, and metadata."""
    content: str                              # answer body
    response_type: ResponseType               # response classification
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0                   # confidence score (0-1)
    metadata: Dict = field(default_factory=dict)
    usage: Dict = field(default_factory=dict)  # token usage

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
        """Render the response as Markdown with source citations."""
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


# /// Prompt template management
class PromptTemplates:
    """
    Prompt template library — the "source code" for LLM responses.

    Versioned templates with system prompts, user message templates, and output constraints.
    """

    # === System prompts ===
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

    # === User message templates ===
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

    # === Intent classification prompt ===
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

    # === Uncertain response template ===
    UNCERTAIN_RESPONSE_TEMPLATE = (
        "根据我目前的知识库，无法为您确认这个问题的答案。"
        "我建议您：\n"
        "1. 换个方式重新描述您的问题\n"
        '2. 联系人工客服获取更准确的帮助（回复"人工"即可转接）\n'
        "3. 查看我们的帮助中心获取更多信息"
    )

    # === Clarification template ===
    CLARIFICATION_TEMPLATE = (
        "为了更好地帮助您，我需要确认一下：\n{clarification_questions}\n"
        "请提供更多信息，我会为您更准确地解答。"
    )

    @classmethod
    def get_system_prompt(cls, scenario: str = "general") -> str:
        """Return the system prompt for the given scenario."""
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
        """Build a complete messages list for the RAG call (system + user)."""
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


# /// DeepSeek API client wrapper
class DeepSeekClient:
    """OpenAI-compatible client for DeepSeek with sync/stream support."""

    def __init__(self):
        from config import get_llm_client
        config = get_config().llm
        self.client = get_llm_client()
        self._available = self.client is not None
        self.chat_model = config.chat_model
        self.max_output_tokens = config.max_output_tokens

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict:
        """Synchronous chat call. Returns {content, usage, finish_reason, model}."""
        if not self._available:
            return {
                "content": "LLM service is not configured. Please set DEEPSEEK_API_KEY.",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "finish_reason": "error",
                "model": "unavailable",
            }
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
            logger.error(f"[DeepSeekClient] API call failed: {e}")
            return {
                "content": f"抱歉，AI 服务暂时不可用（{str(e)[:80]}），请稍后重试。",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "finish_reason": "error",
                "model": "error",
            }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Generator[TokenStream, None, Dict]:
        """Streaming chat call using a Generator that yields one token at a time."""
        if not self._available:
            yield TokenStream(content="LLM service is not configured. Please set DEEPSEEK_API_KEY.")
            yield TokenStream(content="", finish_reason="error")
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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
                    # Record usage (on final chunk)
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
            logger.error(f"[DeepSeekClient] Stream call failed: {e}")
            yield TokenStream(content=f"抱歉，AI 服务暂时不可用（{str(e)[:80]}），请稍后重试。")
            yield TokenStream(content="", finish_reason="error")

        return usage


# /// RAG generation engine
class RAGGenerator:
    """
    RAG generation engine — transforms retrieval results into a final answer.

    Pipeline: build context -> assemble prompt + results -> call LLM -> parse citations -> classify response.
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
        """Generate a RAG response from user query and search results."""
        logger.info(
            f"[RAGGenerator] generating: query='{user_query[:50]}...', "
            f"retrieved={search_result.result_count} chunks, "
            f"top_similarity={search_result.top_similarity:.3f}"
        )

        # Step 1: knowledge boundary pre-check
        if not search_result.is_reliable:
            logger.info(f"[RAGGenerator] unreliable retrieval, triggering fallback")
            return RAGResponse(
                content=PromptTemplates.UNCERTAIN_RESPONSE_TEMPLATE,
                response_type=ResponseType.UNCERTAIN,
                confidence=search_result.top_similarity,
                metadata={
                    "reason": "retrieval_below_threshold",
                    "top_similarity": search_result.top_similarity,
                },
            )

        # Step 2: build retrieval context
        retrieved_context = self._build_context(search_result)

        # Step 3: build messages
        messages = PromptTemplates.build_rag_messages(
            user_query=user_query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_summary,
            recent_messages=recent_messages,
            scenario=self.scenario,
        )

        # Step 4: call LLM
        if stream:
            # Stream mode: return generator + metadata, handled by caller
            return self._generate_stream(messages, search_result, user_query)
        else:
            return self._generate_sync(messages, search_result, user_query)

    def _build_context(self, search_result: SearchResult) -> str:
        """Format retrieved results into a context string for the prompt."""
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
        """Synchronous generation — call LLM, parse citations, classify, score."""
        response = self.llm.chat(
            messages=messages,
            temperature=self.config.llm.temperature,
        )

        # Parse source citations
        citations = self._extract_citations(
            response["content"],
            search_result.results,
        )

        # Classify response type
        response_type = self._classify_response(
            response["content"],
            search_result,
        )

        # Calculate confidence
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
        """Streaming placeholder — returns metadata only; actual tokens via generate_stream()."""
        # Return placeholder; real stream handled by generate_stream method
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
        """Streaming generator yielding (content_chunk, metadata) tuples; returns final RAGResponse."""
        if not search_result.is_reliable:
            content = PromptTemplates.UNCERTAIN_RESPONSE_TEMPLATE
            yield content, {"type": "token"}
            return RAGResponse(
                content=content,
                response_type=ResponseType.UNCERTAIN,
                confidence=search_result.top_similarity,
            )

        # Build context and messages
        retrieved_context = self._build_context(search_result)
        messages = PromptTemplates.build_rag_messages(
            user_query=user_query,
            retrieved_context=retrieved_context,
            conversation_history=conversation_summary,
            recent_messages=recent_messages,
            scenario=self.scenario,
        )

        # Stream from LLM
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
            logger.error(f"[RAGGenerator] stream generation failed: {e}")
            error_msg = f"抱歉，生成回答时出现错误，请稍后重试。"
            yield error_msg, {"type": "error", "error": str(e)}
            full_content = error_msg

        # Build final response
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
        """Extract source citations from generated content. Matches [来源: 1] patterns."""
        citations = []
        # Match all source citation markers
        pattern = r"\[来源:\s*([\d,\s]+)\]"
        matches = re.findall(pattern, content)

        cited_indices = set()
        for match in matches:
            for num_str in match.split(","):
                try:
                    idx = int(num_str.strip()) - 1  # convert to 0-based
                    if 0 <= idx < len(results):
                        cited_indices.add(idx)
                except ValueError:
                    continue

        # Build Citation objects
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
        """Classify the response type based on signal phrases in the content."""
        content_lower = content.lower()

        # Fallback / uncertain signals
        uncertain_signals = [
            "无法确定", "无法确认", "不确定", "无法回答",
            "没有相关信息", "知识库中没有", "无法为您确认",
            "i cannot", "i don't know", "unable to",
        ]
        if any(sig in content_lower for sig in uncertain_signals):
            return ResponseType.UNCERTAIN

        # Human handoff signals
        handoff_signals = [
            "转接人工", "联系人工客服", "人工客服", "提交工单",
            "转人工", "人工坐席",
        ]
        if any(sig in content_lower for sig in handoff_signals):
            return ResponseType.HANDOFF

        # Clarification signals
        clarification_signals = [
            "请问", "能否提供", "请提供", "具体是哪", "您是指",
            "需要确认", "进一步说明",
        ]
        if any(sig in content_lower for sig in clarification_signals):
            return ResponseType.CLARIFICATION

        # Knowledge-based answer
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
        Calculate response confidence as a weighted score:
          1. retrieval top similarity (40%)
          2. source citations (30%)
          3. result count reliability (20%)
          4. content length adequacy (10%)
        """
        scores = []

        # 1. Retrieval similarity score
        retrieval_score = min(search_result.top_similarity / 0.9, 1.0) if search_result.top_similarity > 0 else 0.0
        scores.append(("retrieval_similarity", retrieval_score, 0.4))

        # 2. Source citation score
        citation_score = min(len(citations) / 3.0, 1.0) if citations else 0.0
        scores.append(("source_coverage", citation_score, 0.3))

        # 3. Result count score
        count_score = min(search_result.result_count / 5.0, 1.0)
        scores.append(("result_count", count_score, 0.2))

        # 4. Content length score (too short may be incomplete)
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

        # Weighted sum
        total = sum(score * weight for _, score, weight in scores)
        return round(total, 4)


# /// Self-check
if __name__ == "__main__":
    """Quick validation of generation layer: python generation_layer.py"""
    logger.info("--- Generation Layer self-check ---")

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
    logger.info(f"Context built:\n{ctx}")

    logger.info("Self-check complete.")

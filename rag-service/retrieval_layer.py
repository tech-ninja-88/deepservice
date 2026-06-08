"""
=============================================================================
DeepService RAG — 混合检索模块 (Retrieval Layer)
=============================================================================
职责：
  1. 语义向量检索 — 基于 Embedding 的语义相似度搜索
  2. 关键词 BM25 检索 — 精确关键词匹配
  3. 混合检索融合 — RRF 算法融合语义+关键词结果
  4. 重排序 (Re-ranking) — 精细化打分，过滤低相关度干扰项

企业级设计原则：
  - 单一检索方式存在盲区：向量检索对专有名词不敏感，关键词检索无法理解语义
  - 混合检索 + 重排序是当前工业界最成熟的方案
  - 重排序模型可显著提升 Top-5 召回精度

参考：
  [reference:3] — 在初步召回后引入重排序模型进行精细化打分，过滤相关度低于阈值的干扰项
=============================================================================
"""

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from config import get_config, RetrievalConfig
from data_layer import VectorStoreManager, EmbeddingGenerator, Chunk


# ============================================================================
# 检索结果数据结构
# ============================================================================
@dataclass
class RetrievalResult:
    """统一检索结果"""
    chunk_id: str
    content: str
    metadata: Dict = field(default_factory=dict)

    # 各阶段得分
    vector_score: float = 0.0       # 向量检索得分（归一化后）
    bm25_score: float = 0.0         # BM25 检索得分（归一化后）
    fusion_score: float = 0.0       # 融合得分（RRF）
    rerank_score: float = 0.0       # 重排序得分
    final_score: float = 0.0        # 最终综合得分

    # 用于输出来源标注
    source_label: str = ""          # 格式化的来源标签，如 "[来源: 1]"

    def to_context_string(self, index: int = 1) -> str:
        """生成注入 Prompt 的参考文本"""
        title = self.metadata.get("title", "未知文档")
        section = self.metadata.get("section", "")
        source_info = f"{title}"
        if section:
            source_info += f" > {section}"
        return (
            f"[来源: {index}] (相关度: {self.final_score:.2f}) {source_info}\n"
            f"{self.content}\n"
        )


@dataclass
class SearchResult:
    """完整检索结果"""
    query: str
    results: List[RetrievalResult]

    # 检索质量元数据
    top_similarity: float = 0.0     # 最高相似度
    avg_similarity: float = 0.0     # 平均相似度
    result_count: int = 0           # 有效结果数

    @property
    def is_reliable(self) -> bool:
        """判断检索结果是否可靠（用于知识边界判断）"""
        config = get_config().retrieval
        return (
            self.top_similarity >= config.vector_similarity_threshold
            and self.result_count >= 1
        )


# ============================================================================
# 1. 语义向量检索器
# ============================================================================
class VectorRetriever:
    """
    基于 Embedding 的语义向量检索

    优势：
      - 理解语义：能匹配"退钱"和"退款"这样的同义表达
      - 跨语言：中文查询可匹配英文文档（如果 embedding 模型支持）

    局限：
      - 对专有名词（产品型号、订单号）不敏感
      - 可能将语义相似但无关的文档排在前面
    """

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStoreManager()
        self.config = get_config().retrieval

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where_filter: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        执行语义向量检索

        参数:
          query: 查询文本
          top_k: 返回数量
          where_filter: 元数据过滤（如限制特定文档类型）

        返回: [(result_dict, similarity_score), ...]
        """
        top_k = top_k or self.config.top_k_vector

        # 1. 查询向量化
        query_embedding = self.embedder.embed(query)

        # 2. 向量检索
        results = self.vector_store.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            where_filter=where_filter,
        )

        return [(r, r["similarity"]) for r in results]


# ============================================================================
# 2. BM25 关键词检索器
# ============================================================================
class BM25Retriever:
    """
    基于 BM25 算法的关键词检索

    BM25 (Best Match 25) — 信息检索领域的经典算法：
      - TF-IDF 的改进版，考虑了文档长度归一化
      - 对精确关键词匹配效果极佳

    适用场景：
      - 产品型号查询："SKU-2024-PRO"
      - 错误码查询："Error 503"
      - 专有名词："VIP会员升级规则"

    注意：
      - BM25 基于词频统计，需要分词支持
      - 这里实现了简化的中文+英文混合分词
    """

    def __init__(self):
        self.config = get_config().retrieval
        self._corpus: List[str] = []
        self._corpus_metadata: List[Dict] = []
        self._bm25 = None
        self._initialized = False

    def build_index(self, chunks: List[Dict]):
        """
        从数据库加载所有 chunk 并构建 BM25 索引

        注意：BM25 索引需要全量数据，应在知识库更新后重建。
        """
        from rank_bm25 import BM25Okapi

        self._corpus = []
        self._corpus_metadata = []

        for chunk in chunks:
            content = chunk.get("content") or chunk.get("document", "")
            if content:
                tokens = self._tokenize(content)
                self._corpus.append(tokens)
                self._corpus_metadata.append({
                    "id": chunk.get("id", ""),
                    "content": content,
                    "metadata": chunk.get("metadata", {}),
                })

        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus)
            self._initialized = True
            logger.info(f"[BM25Retriever] 索引构建完成: {len(self._corpus)} 个文档")

    def _tokenize(self, text: str) -> List[str]:
        """
        中文+英文混合分词

        简化实现（生产环境建议接入 jieba 或 LAC）：
          - 中文：按字符 bigram 切分（"退换货" → ["退换", "换货"]）
          - 英文：按空格和标点切分
          - 数字/特殊符号：保留连续序列
        """
        tokens = []

        # 分离中文和非中文部分
        # 中文部分做 bigram
        chinese_chars = re.findall(r"[一-鿿]+", text)
        for segment in chinese_chars:
            for i in range(len(segment) - 1):
                tokens.append(segment[i:i + 2])
            tokens.append(segment[-1])  # 单字也保留

        # 英文/数字部分
        non_chinese = re.findall(r"[a-zA-Z0-9]+", text)
        tokens.extend([t.lower() for t in non_chinese])

        # 独立的数字/特殊标识
        special = re.findall(r"\d+\.?\d*", text)
        tokens.extend(special)

        return tokens

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        执行 BM25 关键词检索

        返回: [(result_dict, bm25_score), ...]
        """
        if not self._initialized:
            logger.warning("[BM25Retriever] 索引未初始化，返回空结果")
            return []

        top_k = top_k or self.config.top_k_bm25
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # BM25 检索
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = scores.max() if scores.max() > 0 else 1.0  # 避免除零

        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                meta = self._corpus_metadata[idx]
                results.append(({
                    "id": meta["id"],
                    "content": meta["content"],
                    "metadata": meta["metadata"],
                    "similarity": normalized_score,  # 与其他检索器统一字段名
                }, normalized_score))

        return results

    @staticmethod
    def load_chunks_from_store() -> List[Dict]:
        """
        从 ChromaDB 加载所有 chunk 用于构建 BM25 索引

        注意：大规模知识库（>10万条）建议使用数据库全文索引替代内存 BM25。
        """
        store = VectorStoreManager()
        # ChromaDB get 方法获取全部数据
        try:
            results = store.collection.get(include=["documents", "metadatas"])
            chunks = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    chunks.append({
                        "id": chunk_id,
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })
            return chunks
        except Exception as e:
            logger.error(f"[BM25Retriever] 加载数据失败: {e}")
            return []


# ============================================================================
# 3. 混合检索融合器
# ============================================================================
class HybridRetriever:
    """
    混合检索融合器

    算法：Reciprocal Rank Fusion (RRF)
      RRF 是当前工业界最常用的混合检索融合算法，优势在于：
      1. 无需归一化不同检索器的得分分布
      2. 对排名敏感而对绝对分值不敏感
      3. 简单高效，可解释性强

    RRF 公式：
      RRF(d) = Σ 1 / (k + rank_i(d))

      其中：
        - d: 文档
        - rank_i(d): 文档 d 在检索器 i 中的排名
        - k: 平滑参数（通常为 60），防止排名靠后的文档被过度抑制

    同时也支持加权线性融合：
      fusion_score = α × vector_score + β × bm25_score
    """

    def __init__(self):
        self.config = get_config().retrieval
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: str = "rrf",  # "rrf" | "weighted" | "vector_only" | "bm25_only"
    ) -> SearchResult:
        """
        混合检索主入口

        流程：
          1. 并行调用向量检索和 BM25 检索
          2. RRF 融合排序
          3. 返回 SearchResult
        """
        top_k = top_k or self.config.top_k_fusion
        logger.info(f"[HybridRetriever] 混合检索: '{query[:50]}...' (strategy={strategy})")

        # Step 1: 并行检索
        vector_results = self.vector_retriever.search(query)
        bm25_results = self.bm25_retriever.search(query)

        logger.debug(
            f"[HybridRetriever] 向量检索: {len(vector_results)} 条, "
            f"BM25检索: {len(bm25_results)} 条"
        )

        # Step 2: 融合
        if strategy == "vector_only":
            fused = self._fuse_weighted(vector_results, [], top_k)
        elif strategy == "bm25_only":
            fused = self._fuse_weighted([], bm25_results, top_k)
        elif strategy == "rrf":
            fused = self._fuse_rrf(vector_results, bm25_results, top_k)
        else:  # weighted
            fused = self._fuse_weighted(vector_results, bm25_results, top_k)

        # Step 3: 构建结果
        results = []
        for result_dict, score_map in fused:
            results.append(RetrievalResult(
                chunk_id=result_dict["id"],
                content=result_dict["content"],
                metadata=result_dict.get("metadata", {}),
                vector_score=score_map.get("vector", 0.0),
                bm25_score=score_map.get("bm25", 0.0),
                fusion_score=score_map.get("fusion", 0.0),
                final_score=score_map.get("fusion", 0.0),
            ))

        # 计算质量元数据
        top_sim = results[0].final_score if results else 0.0
        avg_sim = sum(r.final_score for r in results) / len(results) if results else 0.0

        return SearchResult(
            query=query,
            results=results,
            top_similarity=top_sim,
            avg_similarity=avg_sim,
            result_count=len(results),
        )

    def _fuse_rrf(
        self,
        vector_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[Dict, Dict[str, float]]]:
        """
        Reciprocal Rank Fusion (RRF) 融合

        参数:
          k: RRF 平滑参数，默认 60（经典值）
        """
        # 建立文档 ID 到数据的映射
        id_to_data: Dict[str, Dict] = {}
        id_to_ranks: Dict[str, Dict[str, int]] = {}

        # 记录各检索器排名
        for rank, (result, _) in enumerate(vector_results, start=1):
            chunk_id = result["id"]
            id_to_data[chunk_id] = result
            id_to_ranks.setdefault(chunk_id, {})["vector"] = rank

        for rank, (result, _) in enumerate(bm25_results, start=1):
            chunk_id = result["id"]
            if chunk_id not in id_to_data:
                id_to_data[chunk_id] = result
            id_to_ranks.setdefault(chunk_id, {})["bm25"] = rank

        # 计算 RRF 得分
        rrf_scores = {}
        for chunk_id, ranks in id_to_ranks.items():
            score = 0.0
            if "vector" in ranks:
                score += 1.0 / (k + ranks["vector"])
            if "bm25" in ranks:
                score += 1.0 / (k + ranks["bm25"])
            rrf_scores[chunk_id] = score

        # 按 RRF 得分降序排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # 构建带得分信息的返回
        result = []
        for chunk_id in top_ids:
            data = id_to_data[chunk_id]
            ranks = id_to_ranks.get(chunk_id, {})

            # 反算归一化得分（用于展示）
            vector_score = 1.0 / (1 + ranks.get("vector", 999)) if "vector" in ranks else 0.0
            bm25_score = 1.0 / (1 + ranks.get("bm25", 999)) if "bm25" in ranks else 0.0

            result.append((data, {
                "vector": vector_score,
                "bm25": bm25_score,
                "fusion": rrf_scores[chunk_id],
            }))

        logger.debug(
            f"[HybridRetriever] RRF 融合: {len(id_to_data)} 个唯一文档, "
            f"Top-{top_k} 已选出"
        )
        return result

    def _fuse_weighted(
        self,
        vector_results: List[Tuple[Dict, float]],
        bm25_results: List[Tuple[Dict, float]],
        top_k: int,
    ) -> List[Tuple[Dict, Dict[str, float]]]:
        """
        加权线性融合

        公式: score = α × vector_score + β × bm25_score

        优势：
          - 可调节两种检索方式的重要性
          - 得分分布直观
        """
        alpha = self.config.vector_weight
        beta = self.config.bm25_weight

        # 归一化各检索器得分
        def normalize(scores: List[float]) -> List[float]:
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        vec_scores = [s for _, s in vector_results]
        bm_scores = [s for _, s in bm25_results]
        vec_scores_norm = normalize(vec_scores)
        bm_scores_norm = normalize(bm_scores)

        # 建立文档 ID 到加权得分的映射
        id_to_data: Dict[str, Dict] = {}
        id_to_weighted: Dict[str, float] = {}
        id_to_detail: Dict[str, Dict[str, float]] = {}

        for i, (result, _) in enumerate(vector_results):
            chunk_id = result["id"]
            score = alpha * vec_scores_norm[i]
            id_to_data[chunk_id] = result
            id_to_weighted[chunk_id] = score
            id_to_detail[chunk_id] = {"vector": vec_scores_norm[i], "bm25": 0.0}

        for i, (result, _) in enumerate(bm25_results):
            chunk_id = result["id"]
            score_add = beta * bm_scores_norm[i]
            if chunk_id in id_to_weighted:
                id_to_weighted[chunk_id] += score_add
                id_to_detail[chunk_id]["bm25"] = bm_scores_norm[i]
            else:
                id_to_data[chunk_id] = result
                id_to_weighted[chunk_id] = score_add
                id_to_detail[chunk_id] = {"vector": 0.0, "bm25": bm_scores_norm[i]}

        # 按加权得分排序
        sorted_ids = sorted(id_to_weighted.keys(), key=lambda x: id_to_weighted[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        result = []
        for chunk_id in top_ids:
            detail = id_to_detail[chunk_id]
            detail["fusion"] = id_to_weighted[chunk_id]
            result.append((id_to_data[chunk_id], detail))

        return result


# ============================================================================
# 4. 重排序器 (Re-Ranker)
# ============================================================================
class Reranker:
    """
    重排序器 — 对初步召回结果进行精细化打分

    为什么需要重排序？
      - 向量检索 + BM25 是粗排（追求召回率高）
      - 粗排结果中可能混入"看起来相关但实际不匹配"的文档
      - 重排序用更强的模型（Cross-Encoder 或 LLM）做精排
      - 精排后过滤低分结果，确保送给 LLM 的都是高质量上下文

    实现方案（按效果排序）：
      1. Cross-Encoder 模型（如 bge-reranker-large）— 效果最好
      2. LLM 打分（DeepSeek 逐条评分）— 灵活性最高
      3. 基于 Embedding 的相似度排序 — 最简单（无需额外模型）

    当前实现：LLM-based 重排序（可切换为 Cross-Encoder）
    """

    def __init__(self):
        self.config = get_config()
        self._cross_encoder = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        对候选结果重排序

        流程：
          1. 逐条评估 query-candidate 的相关性
          2. 按相关性得分重排序
          3. 过滤低分结果
          4. 分配来源标签
        """
        top_k = top_k or self.config.retrieval.top_k_final

        if not candidates:
            return []

        logger.info(f"[Reranker] 重排序 {len(candidates)} 条候选结果")

        # 逐一评分
        for candidate in candidates:
            candidate.rerank_score = self._score_candidate(query, candidate)

        # 按重排序得分降序
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)

        # 过滤低分结果
        filtered = [
            r for r in reranked
            if r.rerank_score >= self.config.retrieval.rerank_relevance_threshold
        ]

        # 截取 top_k
        final = filtered[:top_k]

        # 分配来源标签
        for i, result in enumerate(final):
            result.source_label = f"[来源: {i + 1}]"
            result.final_score = result.rerank_score

        logger.info(
            f"[Reranker] 重排序完成: {len(candidates)} → {len(filtered)} "
            f"(过滤 {len(candidates) - len(filtered)} 条低分) → Top-{len(final)}"
        )

        return final

    def _score_candidate(
        self,
        query: str,
        candidate: RetrievalResult,
    ) -> float:
        """
        评估单条候选结果的相关性

        返回: 0.0 ~ 1.0 的相关性得分
        """
        # 方案 A: 基于向量相似度的快速评分（默认）
        if not self.config.retrieval.enable_rerank:
            return candidate.fusion_score

        # 方案 B: LLM 评分（精确但较慢）
        return self._llm_score(query, candidate)

    def _llm_score(self, query: str, candidate: RetrievalResult) -> float:
        """
        使用 LLM 评估相关性

        Prompt 设计要点：
          - 明确评分标准（0-10 分）
          - 要求结构化输出（JSON）
          - 给出评分依据
        """
        from openai import OpenAI
        from config import get_api_key

        api_key = get_api_key()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY 未设置")
        client = OpenAI(
            api_key=api_key,
            base_url=self.config.llm.base_url,
        )

        prompt = f"""请评估以下文档片段对用户查询的相关性。

用户查询: {query}

文档片段: {candidate.content[:500]}

请从以下维度评估（每个维度 0-10 分）：
1. 主题相关性：文档是否涉及查询的核心主题？
2. 信息完整性：文档是否包含回答查询所需的关键信息？
3. 精确度：文档是否直接针对查询（而非泛泛而谈）？
4. 有用性：用户读了这个文档能否解决他们的问题？

请以 JSON 格式输出评分：
```json
{{"theme_relevance": 8, "completeness": 7, "precision": 9, "usefulness": 8, "overall": 8, "reason": "简短的评分理由"}}
```"""

        try:
            response = client.chat.completions.create(
                model=self.config.llm.chat_model,
                messages=[
                    {"role": "system", "content": "你是一个专业的搜索结果评估专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 低温度确保评分一致性
                max_tokens=200,
            )

            result_text = response.choices[0].message.content or "{}"
            # 提取 JSON
            import json
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                overall = scores.get("overall", 5)
                return overall / 10.0  # 归一化到 0-1
            return 0.5

        except Exception as e:
            logger.error(f"[Reranker] LLM 评分失败: {e}")
            return candidate.fusion_score  # 回退到融合得分

    def _cross_encoder_score(
        self,
        query: str,
        candidates: List[RetrievalResult],
    ) -> List[float]:
        """
        Cross-Encoder 批量评分（更快的方案）

        需要: pip install sentence-transformers
        模型: BAAI/bge-reranker-v2-m3（多语言支持）
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("[Reranker] sentence-transformers 未安装，回退到 LLM 评分")
            return [self._llm_score(query, c) for c in candidates]

        if self._cross_encoder is None:
            model_name = "BAAI/bge-reranker-v2-m3"
            logger.info(f"[Reranker] 加载 Cross-Encoder: {model_name}")
            self._cross_encoder = CrossEncoder(model_name)

        pairs = [[query, c.content] for c in candidates]
        scores = self._cross_encoder.predict(pairs)
        # sigmoid 归一化
        scores = 1.0 / (1.0 + np.exp(-np.array(scores)))
        return scores.tolist()


# ============================================================================
# 5. 统一检索服务（Facade 模式）
# ============================================================================
class RetrievalService:
    """
    统一检索服务 — 对外暴露的单一入口

    整合了混合检索 + 重排序的完整流程。
    外部调用者只需使用此类，无需关心底层实现细节。
    """

    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.config = get_config().retrieval

        # 自动构建 BM25 索引
        self._init_bm25_index()

    def _init_bm25_index(self):
        """初始化 BM25 索引"""
        try:
            chunks = BM25Retriever.load_chunks_from_store()
            if chunks:
                self.hybrid_retriever.bm25_retriever.build_index(chunks)
            else:
                logger.info("[RetrievalService] BM25 索引为空（知识库尚未添加文档）")
        except Exception as e:
            logger.warning(f"[RetrievalService] BM25 索引初始化失败: {e}")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        strategy: str = "rrf",
        enable_rerank: Optional[bool] = None,
    ) -> SearchResult:
        """
        统一检索接口

        参数:
          query: 用户查询
          top_k: 最终返回数（默认 5）
          strategy: 融合策略 "rrf" | "weighted" | "vector_only" | "bm25_only"
          enable_rerank: 是否启用重排序（默认使用配置）

        返回:
          SearchResult 包含排好序的 RetrievalResult 列表
        """
        top_k = top_k or self.config.top_k_final
        enable_rerank = enable_rerank if enable_rerank is not None else self.config.enable_rerank

        # Step 1: 混合检索（粗排）
        search_result = self.hybrid_retriever.search(query, top_k=self.config.top_k_fusion, strategy=strategy)

        # Step 2: 重排序（精排）
        if enable_rerank and search_result.results:
            reranked = self.reranker.rerank(query, search_result.results, top_k)
            search_result.results = reranked
            if reranked:
                search_result.top_similarity = reranked[0].final_score

        return search_result

    def rebuild_bm25_index(self):
        """重建 BM25 索引（知识库更新后调用）"""
        chunks = BM25Retriever.load_chunks_from_store()
        self.hybrid_retriever.bm25_retriever.build_index(chunks)


# ============================================================================
# 独立测试入口
# ============================================================================
if __name__ == "__main__":
    """
    快速验证检索层功能：

        python retrieval_layer.py
    """
    logger.info("=" * 60)
    logger.info("DeepService Retrieval Layer — 独立测试")
    logger.info("=" * 60)

    service = RetrievalService()

    test_queries = [
        "如何申请退货？",
        "退款多久到账？",
        "退货运费谁承担？",
    ]

    for query in test_queries:
        logger.info(f"\n--- 查询: {query} ---")
        result = service.search(query, top_k=3)

        if result.results:
            logger.info(f"  检索质量: top_sim={result.top_similarity:.3f}, "
                        f"avg_sim={result.avg_similarity:.3f}")
            for i, r in enumerate(result.results):
                logger.info(f"  [{i + 1}] {r.source_label} "
                            f"score={r.final_score:.3f} | {r.content[:80]}...")
        else:
            logger.info("  无相关结果")

    logger.info("=" * 60)
    logger.info("检索层测试完成 ✓")

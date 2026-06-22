"""
Retrieval Layer — hybrid search combining semantic vector search, BM25 keyword matching,
RRF fusion, and reranking. Vector alone misses exact keywords; BM25 alone misses semantics;
together they complement each other.
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


# /// Retrieval result data structures
@dataclass
class RetrievalResult:
    """Unified retrieval result across all retrieval stages."""
    chunk_id: str
    content: str
    metadata: Dict = field(default_factory=dict)

    # stage scores
    vector_score: float = 0.0       # vector retrieval (normalized)
    bm25_score: float = 0.0         # BM25 retrieval (normalized)
    fusion_score: float = 0.0       # fusion score (RRF or weighted)
    rerank_score: float = 0.0       # rerank score
    final_score: float = 0.0        # final composite score

    source_label: str = ""          # formatted source label, e.g. "[来源: 1]"

    def to_context_string(self, index: int = 1) -> str:
        """Format chunk as context text for prompt injection."""
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
    """Complete search result with quality metadata."""
    query: str
    results: List[RetrievalResult]

    top_similarity: float = 0.0     # highest similarity
    avg_similarity: float = 0.0     # average similarity
    result_count: int = 0           # effective result count

    @property
    def is_reliable(self) -> bool:
        """Whether retrieval results are reliable (for knowledge boundary gating)."""
        config = get_config().retrieval
        return (
            self.top_similarity >= config.vector_similarity_threshold
            and self.result_count >= 1
        )


# /// 1. Semantic vector retriever
class VectorRetriever:
    """Embedding-based semantic vector search. Matches synonyms (e.g., '退钱' vs '退款').
    Limitation: insensitive to proper nouns (product codes, order numbers)."""

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
        """Execute semantic vector search. Returns [(result_dict, similarity_score), ...]."""
        top_k = top_k or self.config.top_k_vector

        # 1. embed query
        query_embedding = self.embedder.embed(query)

        # 2. vector search
        results = self.vector_store.search_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            where_filter=where_filter,
        )

        return [(r, r["similarity"]) for r in results]


# /// 2. BM25 keyword retriever
class BM25Retriever:
    """BM25 keyword search for exact matches (product codes, order numbers). Must be fused with vector search."""

    def __init__(self):
        self.config = get_config().retrieval
        self._corpus: List[str] = []
        self._corpus_metadata: List[Dict] = []
        self._bm25 = None
        self._initialized = False

    def build_index(self, chunks: List[Dict]):
        """Build BM25 index from all chunks. Must be rebuilt after knowledge base updates."""
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
        """Chinese+English hybrid tokenizer. Chinese: character bigrams. English/digits: word-level."""

        tokens = []

        # Chinese: character bigrams
        chinese_chars = re.findall(r"[一-鿿]+", text)
        for segment in chinese_chars:
            for i in range(len(segment) - 1):
                tokens.append(segment[i:i + 2])
            tokens.append(segment[-1])

        # English/digits: word-level
        non_chinese = re.findall(r"[a-zA-Z0-9]+", text)
        tokens.extend([t.lower() for t in non_chinese])

        # standalone numbers
        special = re.findall(r"\d+\.?\d*", text)
        tokens.extend(special)

        return tokens

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        """Execute BM25 keyword search. Returns [(result_dict, bm25_score), ...]."""
        if not self._initialized:
            logger.warning("[BM25Retriever] 索引未初始化，返回空结果")
            return []

        top_k = top_k or self.config.top_k_bm25
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # BM25 search
        scores = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        max_score = scores.max() if scores.max() > 0 else 1.0  # avoid division by zero

        for idx in top_indices:
            if scores[idx] > 0:
                normalized_score = float(scores[idx] / max_score)
                meta = self._corpus_metadata[idx]
                results.append(({
                    "id": meta["id"],
                    "content": meta["content"],
                    "metadata": meta["metadata"],
                    "similarity": normalized_score,  # unified field name
                }, normalized_score))

        return results

    @staticmethod
    def load_chunks_from_store() -> List[Dict]:
        """Load all chunks from ChromaDB for BM25 index building.
        For large knowledge bases (>100k chunks), prefer a database full-text index."""
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


# /// 3. Hybrid fusion (RRF + weighted)
class HybridRetriever:
    """Hybrid retrieval fuser using Reciprocal Rank Fusion (RRF):
    RRF(d) = sum(1 / (k + rank_i(d))), default k=60.
    Also supports weighted linear fusion: alpha * vec_score + beta * bm25_score."""

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
        """Hybrid search main entry. Parallel vector + BM25 search, RRF/weighted fusion, then return SearchResult."""
        top_k = top_k or self.config.top_k_fusion
        logger.info(f"[HybridRetriever] 混合检索: '{query[:50]}...' (strategy={strategy})")

        # Step 1: parallel retrieval
        vector_results = self.vector_retriever.search(query)
        bm25_results = self.bm25_retriever.search(query)

        logger.debug(
            f"[HybridRetriever] 向量检索: {len(vector_results)} 条, "
            f"BM25检索: {len(bm25_results)} 条"
        )

        # Step 2: fusion
        if strategy == "vector_only":
            fused = self._fuse_weighted(vector_results, [], top_k)
        elif strategy == "bm25_only":
            fused = self._fuse_weighted([], bm25_results, top_k)
        elif strategy == "rrf":
            fused = self._fuse_rrf(vector_results, bm25_results, top_k)
        else:  # weighted
            fused = self._fuse_weighted(vector_results, bm25_results, top_k)

        # Step 3: build results
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

        # compute quality metadata
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
        """Reciprocal Rank Fusion. k is the smoothing parameter (default 60)."""

        id_to_data: Dict[str, Dict] = {}
        id_to_ranks: Dict[str, Dict[str, int]] = {}

        # record ranks from each retriever
        for rank, (result, _) in enumerate(vector_results, start=1):
            chunk_id = result["id"]
            id_to_data[chunk_id] = result
            id_to_ranks.setdefault(chunk_id, {})["vector"] = rank

        for rank, (result, _) in enumerate(bm25_results, start=1):
            chunk_id = result["id"]
            if chunk_id not in id_to_data:
                id_to_data[chunk_id] = result
            id_to_ranks.setdefault(chunk_id, {})["bm25"] = rank

        # compute RRF scores
        rrf_scores = {}
        for chunk_id, ranks in id_to_ranks.items():
            score = 0.0
            if "vector" in ranks:
                score += 1.0 / (k + ranks["vector"])
            if "bm25" in ranks:
                score += 1.0 / (k + ranks["bm25"])
            rrf_scores[chunk_id] = score

        # sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:top_k]

        # build result with score info
        result = []
        for chunk_id in top_ids:
            data = id_to_data[chunk_id]
            ranks = id_to_ranks.get(chunk_id, {})

            # reverse-compute normalized scores for display
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
        """Weighted linear fusion: score = alpha * vec_score + beta * bm25_score.
        Weights are configurable for tuning precision vs recall."""

        alpha = self.config.vector_weight
        beta = self.config.bm25_weight

        # normalize scores per retriever
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


# /// 4. Re-ranker
class Reranker:
    """Re-ranker for fine-grained scoring of candidate results. Vector+BM25 is coarse
    recall; re-ranking with LLM (or Cross-Encoder) filters low-relevance noise so
    only high-quality context reaches the generator."""

    def __init__(self):
        self.config = get_config()
        self._cross_encoder = None

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidates: score each, sort by relevance, filter low-scoring, assign source labels."""
        top_k = top_k or self.config.retrieval.top_k_final

        if not candidates:
            return []

        logger.info(f"[Reranker] 重排序 {len(candidates)} 条候选结果")

        # score each candidate
        for candidate in candidates:
            candidate.rerank_score = self._score_candidate(query, candidate)

        # sort by rerank score descending
        reranked = sorted(candidates, key=lambda x: x.rerank_score, reverse=True)

        # filter low-scoring results
        filtered = [
            r for r in reranked
            if r.rerank_score >= self.config.retrieval.rerank_relevance_threshold
        ]

        # take top_k
        final = filtered[:top_k]

        # assign source labels
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
        """Score a single candidate. Uses fusion_score if rerank is disabled, else LLM scoring."""
        # Path A: fast vector-similarity-based scoring (default when rerank disabled)
        if not self.config.retrieval.enable_rerank:
            return candidate.fusion_score

        # Path B: LLM-based scoring (slower but more accurate)
        return self._llm_score(query, candidate)

    def _llm_score(self, query: str, candidate: RetrievalResult) -> float:
        """Use LLM to score query-candidate relevance on 0-10 scale, normalized to 0-1."""
        from config import get_llm_client
        client = get_llm_client()
        if client is None:
            return candidate.fusion_score

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
            # extract JSON
            import json
            json_match = re.search(r"\{[^}]+\}", result_text, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                overall = scores.get("overall", 5)
                return overall / 10.0  # normalize to 0-1
            return 0.5

        except Exception as e:
            logger.error(f"[Reranker] LLM 评分失败: {e}")
            return candidate.fusion_score  # fall back to fusion score


# /// 5. Unified retrieval service (Facade)
class RetrievalService:
    """Unified retrieval service -- single entry point combining hybrid retrieval + reranking.
    External callers use only this class; internal components are hidden."""

    def __init__(self):
        self.hybrid_retriever = HybridRetriever()
        self.reranker = Reranker()
        self.config = get_config().retrieval

        # 自动构建 BM25 索引
        self._init_bm25_index()

    def _init_bm25_index(self):
        """Initialize BM25 index from ChromaDB."""
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
        """Unified retrieval. strategy: "rrf" | "weighted" | "vector_only" | "bm25_only"."""
        top_k = top_k or self.config.top_k_final
        enable_rerank = enable_rerank if enable_rerank is not None else self.config.enable_rerank

        # Step 1: hybrid retrieval (coarse ranking)
        search_result = self.hybrid_retriever.search(query, top_k=self.config.top_k_fusion, strategy=strategy)

        # Step 2: rerank (fine ranking)
        if enable_rerank and search_result.results:
            reranked = self.reranker.rerank(query, search_result.results, top_k)
            search_result.results = reranked
            if reranked:
                search_result.top_similarity = reranked[0].final_score

        return search_result

    def rebuild_bm25_index(self):
        """Rebuild BM25 index after knowledge base updates."""
        chunks = BM25Retriever.load_chunks_from_store()
        self.hybrid_retriever.bm25_retriever.build_index(chunks)


# /// Self-check
if __name__ == "__main__":
    """Quick self-check. Run: python retrieval_layer.py"""
    logger.info("DeepService Retrieval Layer — self-check")

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

    logger.info("Retrieval layer self-check complete.")

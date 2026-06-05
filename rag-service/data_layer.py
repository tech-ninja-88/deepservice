"""
=============================================================================
DeepService RAG — 知识库构建模块 (Data Layer)
=============================================================================
职责：
  1. 文档解析：支持 TXT / Markdown / PDF / Word 多格式
  2. 语义切分：智能分块策略，保证语义完整性
  3. 向量化：文本 → Embedding 向量
  4. 向量存储：ChromaDB 持久化

企业级设计原则：
  - 高质量 Chunking 是消除幻觉的基石
  - 每个 Chunk 保留元数据追溯来源
  - 分段+Embedding 支持批量处理和单条增量

参考：
  [reference:0] — RAG是控制AI幻觉最稳妥的主流技术路径
  [reference:1] — 企业级RAG需形成数据层、检索层、生成层三层体系
=============================================================================
"""

import hashlib
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Generator, Tuple

import numpy as np
from loguru import logger

from config import get_config, ChunkingConfig


# ============================================================================
# 数据结构定义
# ============================================================================
@dataclass
class Document:
    """原始文档"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    content: str = ""
    source_path: str = ""           # 原始文件路径
    source_type: str = ""           # txt / md / pdf / docx
    metadata: Dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class Chunk:
    """
    知识块 — RAG 检索的原子单元

    设计要点：
      - 每个 chunk 独立可检索，同时保留上下文关联
      - metadata 完整追溯原始文档和位置
      - content_hash 用于增量更新（检测变更）
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""                           # 分块文本
    chunk_index: int = 0                        # 在文档中的序号
    embedding: Optional[List[float]] = None     # 向量（待填充）
    content_hash: str = ""                      # 内容哈希（增量索引用）
    metadata: Dict = field(default_factory=dict)  # {title, source_path, page, section, ...}

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict:
        """转换为 ChromaDB 存储格式"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "content_hash": self.content_hash,
            **self.metadata,
        }


# ============================================================================
# 文档解析器 (Strategy Pattern)
# ============================================================================
class BaseDocumentParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """解析文件为 Document 对象"""
        ...

    @staticmethod
    def supports(extension: str) -> bool:
        """判断是否支持该文件扩展名"""
        ...


class TextParser(BaseDocumentParser):
    """纯文本解析器"""

    @staticmethod
    def supports(extension: str) -> bool:
        return extension.lower() in {".txt", ".log", ".csv", ".json", ".xml"}

    def parse(self, file_path: Path) -> Document:
        logger.info(f"[TextParser] 解析文本文件: {file_path.name}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return Document(
            title=file_path.stem,
            content=self._clean_text(content),
            source_path=str(file_path),
            source_type=file_path.suffix.lower(),
            metadata={"encoding": "utf-8", "size_bytes": file_path.stat().st_size},
        )

    def _clean_text(self, text: str) -> str:
        """清洗文本：移除多余空行、统一换行符"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 保留有意义的内容行（移除纯空格行）
        lines = [line for line in text.split("\n") if line.strip()]
        return "\n".join(lines)


class MarkdownParser(BaseDocumentParser):
    """
    Markdown 解析器
    特殊处理：
      - 保留标题层级作为元数据（section 信息）
      - 移除代码块用于向量检索但保留在原文中
    """

    @staticmethod
    def supports(extension: str) -> bool:
        return extension.lower() in {".md", ".markdown"}

    def parse(self, file_path: Path) -> Document:
        logger.info(f"[MarkdownParser] 解析 Markdown: {file_path.name}")
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # 提取标题结构
        sections = self._extract_headings(content)

        return Document(
            title=file_path.stem,
            content=content,
            source_path=str(file_path),
            source_type="md",
            metadata={"sections": sections, "size_bytes": file_path.stat().st_size},
        )

    def _extract_headings(self, content: str) -> List[Dict]:
        """提取 Markdown 标题层级"""
        sections = []
        for line in content.split("\n"):
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                sections.append({"level": level, "title": match.group(2).strip()})
        return sections


class PDFParser(BaseDocumentParser):
    """
    PDF 解析器
    使用 pypdf 提取文本，保留页码信息
    """

    @staticmethod
    def supports(extension: str) -> bool:
        return extension.lower() == ".pdf"

    def parse(self, file_path: Path) -> Document:
        logger.info(f"[PDFParser] 解析 PDF: {file_path.name}")
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("请安装 pypdf: pip install pypdf>=4.0.0")

        reader = PdfReader(str(file_path))
        pages_content = []
        page_metadata = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append(f"[第{i+1}页]\n{text}")
                page_metadata.append({"page": i + 1, "char_count": len(text)})

        content = "\n\n".join(pages_content)
        return Document(
            title=file_path.stem,
            content=content,
            source_path=str(file_path),
            source_type="pdf",
            metadata={
                "total_pages": len(reader.pages),
                "pages_with_content": len(page_metadata),
                "page_info": page_metadata,
            },
        )


class WordParser(BaseDocumentParser):
    """
    Word (.docx) 解析器
    使用 python-docx 提取段落和表格
    """

    @staticmethod
    def supports(extension: str) -> bool:
        return extension.lower() in {".docx", ".doc"}

    def parse(self, file_path: Path) -> Document:
        logger.info(f"[WordParser] 解析 Word: {file_path.name}")
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx>=1.1.0")

        doc = DocxDocument(str(file_path))
        parts = []

        # 提取段落
        for para in doc.paragraphs:
            if para.text.strip():
                style = para.style.name if para.style else "Normal"
                if "Heading" in style or "heading" in style:
                    parts.append(f"## {para.text}")
                else:
                    parts.append(para.text)

        # 提取表格
        for table_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                parts.append(f"\n[表格 {table_idx + 1}]\n" + "\n".join(rows))

        content = "\n\n".join(parts)
        return Document(
            title=file_path.stem,
            content=content,
            source_path=str(file_path),
            source_type="docx",
            metadata={"paragraphs": len(doc.paragraphs), "tables": len(doc.tables)},
        )


# ============================================================================
# 解析器注册表
# ============================================================================
class DocumentParserRegistry:
    """解析器注册表 — 按文件扩展名自动匹配解析器"""

    _parsers: List[BaseDocumentParser] = [
        TextParser(),
        MarkdownParser(),
        PDFParser(),
        WordParser(),
    ]

    @classmethod
    def get_parser(cls, file_path: Path) -> Optional[BaseDocumentParser]:
        extension = file_path.suffix.lower()
        for parser in cls._parsers:
            if parser.supports(extension):
                return parser
        logger.warning(f"[DocumentParserRegistry] 未找到支持 {extension} 的解析器")
        # 兜底：作为纯文本处理
        return TextParser()

    @classmethod
    def register(cls, parser: BaseDocumentParser):
        """注册自定义解析器"""
        cls._parsers.insert(0, parser)  # 新注册的优先
        logger.info(f"[DocumentParserRegistry] 注册解析器: {parser.__class__.__name__}")


# ============================================================================
# 智能语义分块器
# ============================================================================
class SemanticChunker:
    """
    智能语义分块器 — RAG 系统的核心组件

    设计原则：
      1. 段落优先：优先在段落边界切分，不对句子拦腰截断
      2. 语义完整：每个 chunk 包含完整的语义单元
      3. 重叠保留：chunk_overlap 确保上下文连续性
      4. 中文友好：支持中文标点作为分隔符

    分块策略对检索精度的影响：
      - chunk_size 过大 → 检索精度下降，噪音增加
      - chunk_size 过小 → 语义信息不足，召回率降低
      - overlap 过小 → 边界信息丢失
      - overlap 过大 → 冗余信息过多，检索效率降低

    推荐配置（基于实验调优）：
      - chunk_size: 512 tokens（中英文混合场景）
      - chunk_overlap: 64 tokens
      - 分隔符优先级：段落 > 句子 > 子句 > 词
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or get_config().chunking

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        对文档进行智能分块

        策略：
          - Markdown 文档：按标题层级优先切分
          - 普通文档：递归字符分割（RecursiveCharacterTextSplitter 思路）
        """
        logger.info(
            f"[SemanticChunker] 分块文档: {document.title} "
            f"(chunk_size={self.config.chunk_size}, overlap={self.config.chunk_overlap})"
        )

        if document.source_type == "md":
            chunks = self._chunk_markdown(document)
        else:
            chunks = self._chunk_generic(document)

        # 过滤碎片
        chunks = [
            c for c in chunks
            if len(c.content) >= self.config.min_chunk_size
        ]

        # 限制最大分块数
        if len(chunks) > self.config.max_chunks_per_doc:
            logger.warning(
                f"[SemanticChunker] 文档 {document.title} 分块数 {len(chunks)} "
                f"超过上限 {self.config.max_chunks_per_doc}，将被截断"
            )
            chunks = chunks[:self.config.max_chunks_per_doc]

        logger.info(f"[SemanticChunker] 分块完成: {len(chunks)} 个 chunks")
        return chunks

    def _chunk_markdown(self, document: Document) -> List[Chunk]:
        """Markdown 文档按标题 + 段落语义分块"""
        lines = document.content.split("\n")
        chunks = []
        current_section = ""
        current_content: List[str] = []
        char_count = 0
        # 估算：中文约 1.5 字符/token，英文约 4 字符/token
        max_chars = self.config.chunk_size * 2  # 粗略折中

        def flush_chunk():
            nonlocal current_content, char_count
            if not current_content:
                return
            text = "\n".join(current_content)
            chunk = Chunk(
                document_id=document.id,
                content=text,
                chunk_index=len(chunks),
                metadata={
                    "title": document.title,
                    "source_path": document.source_path,
                    "source_type": document.source_type,
                    "section": current_section,
                },
            )
            chunks.append(chunk)
            # 保留 overlap 部分作为下一个 chunk 的起始
            if self.config.chunk_overlap > 0:
                # 取最后约 overlap 大小内容作为前缀
                overlap_chars = self.config.chunk_overlap * 2
                overlap_text = text[-overlap_chars:] if len(text) > overlap_chars else text
                current_content = [overlap_text] if overlap_text else []
                char_count = len(overlap_text)
            else:
                current_content = []
                char_count = 0

        for line in lines:
            # 检测标题行
            if re.match(r"^#{1,6}\s+", line):
                flush_chunk()
                current_section = re.sub(r"^#+\s+", "", line).strip()
                current_content.append(line)
                char_count += len(line) + 1
                continue

            # 空行 → 可能段落边界
            if not line.strip():
                if current_content:
                    current_content.append("")
                    char_count += 1
                continue

            current_content.append(line)
            char_count += len(line) + 1

            # 达到 chunk_size 上限
            if char_count >= max_chars:
                flush_chunk()

        flush_chunk()  # 处理剩余内容
        return chunks

    def _chunk_generic(self, document: Document) -> List[Chunk]:
        """
        通用文档递归分块

        借鉴 LangChain RecursiveCharacterTextSplitter 的实现思路：
          按分隔符优先级依次尝试切分，直到每个块大小在 chunk_size 范围内。
        """
        return self._recursive_split(
            text=document.content,
            document_id=document.id,
            document_title=document.title,
            source_path=document.source_path,
            source_type=document.source_type,
            separators=self.config.separators,
        )

    def _recursive_split(
        self,
        text: str,
        document_id: str,
        document_title: str,
        source_path: str,
        source_type: str,
        separators: List[str],
        chunk_index_base: int = 0,
    ) -> List[Chunk]:
        """递归分割核心算法"""
        max_chars = self.config.chunk_size * 2  # token → 字符粗略转换

        # 基础情况：文本足够小
        if len(text) <= max_chars:
            if not text.strip():
                return []
            return [Chunk(
                document_id=document_id,
                content=text.strip(),
                chunk_index=chunk_index_base,
                metadata={
                    "title": document_title,
                    "source_path": source_path,
                    "source_type": source_type,
                },
            )]

        # 尝试用当前分隔符切分
        separator = separators[0] if separators else "\n"
        next_separators = separators[1:] if len(separators) > 1 else [""]

        if separator:
            splits = text.split(separator)
        else:
            # 最后兜底：按字符强制切分
            splits = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

        chunks: List[Chunk] = []
        current_batch = ""
        current_char_count = 0

        for split in splits:
            split_len = len(split)

            # 当前 split 本身超过上限 → 递归用下一级分隔符
            if split_len > max_chars:
                # 先保存当前批次
                if current_batch.strip():
                    chunks.extend(self._recursive_split(
                        current_batch, document_id, document_title,
                        source_path, source_type,
                        next_separators,
                        chunk_index_base=chunk_index_base + len(chunks),
                    ))
                    current_batch = ""
                    current_char_count = 0
                # 递归处理超长 split
                sub_chunks = self._recursive_split(
                    split, document_id, document_title,
                    source_path, source_type,
                    next_separators,
                    chunk_index_base=chunk_index_base + len(chunks),
                )
                chunks.extend(sub_chunks)
                continue

            # 加入当前批次会超限 → 先保存当前批次
            if current_char_count + split_len > max_chars and current_batch.strip():
                chunks.extend(self._recursive_split(
                    current_batch, document_id, document_title,
                    source_path, source_type,
                    next_separators,
                    chunk_index_base=chunk_index_base + len(chunks),
                ))
                current_batch = split
                current_char_count = split_len
                # 加上分隔符
                if separator and separator != "":
                    current_batch += separator
                    current_char_count += len(separator)
            else:
                if current_batch:
                    current_batch += (separator if separator else "")
                    current_char_count += len(separator) if separator else 0
                current_batch += split
                current_char_count += split_len

        # 处理最后一批
        if current_batch.strip():
            chunks.extend(self._recursive_split(
                current_batch, document_id, document_title,
                source_path, source_type,
                next_separators,
                chunk_index_base=chunk_index_base + len(chunks),
            ))

        # 创建 overlap（在已生成的 chunks 间添加重叠）
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """在相邻 chunks 之间添加重叠内容"""
        overlap_chars = self.config.chunk_overlap * 2
        for i in range(1, len(chunks)):
            prev_content = chunks[i - 1].content
            if len(prev_content) > overlap_chars:
                # 取前一个 chunk 的末尾部分作为当前 chunk 的前缀
                overlap_text = prev_content[-overlap_chars:]
                # 找到第一个完整句子边界
                for sep in ["\n\n", "\n", "。", "！", "？"]:
                    if sep in overlap_text:
                        overlap_text = overlap_text[overlap_text.index(sep) + len(sep):]
                        break
                if overlap_text.strip():
                    chunks[i].content = overlap_text + "\n" + chunks[i].content
        return chunks


# ============================================================================
# Embedding 生成器
# ============================================================================
class EmbeddingGenerator:
    """
    文本向量化模块

    支持多种 Embedding Provider：
      - DeepSeek API（经济型）
      - OpenAI text-embedding-3-small（推荐，1536维）
      - 本地 bge-large-zh-v1.5（离线场景，1024维）

    生产建议：
      - 小规模（<10万条）：OpenAI text-embedding-3-small，成本低效果好
      - 中规模（10-100万条）：DeepSeek API
      - 大规模（>100万条）：本地部署 bge-large-zh-v1.5
    """

    def __init__(self):
        self.config = get_config().llm
        self._embedding_cache: Dict[str, List[float]] = {}
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """获取 Embedding 维度"""
        if self._dimension is None:
            # 生成一个测试 embedding 确定维度
            test_embedding = self.embed("test")
            self._dimension = len(test_embedding)
        return self._dimension

    def embed(self, text: str) -> List[float]:
        """
        文本向量化（单条）

        内置缓存：相同文本不重复调用 API
        """
        cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        logger.debug(f"[EmbeddingGenerator] 向量化文本 ({len(text)} 字符)")

        if self.config.embedding_provider == "openai":
            embedding = self._embed_openai(text)
        elif self.config.embedding_provider == "local":
            embedding = self._embed_local(text)
        else:
            # DeepSeek 作为兜底（通过对话模型间接获取 embedding）
            embedding = self._embed_deepseek(text)

        self._embedding_cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """
        批量文本向量化

        批量处理减少 API 调用次数，降低成本。
        """
        logger.info(f"[EmbeddingGenerator] 批量向量化 {len(texts)} 条文本")

        all_embeddings = []
        uncached = []
        uncached_indices = []

        # 先检查缓存
        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cache_key in self._embedding_cache:
                all_embeddings.append(self._embedding_cache[cache_key])
            else:
                uncached.append(text)
                uncached_indices.append(i)
                all_embeddings.append(None)  # 占位

        if not uncached:
            return all_embeddings

        # 批量调用 API
        for batch_start in range(0, len(uncached), batch_size):
            batch = uncached[batch_start:batch_start + batch_size]
            batch_embeddings = self._embed_batch_api(batch)

            for j, embedding in enumerate(batch_embeddings):
                idx = uncached_indices[batch_start + j]
                all_embeddings[idx] = embedding

                cache_key = hashlib.md5(uncached[batch_start + j].encode("utf-8")).hexdigest()
                self._embedding_cache[cache_key] = embedding

        return all_embeddings

    def _embed_openai(self, text: str) -> List[float]:
        """OpenAI Embedding API"""
        from openai import OpenAI
        client = OpenAI(api_key=self.config.openai_api_key)
        response = client.embeddings.create(
            model=self.config.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _embed_batch_api(self, texts: List[str]) -> List[List[float]]:
        """批量 Embedding API 调用"""
        if self.config.embedding_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            response = client.embeddings.create(
                model=self.config.openai_embedding_model,
                input=texts,
            )
            return [d.embedding for d in response.data]
        else:
            # 非批量模式兜底
            return [self.embed(text) for text in texts]

    def _embed_deepseek(self, text: str) -> List[float]:
        """
        DeepSeek Embedding（暂用 Chat API 的 hidden states 近似）
        注意：DeepSeek 目前未提供专用 Embedding API，此方法为近似实现。
        生产环境建议切换到 OpenAI Embedding 或本地模型。
        """
        logger.warning(
            "[EmbeddingGenerator] DeepSeek 暂未提供专用 Embedding API，"
            "建议将 embedding_provider 设为 'openai' 或 'local'"
        )
        # 兜底：使用简单的哈希特征（仅用于演示，生产不可用）
        # 生产环境请使用 OpenAI 或本地 BGE 模型
        return self._embed_openai(text)  # 自动回退到 OpenAI

    def _embed_local(self, text: str) -> List[float]:
        """
        本地 BGE 模型 Embedding（离线场景）
        需要安装: pip install sentence-transformers
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "本地 Embedding 需要 sentence-transformers。"
                "安装: pip install sentence-transformers>=3.0"
            )

        # 懒加载模型（首次调用时加载）
        if not hasattr(self, "_local_model"):
            model_name = "BAAI/bge-large-zh-v1.5"
            logger.info(f"[EmbeddingGenerator] 加载本地模型: {model_name}")
            self._local_model = SentenceTransformer(model_name)

        # BGE 模型需要在查询前加前缀
        embedding = self._local_model.encode(
            text,
            normalize_embeddings=True,  # L2 归一化，用于余弦相似度
        )
        return embedding.tolist()


# ============================================================================
# 向量数据库管理
# ============================================================================
class VectorStoreManager:
    """
    ChromaDB 向量数据库管理器

    核心功能：
      - 文档批量索引（解析 → 分块 → 向量化 → 存储）
      - 增量索引（检测变更，仅更新变化的 chunk）
      - 文档删除（级联删除所有 chunk）
      - Collection 管理（多知识库隔离）
    """

    def __init__(self, collection_name: str = "deepservice_knowledge"):
        self.collection_name = collection_name
        self.config = get_config()

        import chromadb
        from chromadb.config import Settings

        # 持久化存储
        db_path = str(self.config.app.vector_db_dir)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,     # 不上报数据
                allow_reset=True,               # 允许重置（开发阶段）
            ),
        )

        self.collection = self._get_or_create_collection()
        logger.info(
            f"[VectorStoreManager] 初始化完成: collection={collection_name}, "
            f"chunks={self.collection.count()}"
        )

    def _get_or_create_collection(self):
        """获取或创建 Collection"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            logger.info(f"[VectorStoreManager] 创建新 Collection: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "DeepService 企业智能客服知识库",
                    "created_at": "",
                    "hnsw:space": "cosine",  # 余弦相似度
                },
            )

    def index_document(self, document: Document) -> List[Chunk]:
        """
        索引单个文档（解析 → 分块 → 向量化 → 存储）

        返回: 生成的所有 Chunk
        """
        logger.info(f"[VectorStoreManager] 索引文档: {document.title}")

        # Step 1: 语义分块
        chunker = SemanticChunker()
        chunks = chunker.chunk_document(document)

        if not chunks:
            logger.warning(f"[VectorStoreManager] 文档 {document.title} 未生成有效 chunk")
            return []

        # Step 2: 向量化
        embedder = EmbeddingGenerator()
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_batch(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        # Step 3: 存入 ChromaDB
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[c.to_dict() for c in chunks],
        )

        logger.info(
            f"[VectorStoreManager] 文档 {document.title} 索引完成: "
            f"{len(chunks)} chunks"
        )
        return chunks

    def index_documents(self, documents: List[Document]) -> Dict[str, int]:
        """
        批量索引文档

        返回: {"indexed": N, "failed": N, "total_chunks": N}
        """
        logger.info(f"[VectorStoreManager] 批量索引 {len(documents)} 个文档")
        total_chunks = 0
        indexed = 0
        failed = 0

        for doc in documents:
            try:
                chunks = self.index_document(doc)
                total_chunks += len(chunks)
                indexed += 1
            except Exception as e:
                logger.error(f"[VectorStoreManager] 索引文档 {doc.title} 失败: {e}")
                failed += 1

        return {"indexed": indexed, "failed": failed, "total_chunks": total_chunks}

    def index_file(self, file_path: Path) -> List[Chunk]:
        """
        索引单个文件（自动解析 → 索引）

        使用示例：
            vs = VectorStoreManager()
            chunks = vs.index_file(Path("./knowledge_base/faq.md"))
        """
        # 解析文档
        parser = DocumentParserRegistry.get_parser(file_path)
        if parser is None:
            raise ValueError(f"无法解析文件: {file_path}（不支持的格式）")

        document = parser.parse(file_path)
        return self.index_document(document)

    def index_directory(self, directory: Path, recursive: bool = True) -> Dict[str, int]:
        """
        索引整个目录

        参数：
          directory: 知识库目录路径
          recursive: 是否递归子目录
        """
        logger.info(f"[VectorStoreManager] 索引目录: {directory}")
        documents = self._load_documents_from_dir(directory, recursive)
        return self.index_documents(documents)

    def delete_document(self, document_id: str) -> int:
        """
        删除文档及其所有 chunk

        返回: 删除的 chunk 数量
        """
        # 先查询该文档的所有 chunk
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(
                f"[VectorStoreManager] 删除文档 {document_id}: "
                f"{len(results['ids'])} chunks"
            )
            return len(results["ids"])
        return 0

    def search_by_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where_filter: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        向量语义检索

        参数：
          query_embedding: 查询向量
          top_k: 返回数量
          where_filter: 元数据过滤条件（如 {"source_type": "md"}）
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # 标准化返回格式
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB cosine distance → similarity (1 - distance)
                similarity = 1.0 - distance
                formatted.append({
                    "id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": round(similarity, 4),
                })

        return formatted

    def get_collection_stats(self) -> Dict:
        """获取知识库统计信息"""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
        }

    def reset_collection(self):
        """重置知识库（危险操作，仅开发环境）"""
        logger.warning(f"[VectorStoreManager] 重置 Collection: {self.collection_name}")
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"[VectorStoreManager] 重置失败: {e}")

    def _load_documents_from_dir(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> List[Document]:
        """从目录批量加载文档"""
        documents = []
        pattern = "**/*" if recursive else "*"
        supported_extensions = {".txt", ".md", ".markdown", ".pdf", ".docx"}

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in supported_extensions:
                continue
            # 跳过隐藏文件
            if file_path.name.startswith("."):
                continue

            try:
                parser = DocumentParserRegistry.get_parser(file_path)
                if parser:
                    doc = parser.parse(file_path)
                    documents.append(doc)
            except Exception as e:
                logger.error(f"[VectorStoreManager] 解析 {file_path} 失败: {e}")

        logger.info(f"[VectorStoreManager] 从目录加载了 {len(documents)} 个文档")
        return documents


# ============================================================================
# 独立测试入口
# ============================================================================
if __name__ == "__main__":
    """
    快速验证数据层功能：

        python data_layer.py

    确保在项目目录下创建 knowledge_base/ 目录并放入测试文档。
    """
    logger.info("=" * 60)
    logger.info("DeepService Data Layer — 独立测试")
    logger.info("=" * 60)

    # 1. 测试文档解析
    test_dir = get_config().app.knowledge_dir
    if test_dir.exists() and any(test_dir.iterdir()):
        parser_registry = DocumentParserRegistry
        for f in list(test_dir.glob("*"))[:3]:
            parser = parser_registry.get_parser(f)
            if parser:
                doc = parser.parse(f)
                logger.info(f"解析文档: {doc.title} ({len(doc.content)} 字符)")

    # 2. 测试分块
    test_doc = Document(
        title="测试FAQ",
        content=(
            "# 退换货政策\n\n"
            "## 退换货条件\n\n"
            "1. 自签收之日起7天内，商品未经使用且不影响二次销售，可申请退货。\n"
            "2. 自签收之日起15天内，商品出现质量问题，可申请换货。\n\n"
            "## 退换货流程\n\n"
            "1. 登录账号，进入'我的订单'页面\n"
            "2. 选择需退换货的订单，点击'申请售后'\n"
            "3. 填写退换货原因，上传凭证照片\n"
            "4. 等待客服审核（1-3个工作日）\n"
            "5. 审核通过后，按指引寄回商品\n\n"
            "## 运费说明\n\n"
            "- 质量问题退换货：运费由商家承担\n"
            "- 非质量问题退货：运费由买家承担\n"
            "- 换货运费：商家承担寄回运费\n\n"
            "## 退款时效\n\n"
            "- 审核通过后，退款将在3-7个工作日原路返回\n"
            "- 如超过7个工作日未收到退款，请联系客服\n"
        ),
        source_type="md",
    )

    chunker = SemanticChunker()
    chunks = chunker.chunk_document(test_doc)
    for c in chunks:
        logger.info(f"Chunk {c.chunk_index}: {len(c.content)} 字符 | 来源: {c.metadata.get('section', 'N/A')}")

    logger.info("=" * 60)
    logger.info("数据层测试完成 ✓")

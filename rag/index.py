from typing import List
import pickle
import jieba
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from modelscope import snapshot_download


class IndexBuildModule:
    """文本块向量化 建立索引"""

    def __init__(self, chunks: List[Document], index_save_path: str, bm25_save_path: str):
        self.chunks: List[Document] = chunks  # 文本块
        self.index_save_path = index_save_path  # 向量索引本地存储路径
        self.bm25_save_path = bm25_save_path
        self.vectorstore = None  # 向量数据库
        self.bm25_retriever = None  # bm25检索器
        self.embeddings = None  # embedding模型

    def set_embeddings(self):
        """初始化embedding模型"""
        model_dir = snapshot_download(
            'AI-ModelScope/bge-small-zh-v1.5',
            cache_dir='./models',
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # L2归一化
        )

    @staticmethod
    def chinese_tokenizer(text):
        """使用jieba进行精确模式分词 并过滤掉空格 用于bm25检索器"""
        return [word for word in jieba.lcut(text) if word.strip()]

    def create_index(self):
        """创建索引"""
        if not self.chunks:
            raise ValueError("文档为空")

        contents = [chunk.page_content for chunk in self.chunks]
        metadatas = [chunk.metadata for chunk in self.chunks]

        if not self.embeddings:
            self.set_embeddings()

        self.vectorstore = FAISS.from_texts(
            texts=contents,
            embedding=self.embeddings,
            metadatas=metadatas
        )

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks,
            k=5,
            preprocess_func=self.chinese_tokenizer
        )

        self.save_index()

        return self.vectorstore

    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vectorstore:
            raise ValueError("未构建向量索引")

        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)

        self.vectorstore.save_local(self.index_save_path)

        with open(self.bm25_save_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

    def load_index(self):
        """从配置的路径加载索引"""
        if not (Path(self.index_save_path).exists() and Path(self.bm25_save_path).exists()):
            raise ValueError("本地没有索引配置文件")

        if not self.embeddings:
            self.set_embeddings()

        self.vectorstore = FAISS.load_local(
            self.index_save_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # 用于显式允许加载本地保存的.pkl文件
        )
        with open(self.bm25_save_path, "rb") as f:
            self.bm25_retriever = pickle.load(f)

    def add_documents(self, new_chunks: List[Document]):
        """向现有索引添加新文档"""
        if not self.vectorstore:
            raise ValueError("未构建向量索引")

        self.vectorstore.add_documents(new_chunks)
        self.save_index()

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """相似度搜索"""
        if not self.vectorstore:
            raise ValueError("未构建向量索引")

        docs_content = self.vectorstore.similarity_search(query, k=k)

        return docs_content

    @staticmethod
    def RRF_rerank(vec_docs: List[Document], bm25_docs: List[Document], k: int = 60) -> List[Document]:
        """RRF重排算法 k为RRF参数 用来平滑排名"""
        print("正在执行RRF重排...")
        doc_scores = {}
        doc_objects = {}

        # 计算向量检索结果的RRF分数
        for rank, doc in enumerate(vec_docs):  # rank 检索结果排名
            doc_id = hash(doc.page_content)  # 使用文档内容的哈希作为唯一标识
            doc_objects[doc_id] = doc  # 构建文档id 文档原文的映射

            rrf_score = 1.0 / (k + rank + 1)  # RRF公式
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score  # 获取doc_id对应的值 默认为0

        # 计算BM25检索结果的RRF分数
        for rank, doc in enumerate(bm25_docs):
            doc_id = hash(doc.page_content)
            doc_objects[doc_id] = doc

            rrf_score = (1.0 / (k + rank + 1)) * 0.8  # 配置权重
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score  # 每个文档分数=两个检索器分数相加

        # 按最终RRF分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建最终结果
        reranked_docs = []
        for doc_id, final_score in sorted_docs:
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata['rrf_score'] = final_score
                reranked_docs.append(doc)
                print(f"最终排序 - 文档: {doc.page_content[:50]}... 最终RRF分数: {final_score:.4f}")

        print(
            f"RRF重排完成: 向量检索{len(vec_docs)}个文档, BM25检索{len(bm25_docs)}个文档, 合并后{len(reranked_docs)}个文档")

        return reranked_docs

    def hybrid_similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """混合相似度搜索"""
        if not self.vectorstore:
            raise ValueError("未构建向量索引")

        vec_docs_content = self.vectorstore.similarity_search(query, k=5)
        bm25_docs_content = self.bm25_retriever.invoke(query)
        # print("=============================")
        # print(bm25_docs_content)
        # print(vec_docs_content)
        # print("=============================")

        reranked_docs_content = self.RRF_rerank(vec_docs_content, bm25_docs_content)

        return reranked_docs_content[:k]

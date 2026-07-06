from rag import (
    DataProcessModule,
    IndexBuildModule,
    ReplyModule
)
from pathlib import Path


class RAGSystem:
    """RAG系统"""

    def __init__(self):
        self.data_module = None
        self.index_module = None
        self.reply_module = None
        self.chunks = None
        self.data_path = "data"
        self.index_save_path = "./vector_index"
        self.bm25_save_path = "./bm25_retriever.pkl"

    def rag_initiation(self, if_data_init: bool):
        """RAG系统初始化"""
        if not (Path(self.index_save_path).exists() and Path(self.bm25_save_path).exists()):
            if_data_init = True

        self.data_module = DataProcessModule(self.data_path)
        self.reply_module = ReplyModule()
        if if_data_init:
            self.rag_data_init()
            self.rag_index_init()
        else:
            self.rag_index_load()

    def rag_data_init(self):
        """RAG知识库初始化"""
        self.data_module.load_documents()
        self.chunks = self.data_module.split_documents()

    def rag_index_init(self):
        """RAG索引初始化"""
        self.index_module = IndexBuildModule(
            self.chunks,
            self.index_save_path,
            self.bm25_save_path
        )
        self.index_module.create_index()

    def rag_index_load(self):
        """RAG索引加载"""
        self.index_module = IndexBuildModule(
            [],
            self.index_save_path,
            self.bm25_save_path
        )
        self.index_module.load_index()

    def rag_reply_generation(self, query: str):
        """RAG生成回复"""
        query = query

        docs_content = self.index_module.hybrid_similarity_search(query)
        # print("原始资料：", docs_content)
        docs_content = self.data_module.excel_filter(docs_content)  # 填充表格 去掉表格摘要信息

        print("检索到的资料：", docs_content)
        answer = self.reply_module.reply(query, docs_content)
        print("大模型返回内容：", answer)

        return answer


if __name__ == '__main__':
    print("正在初始化RAG系统...")
    rag = RAGSystem()
    rag.rag_initiation(if_data_init=False)
    print("RAG系统准备就绪！")
    rag.rag_reply_generation("揭阳校区的宿舍环境怎么样")

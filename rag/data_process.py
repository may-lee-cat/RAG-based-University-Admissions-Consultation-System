from pathlib import Path
from typing import List
import pandas as pd
import re
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataProcessModule:
    """数据加载、切片"""

    def __init__(self, data_path: str):
        self.data_path = data_path  # 数据路径
        self.docs: List[Document] = []  # 原始文档
        self.chunks: List[Document] = []  # 文档切片

    def load_documents(self) -> List[Document]:
        """读取文件，用self.docs接收"""
        path = Path(self.data_path)
        print("数据目录为：", path)

        for file_path in path.rglob('*'):
            if not file_path.is_file():
                continue

            print("正在加载：", file_path)

            extension = file_path.suffix
            if extension == ".docx" or extension == ".doc":
                self.docs += UnstructuredWordDocumentLoader(
                    file_path,
                    strategy="fast",  # 不进行OCR
                    languages=["zh"]
                ).load()
            elif extension == ".pdf":
                self.docs += UnstructuredPDFLoader(
                    file_path,
                    strategy="fast",
                    languages=["zh"]
                ).load()
            elif extension == ".xlsx" or extension == ".xls":
                self.docs += self.excel_loader(file_path)
            elif extension == ".txt":
                self.docs += TextLoader(file_path, encoding='utf-8').load()
            else:
                print("暂不支持该格式")
                pass

        if not self.docs:
            raise ValueError("文档为空")

        return self.docs

    def split_documents(self):
        """对self.docs切片，放在self.chunks里"""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "。", "，", " ", ""],
            chunk_size=200,
            chunk_overlap=10,
        )
        self.chunks = text_splitter.split_documents(self.docs)

        return self.chunks

    @staticmethod
    def excel_loader(excel_path: Path) -> List[Document]:
        """处理表格 所有子表的概要全部加进docs"""
        excel_path = str(excel_path)
        if excel_path.split('\\')[-1].startswith('~$'):
            raise BlockingIOError("文件已经被打开，无法读取")

        excel = pd.ExcelFile(excel_path)
        summary_docs = []  # 子表的概要
        match = re.search(r'(\d+)', excel_path.split('/')[-1])  # 提取数字
        if match:
            year = str(match.group(1))
        else:
            year = "某"

        for sheet_name in excel.sheet_names:
            subtable = pd.read_excel(excel, sheet_name=sheet_name)

            # major = '、'.join(subtable.iloc[1:, 2].astype(str).values)
            data = '、'.join(subtable.iloc[0, 3:].astype(str).values)

            # 摘要文档
            # summary_text = f"这个表格的内容是“{year}年揭阳校区在{sheet_name}的录取情况”，专业包括：“{major}”，数据包括：“{data}”。"
            summary_text = f"这个表格的内容是“ {year} 年揭阳校区在 {sheet_name} 的录取情况”，数据包括：“ {data} ”。"
            summary_doc = Document(
                page_content=summary_text,
                metadata={"source": excel_path, "sheet_name": sheet_name}
            )
            summary_docs.append(summary_doc)

        return summary_docs

    @staticmethod
    def excel_filter(docs_content: List[Document]) -> List[Document]:
        """检查得到的资料里面有没有表格 过滤掉摘要信息"""
        excel_text = ""
        source = ""
        sheet_name = ""
        excel_content = []
        new_docs_content = []

        for doc_content in docs_content:
            if 'sheet_name' in doc_content.metadata.keys():
                if not excel_text:
                    source = doc_content.metadata["source"]
                    sheet_name = doc_content.metadata["sheet_name"]
                    xls = pd.ExcelFile(source)
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    excel_text = df.to_string(index=False)
            else:
                new_docs_content.append(doc_content)

        if excel_text:
            excel_content.append(Document(
                page_content=excel_text, metadata={"source": source, "sheet_name": sheet_name}
            ))

        return excel_content + new_docs_content


if __name__ == '__main__':
    data_module = DataProcessModule("../data")
    # data_module.load_documents()
    # data_module.split_documents()
    data_module.excel_loader(Path("../data/分数线/广东工业大学揭阳校区2025年录取情况.xlsx"))

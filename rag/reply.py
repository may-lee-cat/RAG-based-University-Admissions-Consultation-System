import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class ReplyModule:
    """生成回复"""

    def __init__(self):
        self.prompt = ""
        self.llm = None

    def set_llm(self):
        """初始化大语言模型"""
        load_dotenv()

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.7,
            max_tokens=4096,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        return self.llm

    def set_prompt(self):
        """初始化prompt"""
        self.prompt = ChatPromptTemplate.from_template(
            """你是一名高校招生问答机器人。
            请根据下面提供的资料信息来回答用户提出的问题。
            请确保你的回答完全基于这些资料。
            如果资料中没有足够的信息来回答问题，请直接回答用户：“抱歉，我目前无法回答此问题。”
            回答风格：下面提供给你的资料就像是你本来就知道的一样，不要让用户认为你看了资料。回答可以使用markdown格式。

            资料:
            {context}

            用户问题: {question}
            
            回答:"""
        )

        return self.prompt

    def reply(self, query: str, docs_content):
        """生成回复"""
        if not self.llm:
            self.set_llm()
        if not self.prompt:
            self.set_prompt()

        answer = self.llm.invoke(self.prompt.format(question=query, context=docs_content))

        return answer

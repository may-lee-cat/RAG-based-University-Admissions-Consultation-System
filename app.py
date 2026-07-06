import webbrowser
import threading
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="jieba")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from main import RAGSystem
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动后执行
    def _open():
        webbrowser.open("http://127.0.0.1:8000")

    threading.Timer(1.0, _open).start()

    yield


# 初始化FastAPI
app = FastAPI(lifespan=lifespan)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化RAG
print("正在初始化RAG系统...")
rag = RAGSystem()
rag.rag_initiation(if_data_init=False)
print("RAG系统准备就绪！")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="."), name="static")


# 首页返回 index.html
@app.get("/")
def read_index():
    return FileResponse("index.html")


# API 接口
class ChatRequest(BaseModel):
    query: str


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    answer = rag.rag_reply_generation(request.query)

    if hasattr(answer, 'content'):
        reply_text = answer.content
    else:
        reply_text = str(answer)

    return {"answer": reply_text}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

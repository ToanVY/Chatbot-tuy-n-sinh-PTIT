from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from typing import List

import os
import json
import asyncio
import threading
import time
import re

from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────────
# FASTAPI
# ──────────────────────────────────────────────────────────────────

app = FastAPI(title="PTIT Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────

FAISS_PATH = "faiss_db"

# ──────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# ──────────────────────────────────────────────────────────────────

print("⏳ Đang tải mô hình embedding...", flush=True)

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

print("✅ Embedding model sẵn sàng!", flush=True)

# ──────────────────────────────────────────────────────────────────
# GROQ
# ──────────────────────────────────────────────────────────────────

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))

GROQ_MODEL = "llama-3.1-8b-instant"

# ──────────────────────────────────────────────────────────────────
# LOAD FAISS
# ──────────────────────────────────────────────────────────────────

print("⏳ Đang tải FAISS DB...", flush=True)

vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True,
)

print("✅ FAISS DB sẵn sàng!", flush=True)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 10
    }
)

# ──────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []

# ──────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ──────────────────────────────────────────────────────────────────

def tokenize(text):
    text = text.lower()
    return re.findall(r"\w+", text)


def keyword_match_score(question, answer):

    q_words = set(tokenize(question))
    a_words = set(tokenize(answer))

    if not q_words:
        return 0

    overlap = q_words.intersection(a_words)

    return round(len(overlap) / len(q_words), 2)


def context_relevance_score(question, docs):

    q_words = set(tokenize(question))

    if not docs:
        return 0

    context = " ".join(
        doc.page_content
        for doc in docs
    )

    c_words = set(tokenize(context))

    overlap = q_words.intersection(c_words)

    return round(
        len(overlap) / max(len(q_words), 1),
        2
    )


def faithfulness_score(answer, docs):

    if not docs:
        return 0

    context = " ".join(
        doc.page_content
        for doc in docs
    )

    a_words = set(tokenize(answer))
    c_words = set(tokenize(context))

    overlap = a_words.intersection(c_words)

    return round(
        len(overlap) / max(len(a_words), 1),
        2
    )


def answer_relevance_score(question, answer):

    q_words = set(tokenize(question))
    a_words = set(tokenize(answer))

    overlap = q_words.intersection(a_words)

    return round(
        len(overlap) / max(len(q_words), 1),
        2
    )

# ──────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────

def build_prompt(message: str, history: List[Message], docs: list) -> str:

    # Giữ tối đa 10 lượt hỏi đáp
    recent = history[-20:]

    history_text = ""

    for msg in recent:

        role = (
            "Người dùng"
            if msg.role == "user"
            else "Trợ lý"
        )

        history_text += f"{role}: {msg.content}\n"

    # Context retrieval
    knowledge = "\n\n".join(
        doc.page_content[:700]
        for doc in docs[:3]
    )

    history_section = (
        f"Lịch sử trò chuyện:\n{history_text}\n"
        if history_text else ""
    )

    return f"""Bạn là BOT Teasicale, bot tư vấn tuyển sinh thân thiện và nhiệt tình của Học viện Công nghệ Bưu chính Viễn thông (PTIT).

CÁCH TRẢ LỜI:
- Xưng "tôi" (hoặc "Teasicale"), gọi người hỏi là "bạn" — giọng ngang hàng, thân thiện như bạn bè.
- MẶC ĐỊNH NĂM TUYỂN SINH: Nếu người dùng không nói rõ năm, luôn ưu tiên tìm và trả lời bằng thông tin của năm 2026.
- ĐIỂM CHUẨN 2026: Do kỳ thi 2026 chưa diễn ra, CHƯA CÓ ĐIỂM CHUẨN NĂM 2026. Nếu người dùng hỏi điểm chuẩn, điểm trúng tuyển mà không nói năm hoặc hỏi năm 2026, hãy thông báo rõ là "Năm 2026 chưa có điểm chuẩn chính thức", sau đó LẤY ĐIỂM CỦA NĂM 2025 HOẶC 2024 (có trong tài liệu) để họ tham khảo. TUYỆT ĐỐI không lấy điểm năm cũ và gán mác là điểm 2026.
- XỬ LÝ KHI THIẾU THÔNG TIN 2026: Nếu thông tin năm 2026 chưa có (ví dụ chỉ tiêu, học phí...), hãy thông báo rõ ràng: "Hiện tại chưa có thông tin chính thức cho năm 2026", sau đó TỰ ĐỘNG dùng thông tin của năm 2025 (hoặc năm gần nhất) để tư vấn mang tính chất tham khảo.
- Nếu người dùng hỏi "Học viện có những ngành nào?" hoặc các câu hỏi tương tự về danh sách ngành, BẮT BUỘC PHẢI liệt kê một số hoặc tất cả các ngành (cùng mã ngành nếu có) dựa theo thông tin tham khảo. Không được trả lời chung chung.
- Nói chuyện tự nhiên, trực tiếp vào câu hỏi, KHÔNG đề cập đến "tài liệu", "tri thức", "dữ liệu" hay bất kỳ nguồn nào.
- KHÔNG bao giờ mở đầu bằng "Theo tài liệu...", "Dựa trên tài liệu...", "Theo thông tin tôi có..." hay bất kỳ cụm từ tương tự.
- Giọng văn thân thiện, nhiệt tình, tự nhiên — không cứng nhắc hay mang tính báo cáo.
- CHỈ dùng thông tin bên dưới để trả lời. KHÔNG tự bịa ra số liệu hay thông tin.
- Dùng danh sách, **in đậm** để câu trả lời rõ ràng, dễ đọc hơn.
- Nếu không có đủ thông tin của bất kỳ năm nào, hãy nói thẳng: "Cái này tôi chưa có thông tin chính xác. Bạn có thể liên hệ hotline **(024) 33528122** hoặc fanpage PTIT để được hỗ trợ tr
{history_section}

Câu hỏi hiện tại:
{message}

Thông tin tham khảo:
{knowledge}
"""

# ──────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

# ──────────────────────────────────────────────────────────────────
# CHAT API
# ──────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):

    message = request.message
    history = request.history

    async def generate():

        start_time = time.time()

        print("\n" + "=" * 60, flush=True)
        print(f"📩 USER QUESTION: {message}", flush=True)
        print("=" * 60, flush=True)

        # ─────────────────────────────────────────
        # RETRIEVAL
        # ─────────────────────────────────────────

        docs = await asyncio.to_thread(
            retriever.invoke,
            message
        )

        print(
            f"✅ Retrieved {len(docs)} documents",
            flush=True
        )

        # ─────────────────────────────────────────
        # NO DOCS
        # ─────────────────────────────────────────

        if not docs:

            fallback = (
                "Dạ, tôi không tìm thấy thông tin liên quan "
                "trong cơ sở dữ liệu PTIT."
            )

            yield (
                f"data: "
                f"{json.dumps({'type': 'chunk', 'content': fallback})}\n\n"
            )

            yield (
                f"data: "
                f"{json.dumps({'type': 'done'})}\n\n"
            )

            return

        # ─────────────────────────────────────────
        # BUILD PROMPT
        # ─────────────────────────────────────────

        prompt = build_prompt(
            message,
            history,
            docs
        )

        # ─────────────────────────────────────────
        # STREAMING
        # ─────────────────────────────────────────

        q: asyncio.Queue = asyncio.Queue()

        loop = asyncio.get_event_loop()

        full_response = ""

        def run_stream():

            try:

                print(
                    "🤖 Calling Groq API...",
                    flush=True
                )

                completion = llm.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=512,
                    top_p=1,
                    stream=True,
                    stop=None,
                    timeout=30,
                )

                for chunk in completion:

                    content = (
                        chunk
                        .choices[0]
                        .delta
                        .content
                    )

                    if content:

                        loop.call_soon_threadsafe(
                            q.put_nowait,
                            content
                        )

            except Exception as e:

                error_msg = (
                    f"\n\n⚠️ Lỗi kết nối Groq: {str(e)}"
                )

                print(error_msg, flush=True)

                loop.call_soon_threadsafe(
                    q.put_nowait,
                    error_msg
                )

            finally:

                loop.call_soon_threadsafe(
                    q.put_nowait,
                    None
                )

        threading.Thread(
            target=run_stream,
            daemon=True
        ).start()

        # ─────────────────────────────────────────
        # RECEIVE STREAM
        # ─────────────────────────────────────────

        while True:

            content = await q.get()

            if content is None:
                break

            full_response += content

            yield (
                f"data: "
                f"{json.dumps({'type': 'chunk', 'content': content})}\n\n"
            )

        # ─────────────────────────────────────────
        # METRICS
        # ─────────────────────────────────────────

        latency = round(
            time.time() - start_time,
            2
        )

        context_score = (
            context_relevance_score(
                message,
                docs
            )
        )

        faithfulness = (
            faithfulness_score(
                full_response,
                docs
            )
        )

        answer_relevance = (
            answer_relevance_score(
                message,
                full_response
            )
        )

        keyword_score = (
            keyword_match_score(
                message,
                full_response
            )
        )

        print("\n" + "=" * 60, flush=True)
        print(
            "📊 RAG EVALUATION METRICS",
            flush=True
        )
        print("=" * 60, flush=True)

        print(
            f"🕒 Latency: {latency}s",
            flush=True
        )

        print(
            f"📚 Context Relevance: {context_score}",
            flush=True
        )

        print(
            f"✅ Faithfulness: {faithfulness}",
            flush=True
        )

        print(
            f"💬 Answer Relevance: {answer_relevance}",
            flush=True
        )

        print(
            f"🔑 Keyword Match: {keyword_score}",
            flush=True
        )

        print("=" * 60 + "\n", flush=True)

        yield (
            f"data: "
            f"{json.dumps({'type': 'done'})}\n\n"
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

# ──────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok"}

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False
    )
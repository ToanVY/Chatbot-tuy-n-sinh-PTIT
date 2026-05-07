from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from typing import List
import os, json, asyncio, threading
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PTIT Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FAISS_PATH = "faiss_db"

print("⏳ Đang tải mô hình embedding...")
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
print("✅ Embedding model sẵn sàng!")

llm = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "openai/gpt-oss-120b"

vector_store = FAISS.load_local(
    FAISS_PATH,
    embeddings_model,
    allow_dangerous_deserialization=True,
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)


# ─── Models ──────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


# ─── Helpers ─────────────────────────────────────────────────────────────────

def build_prompt(message: str, history: List[Message], docs: list) -> str:
    # Giữ tối đa 10 lượt hỏi đáp (20 messages) để nhớ ngữ cảnh lâu hơn
    recent = history[-20:]
    history_text = ""
    for msg in recent:
        role = "Người dùng" if msg.role == "user" else "Trợ lý"
        history_text += f"{role}: {msg.content}\n"

    # Không gắn nhãn tài liệu để tránh model trích dẫn nguồn
    knowledge = "\n\n".join(doc.page_content[:1200] for doc in docs[:5])

    history_section = f"Lịch sử trò chuyện:\n{history_text}\n" if history_text else ""

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
- Nếu không có đủ thông tin của bất kỳ năm nào, hãy nói thẳng: "Cái này tôi chưa có thông tin chính xác. Bạn có thể liên hệ hotline **(024) 33528122** hoặc fanpage PTIT để được hỗ trợ trực tiếp nhé!"

{history_section}Câu hỏi hiện tại: {message}

Thông tin tham khảo:
{knowledge}"""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    message = request.message
    history = request.history

    async def generate():
        # Retrieve trong thread pool để không block event loop
        docs = await asyncio.to_thread(retriever.invoke, message)

        if not docs:
            fallback = "Dạ, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu. Bạn có thể liên hệ hotline **(024) 33528122** hoặc fanpage PTIT để được hỗ trợ thêm nhé! 😊"
            yield f"data: {json.dumps({'type': 'chunk', 'content': fallback})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        prompt = build_prompt(message, history, docs)

        # Stream LLM (Groq) trong background thread, giao tiếp qua asyncio.Queue
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def run_stream():
            try:
                completion = llm.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.5,
                    max_completion_tokens=65536,
                    top_p=1,
                    reasoning_effort="high",
                    stream=True,
                    stop=None,
                )
                for chunk in completion:
                    content = chunk.choices[0].delta.content
                    if content:
                        loop.call_soon_threadsafe(q.put_nowait, content)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, f"\n\n⚠️ Lỗi kết nối Groq: {e}")
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)  # sentinel

        threading.Thread(target=run_stream, daemon=True).start()

        while True:
            content = await q.get()
            if content is None:
                break
            yield f"data: {json.dumps({'type': 'chunk', 'content': content})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)

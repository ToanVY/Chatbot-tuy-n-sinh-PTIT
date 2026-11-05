from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()

FAISS_PATH = "faiss_db"

# Embedding
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

# LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0.3,
) if GROQ_API_KEY else None

# Vector DB
vector_store = FAISS.load_local(
    folder_path=FAISS_PATH,
    embeddings=embeddings_model,
    allow_dangerous_deserialization=True
)

# Retriever (tăng k để lấy nhiều thông tin hơn cho các câu hỏi liệt kê)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 15}
)

def stream_response(message, history):

    docs = retriever.invoke(message)

    if not docs:
        yield "Mình không thấy thông tin trong cơ sở dữ liệu"
        return

    # chỉ giữ history gần
    history = history[-2:]
    history_text = ""

    for msg in history:
        role = "Người dùng" if msg["role"]=="user" else "Trợ lý"
        history_text += f"{role}: {msg['content']}\n"

    # build knowledge — không gắn nhãn tài liệu để tránh model trích dẫn nguồn
    knowledge = "\n\n".join(doc.page_content[:1000] for doc in docs[:5])

    rag_prompt = f"""
Bạn là BOT Teasicale, bot tư vấn tuyển sinh thân thiện và nhiệt tình của Học viện Công nghệ Bưu chính Viễn thông (PTIT).

CÁCH TRẢ LỜI:
- Xưng "tôi" (hoặc "Teasicale"), gọi người hỏi là "bạn" — giọng ngang hàng, thân thiện như bạn bè.
- MẶC ĐỊNH NĂM TUYỂN SINH: Nếu người dùng không nói rõ năm, luôn ưu tiên tìm và trả lời bằng thông tin của năm 2026.
- ĐIỂM CHUẨN 2026: Do kỳ thi 2026 chưa diễn ra, CHƯA CÓ ĐIỂM CHUẨN NĂM 2026. Nếu người dùng hỏi điểm chuẩn, điểm trúng tuyển mà không nói năm hoặc hỏi năm 2026, hãy thông báo rõ là "Năm 2026 chưa có điểm chuẩn chính thức", sau đó LẤY ĐIỂM CỦA NĂM 2025 HOẶC 2024 (có trong tài liệu) để họ tham khảo. TUYỆT ĐỐI không lấy điểm năm cũ và gán mác là điểm 2026.
- XỬ LÝ KHI THIẾU THÔNG TIN 2026: Nếu thông tin năm 2026 chưa có (ví dụ chỉ tiêu, học phí...), hãy thông báo rõ ràng: "Hiện tại chưa có thông tin chính thức cho năm 2026", sau đó TỰ ĐỘNG dùng thông tin của năm 2025 (hoặc năm gần nhất) để tư vấn mang tính chất tham khảo.
- Nếu người dùng hỏi các ngành học, BẮT BUỘC PHẢI liệt kê một số ngành học hoặc tất cả các ngành mà bạn thấy trong thông tin tham khảo. Đừng chỉ trả lời chung chung.
- Nói chuyện tự nhiên, trực tiếp vào câu hỏi, KHÔNG đề cập đến "tài liệu", "tri thức", "dữ liệu" hay bất kỳ nguồn nào.
- KHÔNG bao giờ mở đầu bằng "Theo tài liệu...", "Dựa trên tài liệu...", "Theo thông tin tôi có..." hay bất kỳ cụm từ tương tự.
- Giọng văn thân thiện, nhiệt tình, tự nhiên — không cứng nhắc hay mang tính báo cáo.
- CHỈ dùng thông tin bên dưới để trả lời. KHÔNG tự bịa ra số liệu hay thông tin.
- Nếu không có đủ thông tin của bất kỳ năm nào, hãy nói thẳng: "Cái này tôi chưa có thông tin chính xác. Bạn có thể liên hệ hotline (024) 33528122 hoặc fanpage PTIT để được hỗ trợ trực tiếp nhé!"

Lịch sử trò chuyện:
{history_text}

Thông tin tham khảo:
{knowledge}

Câu hỏi: {message}
"""

    if not llm:
        yield "⚠️ Groq chưa được cấu hình. Vui lòng điền API key hợp lệ vào file .env và khởi động lại ứng dụng."
        return

    try:
        partial = ""
        for chunk in llm.stream(rag_prompt):
            partial += chunk.content
            yield partial
    except Exception as e:
        yield (
            "⚠️ Dịch vụ Groq đang gặp sự cố hoặc model không hợp lệ với tài khoản của bạn. "
            "Vui lòng kiểm tra API key và chọn một model có quyền truy cập trong Groq Dashboard. "
            f"Chi tiết: {e}"
        )

chatbot = gr.ChatInterface(
    stream_response,
    title="🔴 Chatbot Tuyển sinh - Học viện Công nghệ Bưu chính Viễn thông (PTIT)",
    description="**Giải đáp nhanh chóng quy chế, học phí, và ngành học năm 2026 của PTIT.**",
    textbox=gr.Textbox(
        placeholder="Ví dụ: Học viện có những phương thức xét tuyển nào?",
        container=False,
        scale=7
    ),
    examples=[
        "Học viện có các phương thức xét tuyển nào năm 2026?",
        "Học phí dự kiến ngành Công nghệ thông tin là bao nhiêu?",
        "Chỉ tiêu tuyển sinh của ngành Trí tuệ nhân tạo?"
    ],
    chatbot=gr.Chatbot(
        height=550,
        avatar_images=(None, "static/logo.png")
    )
)

chatbot.launch(share=True)

# Hướng Dẫn Build Chatbot Tuyển Sinh Từ Repo Gốc

Tài liệu này hướng dẫn bạn cách chuyển đổi repository `Chatbot_tuyen_sinh_FPTU_2026` thành một chatbot tuyển sinh mang thương hiệu riêng của trường bạn, tối ưu hóa cấu trúc và hoạt động ổn định nhất.

## 1. Kiến trúc của Repo hiện tại
*   **Dữ liệu (Data Pipeline - `ingest_database.py`)**: Đọc dữ liệu thô (`.txt` và `.pdf`) từ thư mục `data/`, cắt nhỏ văn bản (chunking) với kích thước 800 ký tự (overlap 150), sau đó mã hóa (embedding) và lưu vào Vector Database.
*   **Vector Database**: Sử dụng **ChromaDB** lưu trữ nội dung offline tại thư mục `chroma_db/`.
*   **Mô hình Embedding**: Dùng `BAAI/bge-m3` từ HuggingFace. Đây là model đa ngôn ngữ xử lý tiếng Việt rất tốt.
*   **Mô hình Ngôn ngữ (LLM - `chatbot.py`)**: Đang sử dụng bản LLM miễn phí `deepseek/deepseek-r1-0528:free` thông qua OpenRouter API.
*   **Giao diện (UI)**: Sử dụng **Gradio** (`gr.ChatInterface`) tạo giao diện chat đơn giản, chạy nhanh ở local.

---

## 2. Cần thay đổi những gì để dùng cho trường của bạn?

Để chuyển đổi sang dữ liệu trường bạn, bạn cần sửa đổi 3 yếu tố cốt lõi: **Dữ liệu, Cấu hình định danh, và System Prompt (Câu lệnh hệ thống)**.

1.  **Thay thế toàn bộ dữ liệu**: Xóa hết các file trong thư mục `data/` và bỏ các tài liệu tuyển sinh của trường bạn vào (hỗ trợ file `.txt` và `.pdf`). Xóa luôn thư mục `chroma_db/` hiện tại (vì nó đang chứa database của FPT).
2.  **Đổi tên Collection**: Trong ChromaDB, đổi tên `rag_fptu_2026` thành tên trường bạn (VD: `rag_my_uni_2026`) ở cả 2 file `ingest_database.py` và `chatbot.py`.
3.  **Sửa System Prompt**: Sửa đổi danh tính của bot ở dòng 63 file `chatbot.py`.
4.  **Tùy chỉnh Giao diện**: Chỉnh sửa các câu Text/Placeholder trên giao diện Gradio cho phù hợp.

---

## 3. Hướng dẫn từng bước thực hiện chi tiết

### Bước 1: Chuẩn bị "Dữ liệu Sạch" (Quan trọng nhất)
Chất lượng của RAG phụ thuộc 80% vào dữ liệu.
*   **Làm sạch dữ liệu**: Khuyến khích chuyển đổi dữ liệu sang định dạng `.txt` thay vì dùng `.pdf`. Dùng file PDF nguyên bản thường hay bị lỗi font hoặc đọc nhầm bảng biểu.
*   **Chia file logic**: Giống như repo mẫu, hãy chia nhỏ theo từng chủ đề để bot dễ tìm kiếm: `Hoc_phi.txt`, `Hoc_bong.txt`, `Phuong_thuc_xet_tuyen.txt`, `FAQ.txt`...
*   **Format FAQ**: Nếu có các câu hỏi thường gặp, hãy định dạng thật rõ ràng: `Câu hỏi: ... \n Trả lời: ...`.

### Bước 2: Thay đổi Code gốc

**1. Trong file `ingest_database.py`**:
*   Sửa dòng 59 (thuộc tính `collection_name`): Thay đổi thành `collection_name="rag_ten_truong_ban_2026"`
*   *(Tùy chọn nâng cao)*: Cân nhắc chỉnh `chunk_size=1000` và `chunk_overlap=200` nếu tài liệu trường bạn có các đoạn văn dài, giúp tránh bị cắt cụt ý nghĩa.

**2. Trong file `chatbot.py`**:
*   Sửa dòng 27 (thuộc tính `collection_name`): Thay đổi thành `collection_name="rag_ten_truong_ban_2026"`
*   **Cập nhật Prompt (dòng 62-76)**: Nâng cấp prompt để bot phản hồi thông minh, lịch sự và không bịa đặt thông tin:
```python
    rag_prompt = f"""
Bạn là chuyên viên tư vấn tuyển sinh nhiệt tình và chuyên nghiệp của Trường Đại học [Tên trường bạn].
Nhiệm vụ của bạn là giải đáp thắc mắc của học sinh/phụ huynh dựa trên các thông tin được cung cấp dưới đây.

LƯU Ý QUAN TRỌNG:
- CHỈ trả lời dựa trên phần "Tri thức" được cung cấp. KHÔNG tự bịa ra thông tin.
- Nếu thông tin không có trong Tri thức, hãy nói lịch sự: "Dạ, hiện tại em chưa có thông tin chính xác về vấn đề này. Anh/chị có thể liên hệ trực tiếp hotline hoặc fanpage của trường để được hỗ trợ ạ."
- Trả lời thân thiện, lịch sự, xưng "em" hoặc "mình" và gọi "bạn" hoặc "anh/chị".

Câu hỏi: {message}

Lịch sử trò chuyện:
{history_text}

Tri thức:
{knowledge}
"""
```

*   **Sửa phần giao diện ở cuối file**:
```python
chatbot = gr.ChatInterface(
    stream_response,
    title="Chatbot Tuyển sinh - Đại học [Tên trường bạn]",
    description="Hỏi bất kỳ điều gì về quy chế, học phí, ngành học năm 2026",
    textbox=gr.Textbox(
        placeholder="Ví dụ: Trường có những phương thức xét tuyển nào?",
        container=False,
        scale=7
    ),
)
```

### Bước 3: Nâng cấp Model LLM (Khuyên dùng cho Thực tế)
Project hiện tại đang dùng `deepseek-r1-0528:free`. Các model miễn phí qua OpenRouter thường bị rate-limit (phản hồi chậm) hoặc dễ quá tải vào giờ cao điểm.
*   **Giải pháp**: Nên đăng ký API key của **Google Gemini** (Gemini 1.5 Flash rất rẻ và siêu nhanh) hoặc sử dụng **GPT-4o-mini** của OpenAI.
*   Ví dụ chuyển sang dùng Google Gemini trong `chatbot.py`:
```python
# Cài đặt thư viện: pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GEMINI_API_KEY"), 
    temperature=0.3
)
```

### Bước 4: Chạy và Đánh giá (Testing)
1. Xóa folder `chroma_db` (nếu đã chạy thử và có sẵn folder này).
2. Chạy file khởi tạo database: `python ingest_database.py` để tạo Vector Database mới từ dữ liệu trường bạn.
3. Chạy file bot: `python chatbot.py` và mở trình duyệt để thử nghiệm.
4. **Các tình huống cần Test**: Hỏi những câu khó, câu hỏi mập mờ, và cố tình hỏi thông tin ngoài luồng (VD: "Trường bạn có dạy nấu ăn không?") để xem bot có biết từ chối và bám sát system prompt hay không.

### Bước 5: Triển khai (Deployment)
Hiện tại `share=True` trong Gradio chỉ tạo một link tạm thời sống trong 72 giờ và phụ thuộc vào máy tính của bạn phải luôn bật.
Để project hoạt động chuyên nghiệp và online 24/7:
*   Bạn có thể tải project lên **Hugging Face Spaces** (chọn Gradio Space), hệ thống sẽ host miễn phí và public project của bạn 24/7.
*   Hoặc nếu muốn nhúng (embed) vào website của trường, bạn cần thay thế Gradio bằng **FastAPI** làm backend và xây dựng phần giao diện (UI) bằng HTML/JS/React.

"""
ingest_database.py – Tối ưu embedding theo từng loại file:
  - TXT  : Xử lý sạch header/nav thừa, chunk theo đoạn tự nhiên.
  - PDF  : Dùng Docling chuyển thành Markdown có cấu trúc (bảng, tiêu đề, đầu mục...).
  - JSON : Duyệt cấu trúc, tạo câu văn tự nhiên cho từng mục → 1 ngành = 1 chunk.
"""

import os
import re
import json
import glob
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Docling – chuyển tài liệu phi cấu trúc → Markdown có cấu trúc
try:
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    from langchain_community.document_loaders import PyMuPDFLoader
    print("  Docling chưa cài, sử dụng PyMuPDF làm fallback. Chạy: pip install langchain-docling docling")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

DATA_PATH  = r"data"
FAISS_PATH = r"faiss_db"

# ─── 0. Khởi tạo model embedding ─────────────────────────────────────────────
print("⏳ Đang tải mô hình embedding BAAI/bge-m3...")
embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
print("✅ Embedding model sẵn sàng!\n")


# ─── 1. TXT – dọn nav thừa, chunk theo đoạn ─────────────────────────────────
NAV_PATTERNS = [
    r"Trang chủ\n.*?Nộp hồ sơ trực tuyến\n",   # nav bar dài
    r"Menu\n.*?Nộp hồ sơ trực tuyến\n",
]

def clean_txt(text: str) -> str:
    for pat in NAV_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    # Xóa nhiều dòng trắng liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_txt_docs(data_path: str) -> list[Document]:
    txt_files = glob.glob(os.path.join(data_path, "**", "*.txt"), recursive=True)
    docs = []
    for fp in txt_files:
        try:
            loader = TextLoader(fp, encoding="utf-8", autodetect_encoding=True)
            raw = loader.load()
            for doc in raw:
                doc.page_content = clean_txt(doc.page_content)
                if len(doc.page_content) > 50:
                    docs.append(doc)
        except Exception as e:
            print(f"  ⚠️  Lỗi load TXT {fp}: {e}")
    print(f"📄 TXT: load {len(docs)} file")
    return docs


# ─── 2. PDF – Docling → Markdown có cấu trúc (bảng, heading giữ nguyên) ──────
def load_pdf_docs(data_path: str) -> list[Document]:
    pdf_files = glob.glob(os.path.join(data_path, "**", "*.pdf"), recursive=True)
    docs = []

    if DOCLING_AVAILABLE:
        # ── Docling: chuyển PDF → Markdown giữ bảng + heading ──
        for fp in pdf_files:
            try:
                loader = DoclingLoader(file_path=fp, export_type=ExportType.MARKDOWN)
                raw = loader.load()
                for doc in raw:
                    text = doc.page_content.strip()
                    # Bỏ chunk quá ngắn hoặc chỉ có nav
                    if len(text) > 150:
                        doc.metadata["type"] = "pdf_docling"
                        docs.append(doc)
            except Exception as e:
                print(f"   Lỗi Docling PDF {fp}: {e}")
        print(f"📕 PDF (Docling → Markdown): load {len(docs)} chunk")
    else:
        # ── Fallback: PyMuPDF từng trang ──
        for fp in pdf_files:
            try:
                loader = PyMuPDFLoader(fp)
                pages = loader.load()
                for page in pages:
                    text = page.page_content.strip()
                    if len(text) > 100:
                        page.page_content = text
                        docs.append(page)
            except Exception as e:
                print(f"  ⚠️  Lỗi PyMuPDF {fp}: {e}")
        print(f"📕 PDF (PyMuPDF fallback): load {len(docs)} trang")

    return docs


# ─── 3. JSON – render câu văn tự nhiên, 1 mục = 1 chunk ─────────────────────
def render_bang(bang: dict, context_prefix: str = "") -> str:
    """
    Chuyển một row của bảng (ngành học) thành câu văn tự nhiên.
    """
    ten = bang.get("Tên ngành, chương trình") or bang.get("Tên chương trình/ngành xét tuyển", "")
    ma_xet = bang.get("Mã xét tuyển", "")
    ma_nganh = bang.get("Mã ngành, nhóm ngành") or bang.get("Mã ngành", "")
    chi_tieu = bang.get("Chỉ tiêu dự kiến") or bang.get("Số lượng dự kiến", "")
    to_hop = bang.get("Tổ hợp xét tuyển", "")
    phuong_thuc = bang.get("Phương thức tuyển sinh", "")
    # Điểm chuẩn các năm
    diem_parts = []
    for key in bang:
        if "Điểm" in key or "điểm" in key:
            val = bang[key]
            if val:
                diem_parts.append(f"{key}: {val}")

    parts = []
    if context_prefix:
        parts.append(context_prefix)
    if ten:
        parts.append(f"Ngành/Chương trình: {ten}")
    if ma_xet:
        parts.append(f"Mã xét tuyển: {ma_xet}")
    if ma_nganh and ma_nganh != ma_xet:
        parts.append(f"Mã ngành: {ma_nganh}")
    if chi_tieu:
        parts.append(f"Chỉ tiêu dự kiến: {chi_tieu}")
    if to_hop:
        to_hop_clean = re.sub(r"\r\n|\r|\n", ", ", to_hop).strip(", ")
        parts.append(f"Tổ hợp xét tuyển: {to_hop_clean}")
    if phuong_thuc:
        pm_clean = re.sub(r"\r\n|\r|\n", " ", phuong_thuc).strip()
        parts.append(f"Phương thức tuyển sinh: {pm_clean}")
    if diem_parts:
        parts.append("Điểm trúng tuyển – " + "; ".join(diem_parts))
    return ". ".join(parts) + "."


def json_to_docs(data_path: str) -> list[Document]:
    """
    Duyệt toàn bộ file JSON trong data_path:
      - Các key "noi_dung" → chunk theo đoạn văn (text splitter sau)
      - Các list "bang" → mỗi row thành 1 Document riêng
    """
    json_files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
    text_docs: list[Document] = []   # sẽ chunk tiếp
    atom_docs: list[Document] = []   # 1 row = 1 chunk, không cắt thêm

    for fp in json_files:
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ⚠️  Lỗi load JSON {fp}: {e}")
            continue

        source_name = Path(fp).name

        # Hỗ trợ 2 kiểu cấu trúc JSON:
        # A) {"key": {"noi_dung": "...", "bang": [...]}} (ptit_tuyen_sinh_day_du.json)
        # B) list flat [{"question": ..., "answer": ...}]  (ptit_chatbot_data.json)

        if isinstance(data, list):
            # Kiểu B – Q&A flat list
            for item in data:
                q = item.get("question") or item.get("câu_hỏi") or ""
                a = item.get("answer") or item.get("trả_lời") or ""
                if q or a:
                    text = f"Câu hỏi: {q}\nCâu trả lời: {a}".strip()
                    if len(text) > 20:
                        atom_docs.append(Document(
                            page_content=text,
                            metadata={"source": source_name, "type": "qa"}
                        ))

        elif isinstance(data, dict):
            # Kiểu A – dict lồng nhau với noi_dung + bang
            def traverse(node, prefix=""):
                if isinstance(node, dict):
                    noi_dung = node.get("noi_dung", "")
                    url      = node.get("url", "")
                    bangs    = node.get("bang", [])

                    # Thêm nội dung text vào pool text_docs
                    if noi_dung and len(noi_dung.strip()) > 80:
                        meta = {"source": source_name, "type": "text"}
                        if url:
                            meta["url"] = url
                        text_docs.append(Document(
                            page_content=noi_dung.strip(),
                            metadata=meta
                        ))

                    # Xử lý bảng – mỗi row là 1 Document atom
                    if isinstance(bangs, list):
                        for table in bangs:
                            # table = {"headers": [...], "rows": [...]}
                            ctx_rows = table.get("rows", []) if isinstance(table, dict) else []
                            # Tìm context (section name) từ row header
                            current_section = ""
                            for row in ctx_rows:
                                if not isinstance(row, dict):
                                    continue
                                # Row tiêu đề section (VD: TT="I", chỉ có 1 key ngoài TT)
                                if len(row) == 2 and "TT" in row:
                                    current_section = list(row.values())[-1]
                                    continue
                                rendered = render_bang(row, context_prefix=current_section)
                                if len(rendered) > 30:
                                    meta = {"source": source_name, "type": "table"}
                                    if url:
                                        meta["url"] = url
                                    atom_docs.append(Document(
                                        page_content=rendered,
                                        metadata=meta
                                    ))

                    # Đệ quy vào các sub-key còn lại
                    for key, val in node.items():
                        if key not in ("noi_dung", "url", "bang"):
                            traverse(val, prefix=key)

                elif isinstance(node, list):
                    for item in node:
                        traverse(item, prefix=prefix)

            traverse(data)

    print(f"📦 JSON: {len(text_docs)} khối text + {len(atom_docs)} dòng bảng/QA nguyên tử")
    return text_docs, atom_docs


# ─── 4. Chunking ─────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
    separators=["\n\n", "\n", ".", " "],
)

# Load tất cả
txt_docs              = load_txt_docs(DATA_PATH)
pdf_docs              = load_pdf_docs(DATA_PATH)
json_text_docs, json_atom_docs = json_to_docs(DATA_PATH)

# Chunk các nguồn dùng text splitter
chunked_txt  = text_splitter.split_documents(txt_docs)
chunked_pdf  = text_splitter.split_documents(pdf_docs)
chunked_json_text = text_splitter.split_documents(json_text_docs)

# Atom docs (bảng/QA) giữ nguyên – không cắt thêm
all_chunks_raw = chunked_txt + chunked_pdf + chunked_json_text + json_atom_docs

# Lọc chunk rỗng / quá ngắn
all_chunks = []
for chunk in all_chunks_raw:
    text = chunk.page_content.strip()
    if len(text) > 20:
        chunk.page_content = text
        all_chunks.append(chunk)

print(f"\n📊 Tổng kết:")
print(f"   TXT chunks   : {len(chunked_txt)}")
print(f"   PDF chunks   : {len(chunked_pdf)}")
print(f"   JSON text    : {len(chunked_json_text)}")
print(f"   JSON atoms   : {len(json_atom_docs)}")
print(f"   ────────────────────────")
print(f"   TỔNG hợp lệ : {len(all_chunks)}\n")


# ─── 5. Lưu vào FAISS ────────────────────────────────────────────────────────
print("⏳ Đang embed và lưu vào FAISS (có thể mất vài phút)...")

vector_store = FAISS.from_documents(all_chunks, embeddings_model)
vector_store.save_local(FAISS_PATH)

print(f"\n🎉 Hoàn thành! Tổng {len(all_chunks)} chunks đã được lưu vào FAISS tại '{FAISS_PATH}/'.")
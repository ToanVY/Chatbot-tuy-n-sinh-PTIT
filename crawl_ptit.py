import asyncio
import os
import re
from crawl4ai import AsyncWebCrawler

# Hàm dọn dẹp tên file (xóa các ký tự cấm trong Windows như / \ : * ? " < > |)
def clean_filename(title):
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
    return safe_title.strip()

async def main():
    # Để có nhiều file, bạn phải liệt kê chi tiết các bài viết tại đây
    urls_to_crawl = [
        "https://tuyensinh.ptit.edu.vn/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/tong-quan-hoc-vien/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/ba-cong-khai/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/chinh-sach-hoc-bong/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/cau-hoi-thuong-gap/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-nam-2025/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2024/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2023/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2022/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2021/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/xem-diem-cac-nam-truoc/diem-trung-tuyen-2020/",
        "https://tuyensinh.ptit.edu.vn/tin-tuc/thong-tin-bao-chi/",
        "https://tuyensinh.ptit.edu.vn/tin-tuc/hoat-dong-su-kien/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2025/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh/de-an-2024/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2023/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2022/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2021-3/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-nam-2020/",
        "https://tuyensinh.ptit.edu.vn/de-an-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2019/",
        "https://tracuu.ptit.edu.vn/",
        "https://xettuyen.ptit.edu.vn/"
        "https://tuyensinh.ptit.edu.vn/phuong-thuc-xet-tuyen-dua-vao-ket-qua-thi-thpt/",
        "https://tuyensinh.ptit.edu.vn/xet-tuyen-ket-hop-chung-chi-quoc-te/",
        "https://tuyensinh.ptit.edu.vn/chi-tieu-nganh-dao-tao/",
        "https://tuyensinh.ptit.edu.vn/brochure-ptit-2025-2/",
        "https://tuyensinh.ptit.edu.vn/huong-dan-dkxt/",
        "https://tuyensinh.ptit.edu.vn/thong-bao-xet-tuyen-thang-va-uu-tien-xet-tuyen-vao-dai-hoc-he-chinh-quy-nam-2026/",
        "https://tuyensinh.ptit.edu.vn/thong-tin-tuyen-sinh-dai-hoc-chinh-quy/",
        "https://tuyensinh.ptit.edu.vn/thong-baophuong-thuc-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026/",
        "https://tuyensinh.ptit.edu.vn/thong-bao-thong-tin-chi-tiet-cac-phuong-thuc-tuyen-sinh-dai-hoc-he-chinh-quy-nam-2026/",
        "https://tuyensinh.ptit.edu.vn/gioi-thieu/chinh-sach-hoc-bong/"
    ]
    
    os.makedirs("data/ptit_knowledge", exist_ok=True)
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        for url in urls_to_crawl:
            print(f"Đang crawl: {url}")
            result = await crawler.arun(url=url)
            
            if result.success:
                # 1. Cố gắng lấy tiêu đề thật của trang web từ metadata
                page_title = result.metadata.get("title")
                if not page_title:
                    page_title = url.strip("/").split("/")[-1] 
                
                # 2. Làm sạch tiêu đề để làm tên file
                safe_name = clean_filename(page_title)
                filename = f"data/ptit_knowledge/{safe_name}.txt"
                
                # 3. Lưu file
                with open(filename, "w", encoding="utf-8") as f:
                    # Ghi thêm Metadata ở dòng đầu để sau này RAG có context
                    f.write(f"Nguồn: {url}\n")
                    f.write(f"Tiêu đề: {page_title}\n")
                    f.write("-" * 50 + "\n\n")
                    
                    # Ghi nội dung cốt lõi
                    f.write(result.markdown_v2.fit_markdown if hasattr(result, 'markdown_v2') else result.markdown)
                    
                print(f"-> Đã lưu thành công: {filename}\n")
            else:
                print(f"-> Thất bại khi crawl {url}: {result.error_message}\n")

if __name__ == "__main__":
    asyncio.run(main())

# import paddle
# import paddleocr
# import cv2
# import fitz
# import numpy as np

# paddle.utils.run_check()
# print(f"✅ PaddlePaddle版本: {paddle.__version__}")

# print(f"✅ PaddleOCR版本: {paddleocr.__version__}")
# ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)  # 2.x 版本支持 use_gpu
# print("✅ PaddleOCR初始化成功（GPU模式）")

# print(f"✅ OpenCV版本: {cv2.__version__}")
# print(f"✅ NumPy版本: {np.__version__}")
# print(f"✅ PyMuPDF版本: {fitz.__version__}")

import os
import fitz  # PyMuPDF
import paddleocr
import numpy as np
from tqdm import tqdm

# 初始化 OCR（GPU + 角度分类）
ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True)
print("✅ PaddleOCR 初始化完成")

# 定义路径
base_dir = r"G:\Agent\customer-service-ai-agent\Smart_City"
output_dir = base_dir

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 处理 1.pdf 到 9.pdf
for i in range(1, 10):
    pdf_path = os.path.join(base_dir, f"{i}.pdf")
    if not os.path.exists(pdf_path):
        print(f"⚠️ 文件不存在: {pdf_path}，跳过")
        continue

    print(f"\n📄 正在处理: {pdf_path}")
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    full_text = []

    # 逐页处理
    for page_num in tqdm(range(total_pages), desc=f"处理 {i}.pdf"):
        page = doc.load_page(page_num)

        # 强制转成 RGB，避免通道数不稳定
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), colorspace=fitz.csRGB, alpha=False)

        # 使用原始像素数据，而不是 pix.tobytes()
        img = np.frombuffer(pix.samples, dtype=np.uint8)
        img = img.reshape(pix.height, pix.width, pix.n)

        # OCR 识别
        result = ocr.ocr(img, cls=True)

        # 提取文本
        if result and result[0]:
            page_text = "\n".join([line[1][0] for line in result[0]])
            full_text.append(page_text)
        else:
            full_text.append("")

    doc.close()

    # 合并全文
    final_text = "\n\n".join(full_text)

    # 保存文本文件：1.pdf -> 11.txt
    output_filename = f"{i + 10}.txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    print(f"✅ 已保存到: {output_path}，共 {total_pages} 页，识别字符数: {len(final_text)}")

print("\n🎉 所有 PDF 处理完成！")

#python ocr.py
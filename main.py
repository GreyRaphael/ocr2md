import base64
import time
import argparse
import logging
from pathlib import Path
from typing import List, Union
from functools import lru_cache

import cv2
import niquests
import numpy as np
from rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput

@lru_cache(maxsize=1)
def get_layout_engine():
    return RapidLayout(
        cfg=RapidLayoutInput(
            model_type=ModelType.PP_DOC_LAYOUTV3,
            engine_type=EngineType.ONNXRUNTIME,
            conf_thresh=0.35,
            iou_thresh=0.35,
        )
    )

PROMPT_MAPPING = {
    "text": "OCR:",
    "content": "OCR:",
    "doc_title": "OCR:",
    "paragraph_title": "OCR:",
    "table": "Convert this table into a Markdown table:",
    "display_formula": "Formula Recognition:",
    "inline_formula": "Formula Recognition:",
    "seal": "Seal Recognition:",
}


def crop_to_base64(img_cv, bbox: list) -> str:
    """根据边界框裁剪图片区域，并转换为 base64 格式，供给 VLM 识别"""
    x1, y1, x2, y2 = map(int, bbox)
    cropped = img_cv[y1:y2, x1:x2]
    _, buffer = cv2.imencode(".png", cropped)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def save_crop_image(img_cv, bbox: list, save_path: Path):
    """直接裁剪图片区域并保存到本地"""
    x1, y1, x2, y2 = map(int, bbox)
    cropped = img_cv[y1:y2, x1:x2]
    cv2.imwrite(str(save_path), cropped)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(log_file.stem)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _process_single_image(img_input: Union[str, np.ndarray], image_stem: str, output_dir: Path, server_url: str, model_name: str, ignore_labels: List[str], layout_engine, logger: logging.Logger):
    imgs_dir = output_dir / "imgs"
    if isinstance(img_input, str):
        img_cv = cv2.imread(img_input)
        if img_cv is None:
            logger.error(f"读取图片失败: {img_input}")
            return
    else:
        img_cv = img_input

    results = layout_engine(img_cv)
    if results.boxes is None:
        logger.warning(f"[{image_stem}] 未检测到任何版面元素")
        return

    items_to_process = []
    chart_counter = 0

    # 启用 multiplexed=True 实现请求并发优化
    with niquests.Session(multiplexed=True) as client:
        client.trust_env = False
        for box, label, _ in zip(results.boxes, results.class_names, results.scores):
            if label in ignore_labels:
                continue

            # === 图表拦截与保存 ===
            if label in ["chart", "figure"]:
                chart_counter += 1
                img_name = f"{image_stem}_{label}_{chart_counter}.png"
                img_save_path = imgs_dir / img_name

                save_crop_image(img_cv, box, img_save_path)
                items_to_process.append(("chart", f"![{label}](imgs/{img_name})"))
                logger.info(f"已成功提取图表并保存至: {img_save_path}")
                continue

            # === 文本/表格/公式调用 VLM 识别 ===
            prompt = PROMPT_MAPPING.get(label, "OCR:")
            base64_image = crop_to_base64(img_cv, box)

            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    }
                ],
                "temperature": 0.0,
                "top_k": 1,
            }

            # 提交请求，因为启用了 multiplexed=True，这里不会阻塞，会立即返回一个 Lazy Response
            response = client.post(f"{server_url}/chat/completions", json=payload, timeout=60.0)
            items_to_process.append(("vlm", label, response))

        # 统一等待所有请求完成
        client.gather()

    # === 按原顺序处理响应并生成 Markdown ===
    markdown_chunks = []
    for item in items_to_process:
        if item[0] == "chart":
            markdown_chunks.append(item[1])
        elif item[0] == "vlm":
            _, label, response = item
            try:
                if not response.ok:
                    logger.error(f"Server Error: {response.text}")
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                if not content:
                    continue

                if label in ["doc_title", "paragraph_title"]:
                    markdown_chunks.append(f"## {content}")
                else:
                    markdown_chunks.append(content)
            except Exception as e:
                logger.error(f"处理区块 [{label}] 失败: {e}")

    output_file = output_dir / f"{image_stem}.md"
    final_markdown = "\n\n".join(markdown_chunks)
    output_file.write_text(final_markdown, encoding="utf-8")
    logger.info(f"[{image_stem}] 处理完成, 结果已保存至 {output_file}")


def process_document(input_path: str, server_url: str, model_name: str, ignore_labels: List[str]):
    start = time.time()
    path_obj = Path(input_path)
    base_stem = path_obj.stem

    output_dir = Path(f"output_{base_stem}")
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"{base_stem}.log"
    logger = setup_logger(log_file)

    layout_engine = get_layout_engine()

    image_inputs_to_process = []
    if path_obj.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError:
            logger.error("处理 PDF 需要安装 pymupdf: uv add pymupdf")
            return
        logger.info("正在将 PDF 转换为内存图片...")
        try:
            doc = fitz.open(input_path)
            for i in range(len(doc)):
                page = doc[i]
                # 强制 alpha=False 丢弃透明通道，保持默认大小以提升处理速度，并在内存中处理
                pix = page.get_pixmap(alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 3:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                
                image_inputs_to_process.append((img_cv, f"{base_stem}_page{i+1}"))
        except Exception as e:
            logger.error(f"打开或处理 PDF 失败: {e}")
            return
    else:
        image_inputs_to_process.append((input_path, base_stem))

    for img_input, current_stem in image_inputs_to_process:
        _process_single_image(img_input, current_stem, output_dir, server_url, model_name, ignore_labels, layout_engine, logger)

    logger.info(f"全部处理完成！总计耗时: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run document processing")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="input/formula-handwritten.webp",
        help="Path to the input image or pdf",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://172.31.80.1:8080/v1",
        help="VLM server url",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PaddleOCR-Q4_K_M",
        help="VLM model name",
    )
    args = parser.parse_args()

    markdown_ignore_labels = [
        "number",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    ]

    process_document(
        input_path=args.input,
        server_url=args.server_url,
        model_name=args.model_name,
        ignore_labels=markdown_ignore_labels,
    )

import base64
import time
import argparse
from pathlib import Path
from typing import List

import cv2
import niquests
from rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput

# 1. 初始化 PP-DocLayoutV3 模型
layout_engine = RapidLayout(
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


def process_document(image_path: str, server_url: str, model_name: str, ignore_labels: List[str]):
    start = time.time()
    image_stem = Path(image_path).stem

    output_dir = Path(f"output_{image_stem}")
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"读取图片失败: {image_path}")
        return

    results = layout_engine(image_path)
    if results.boxes is None:
        print(f"{image_path}: 未检测到任何版面元素")
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
                print(f"已成功提取图表并保存至: {img_save_path}")
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
                    print(f"Server Error: {response.text}")
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                if not content:
                    continue

                if label in ["doc_title", "paragraph_title"]:
                    markdown_chunks.append(f"## {content}")
                else:
                    markdown_chunks.append(content)
            except Exception as e:
                print(f"处理区块 [{label}] 失败: {e}")

    output_file = output_dir / f"{image_stem}.md"
    final_markdown = "\n\n".join(markdown_chunks)
    output_file.write_text(final_markdown, encoding="utf-8")

    print(f"全部处理完成！总计耗时: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rapid2 document processing")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="input/formula-handwritten.webp",
        help="Path to the input image or pdf",
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
        image_path=args.input,
        server_url="http://172.31.80.1:8080/v1",
        model_name="PaddleOCR-Q4_K_M",
        ignore_labels=markdown_ignore_labels,
    )

import base64
import io
import time
import argparse
from pathlib import Path
import httpx
from PIL import Image
from rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput

# 1. 初始化 PP-DocLayoutV3 模型
cfg = RapidLayoutInput(
    model_type=ModelType.PP_DOC_LAYOUTV3,
    engine_type=EngineType.ONNXRUNTIME,
    conf_thresh=0.5,
    iou_thresh=0.5,
)
layout_engine = RapidLayout(cfg=cfg)


def crop_to_base64(image_path: str, bbox: list) -> str:
    """根据边界框裁剪图片区域，并转换为 base64 格式，供给 VLM 识别"""
    with Image.open(image_path) as img:
        cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        buffered = io.BytesIO()
        cropped.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"


def save_crop_image(image_path: str, bbox: list, save_path: Path):
    """直接裁剪图片区域并保存到本地"""
    with Image.open(image_path) as img:
        cropped = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        cropped.save(save_path, format="PNG")


def process_document(
    image_path: str, server_url: str, model_name: str, ignore_labels: list
):
    start = time.time()
    image_stem = Path(image_path).stem

    # 定义输出目录结构
    output_dir = Path("./output_rapid")
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # 运行版面分析
    results = layout_engine(image_path)
    if results.boxes is None:
        print(f"{image_path}: 未检测到任何版面元素")
        return

    # 提示词映射
    prompt_mapping = {
        "text": "OCR:",
        "content": "OCR:",
        "doc_title": "OCR:",
        "paragraph_title": "OCR:",
        "table": "Table Recognition:",
        "display_formula": "Formula Recognition:",
        "inline_formula": "Formula Recognition:",
        "seal": "Seal Recognition:",
    }

    markdown_chunks = []
    chart_counter = 0

    with httpx.Client(timeout=60.0) as client:
        for box, label, score in zip(
            results.boxes, results.class_names, results.scores
        ):
            if label in ignore_labels:
                continue

            # === 图表拦截与保存 ===
            if label in ["chart", "figure"]:
                chart_counter += 1
                img_name = f"{image_stem}_{label}_{chart_counter}.png"
                img_save_path = imgs_dir / img_name

                save_crop_image(image_path, box, img_save_path)

                # 干净的图片引用，两边不留多余的换行符，由后面的 join 统一处理
                markdown_chunks.append(f"![{label}](imgs/{img_name})")
                print(f"已成功提取图表并保存至: {img_save_path}")
                continue

            # === 文本/表格调用 VLM 识别 ===
            prompt = prompt_mapping.get(label, "OCR:")
            base64_image = crop_to_base64(image_path, box)

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

            try:
                response = client.post(f"{server_url}/chat/completions", json=payload)
                response.raise_for_status()

                # 关键改动 1：使用 .strip() 斩断模型返回文本两端可能存在的隐形换行或空格
                content = response.json()["choices"][0]["message"]["content"].strip()

                if not content:
                    continue

                # 关键改动 2：去掉原来手工包裹的 \n，保持内容块的“纯净”
                if label in ["doc_title", "paragraph_title"]:
                    markdown_chunks.append(f"## {content}")
                else:
                    markdown_chunks.append(content)

            except Exception as e:
                print(f"处理区块 [{label}] 失败: {e}")

    # 关键改动 3：用双换行符 "\n\n" 来缝合所有版面区块，确保在任何 Markdown 编辑器里都不会黏在一起
    # 4. 保存 Markdown 结果
    output_file = output_dir / f"{image_stem}.md"

    final_markdown = "\n\n".join(markdown_chunks)
    output_file.write_text(final_markdown, encoding="utf-8")

    print(f"全部处理完成！总计耗时: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rapid document processing")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="input/formula-handwritten.webp",
        help="Path to the input image or pdf",
    )
    args = parser.parse_args()

    # ⚠️ 特别注意：这里千万不要包含 "chart" 和 "figure"，否则它们会被提前过滤掉
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

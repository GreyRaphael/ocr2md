import base64
import io
import time
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


def get_cropped_image(image_path: str, bbox: list, padding: int = 10) -> Image.Image:
    """带 Padding（外扩缓冲）的图片裁剪核心函数，防止边缘手写字符被切碎"""
    with Image.open(image_path) as img:
        width, height = img.size
        # 向外扩展 bbox，同时防止越界
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(width, int(bbox[2]) + padding)
        y2 = min(height, int(bbox[3]) + padding)
        return img.crop((x1, y1, x2, y2))


def crop_to_base64(image_path: str, bbox: list, padding: int = 10) -> str:
    """裁剪图片区域并转换为 base64 格式，加入 padding"""
    cropped = get_cropped_image(image_path, bbox, padding=padding)
    buffered = io.BytesIO()
    cropped.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def save_crop_image(image_path: str, bbox: list, save_path: Path, padding: int = 10):
    """直接裁剪图片区域并保存到本地，加入 padding"""
    cropped = get_cropped_image(image_path, bbox, padding=padding)
    cropped.save(save_path, format="PNG")


def clean_latex_content(content: str, label: str) -> str:
    """后处理清洗：统一将 VLM 返回的 \\( \\) 或 \\[ \\] 替换为标准的 $ 或 $$"""
    # 替换常见的行内和独立公式转义符
    content = content.replace(r"\(", "$").replace(r"\)", "$")
    content = content.replace(r"\[", "$$").replace(r"\]", "$$")

    # 根据版面标签强制进行二次包装保护（如果模型漏写了包裹符号）
    if label == "display_formula":
        if not content.startswith("$$"):
            if content.startswith("$") and content.endswith("$"):
                content = f"$${content[1:-1]}$$"  # $ 改为 $$
            else:
                content = f"$${content}$$"
    elif label == "inline_formula":
        if not content.startswith("$"):
            content = f"${content}$"

    return content


def process_document(
    image_path: str, server_url: str, model_name: str, ignore_labels: list
):
    start = time.time()
    image_stem = Path(image_path).stem

    output_dir = Path("./output")
    imgs_dir = output_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    results = layout_engine(image_path)
    if results.boxes is None:
        print(f"{image_path}: 未检测到任何版面元素")
        return

    # === 修改点 1：升级 Prompt，规范 VLM 的输出行为 ===
    prompt_mapping = {
        "text": "OCR:",
        "content": "OCR:",
        "doc_title": "OCR:",
        "paragraph_title": "OCR:",
        "table": "Table Recognition:",
        "display_formula": "Convert this independent formula into standard LaTeX code. Do NOT include markdown delimiters like $$ or \\[, just the raw LaTeX formula content:",
        # "display_formula": "Formula Recognition into the raw LaTeX formula content:",
        "inline_formula": "Convert this inline formula into standard LaTeX code. Do NOT include markdown delimiters like $ or \\(, just the raw LaTeX formula content:",
        # "inline_formula": "Formula Recognition into the raw LaTeX formula content:",
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

            # === 图表拦截与保存（应用 Padding） ===
            if label in ["chart", "figure"]:
                chart_counter += 1
                img_name = f"{image_stem}_{label}_{chart_counter}.png"
                img_save_path = imgs_dir / img_name

                save_crop_image(
                    image_path, box, img_save_path, padding=5
                )  # 图表边缘外扩5像素即可
                markdown_chunks.append(f"![{label}](imgs/{img_name})")
                print(f"已成功提取图表并保存至: {img_save_path}")
                continue

            # === 文本/表格/公式调用 VLM 识别 ===
            prompt = prompt_mapping.get(label, "OCR:")

            # === 修改点 2：裁剪公式和文本时，外扩 10 像素，防止手写字符边缘丢失 ===
            box_padding = 12 if "formula" in label else 8
            base64_image = crop_to_base64(image_path, box, padding=box_padding)

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

                content = response.json()["choices"][0]["message"]["content"].strip()
                if not content:
                    continue

                # === 修改点 3：对识别出的公式进行后处理规范化清洗 ===
                if "formula" in label or "$" in content or "\\" in content:
                    content = clean_latex_content(content, label)

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
    markdown_ignore_labels = [
        "number",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    ]

    process_document(
        # image_path="input/fupeng3_1.png",
        image_path="input/formula-handwritten.webp",
        server_url="http://172.31.80.1:8080/v1",
        model_name="PaddleOCR-Q4_K_M",
        # model_name="GLMOCR-Q4_K_M",
        ignore_labels=markdown_ignore_labels,
    )

"""
轻量 OCR 管线: PP-DocLayoutV3 (ONNX) + llama-cpp-server PaddleOCR-Q4_K_M

依赖: nopaddle[onnx] + httpx  (无 PaddlePaddle / PyTorch)

原理:
  1. nopaddle 的 PP-DocLayoutV3 ONNX 检测器定位文档区域 (text/table/formula/...)
  2. 裁剪每个需要 OCR 的区域, 通过 OpenAI-compatible API 发送到 llama-cpp-server
  3. 用 PaddleOCR 原始 prompt ("OCR:", "Table Recognition:") 保持识别质量
  4. 按 reading order 组装为 Markdown
"""

import base64
import io
import logging
import time
from pathlib import Path

import httpx
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── nopaddle 布局检测器 (PP-DocLayoutV3 ONNX) ─────────────────────────────
from nopaddle.layout.detector import (
    PPDocLayoutDetector,
    VISUAL_CLASSES,
)

# ─── 配置 ────────────────────────────────────────────────────────────────────
LLAMA_CPP_BASE_URL = "http://172.31.80.1:8080/v1"
LLAMA_CPP_MODEL = "PaddleOCR-Q4_K_M"

# 与原 PaddleOCRVL 的 markdown_ignore_labels 对齐
IGNORE_CLASSES = {
    "number",
    # "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
}

# ─── llama-cpp Backend ────────────────────────────────────────────────────────


class LlamaCppBackend:
    """通过 OpenAI-compatible Chat API 调用 llama-cpp-server 做 VLM 推理。

    关键: 直接使用 PaddleOCR 原始 prompt ("OCR:", "Table Recognition:"), 
    不做自然语言翻译, 保持与模型训练时一致的指令格式。
    """

    def __init__(
        self,
        base_url: str = LLAMA_CPP_BASE_URL,
        model: str = LLAMA_CPP_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=timeout)

    def generate(self, image: Image.Image, prompt: str) -> str:
        """发送裁剪区域图片 + prompt 到 llama-cpp-server, 返回识别文本。"""
        # 图片 → base64 data URL
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        data_url = f"data:image/png;base64,{b64}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
        }

        resp = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self):
        self._client.close()


# ─── 图片/PDF 渲染 ──────────────────────────────────────────────────────────


def load_images(input_path: str) -> list[tuple[Image.Image, int]]:
    """加载输入, 返回 [(PIL.Image, page_index), ...]。支持 PNG/JPG/PDF。"""
    p = Path(input_path)
    suffix = p.suffix.lower()

    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
        img = Image.open(p).convert("RGB")
        return [(img, 0)]

    if suffix == ".pdf":
        import pymupdf

        doc = pymupdf.open(str(p))
        images = []
        scale = 2.0  # ~144 DPI, 与 nopaddle 默认一致
        for i in range(doc.page_count):
            page = doc[i]
            mat = pymupdf.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", (int(pix.width), int(pix.height)), pix.samples)
            images.append((img, i))
        return images

    raise ValueError(f"不支持的文件格式: {suffix}")


# ─── 核心管线 ─────────────────────────────────────────────────────────────────


def process_image(
    image: Image.Image,
    page_index: int,
    detector: PPDocLayoutDetector,
    backend: LlamaCppBackend,
    imgs_dir: Path | None = None,
) -> dict:
    """处理单张图片: 布局检测 → 区域 OCR/裁剪 → 结构化结果。"""
    # 1. 布局检测
    detections = detector.detect(image)

    # 2. 过滤 + 分类
    regions = []
    for det in detections:
        cls = det["class"]
        if cls in IGNORE_CLASSES:
            continue
        bbox = det["bbox"]

        if cls in VISUAL_CLASSES:
            # 图片/图表: 裁剪保存到 imgs_dir
            cropped = image.crop((
                bbox["x_min"], bbox["y_min"],
                bbox["x_max"], bbox["y_max"],
            ))
            img_filename = f"img_{cls}_{page_index}_{det['read_order']}.png"
            img_path = None
            if imgs_dir is not None:
                imgs_dir.mkdir(parents=True, exist_ok=True)
                img_path = imgs_dir / img_filename
                cropped.save(str(img_path))
            regions.append({
                "type": "visual",
                "class": cls,
                "bbox": det["bbox"],
                "read_order": det["read_order"],
                "content": img_filename,  # 相对路径: imgs/xxx.png
            })
            continue
        if det["prompt"] is None:
            # 无 prompt 的跳过类
            continue

        # 3. 裁剪区域
        cropped = image.crop((
            bbox["x_min"], bbox["y_min"],
            bbox["x_max"], bbox["y_max"],
        ))

        # 4. OCR / Table / Formula 识别
        prompt = det["prompt"]  # "OCR:" / "Table Recognition:" / "Formula Recognition:"
        try:
            text = backend.generate(cropped, prompt)
        except Exception as e:
            logger.warning("OCR failed for %s region: %s", cls, e)
            text = f"[识别失败: {cls}]"

        regions.append({
            "type": "content",
            "class": cls,
            "element_type": det["element_type"],
            "bbox": det["bbox"],
            "bbox_normalized": det["bbox_normalized"],
            "read_order": det["read_order"],
            "content": text,
        })

    # 按 reading order 排序
    regions.sort(key=lambda r: r["read_order"])

    return {
        "page_index": page_index,
        "regions": regions,
    }


def regions_to_markdown(result: dict, page_index: int, imgs_rel: str = "imgs") -> str:
    """将区域结果转为 Markdown 文本。"""
    lines = []
    for region in result["regions"]:
        if region["type"] == "visual":
            # 图片: 输出 Markdown 图片引用
            lines.append(f"![{region['class']}]({imgs_rel}/{region['content']})")
            lines.append("")
            continue

        content = region["content"].strip()
        if not content:
            continue

        # 根据元素类型决定 Markdown 格式
        elem_type = region.get("element_type", "text")
        if elem_type == "title":
            lines.append(f"# {content}")
        elif elem_type == "section_header":
            lines.append(f"## {content}")
        elif elem_type == "table":
            lines.append(content)  # Table Recognition 已返回 Markdown 表格
        elif elem_type == "formula":
            lines.append(f"$$\n{content}\n$$")
        elif elem_type == "caption":
            lines.append(f"*{content}*")
        elif elem_type == "footnote":
            lines.append(f"[^{content}]")
        else:
            lines.append(content)

        lines.append("")  # 段落间空行

    return "\n".join(lines)


def save_markdown(md_text: str, save_path: str, filename: str):
    """保存 Markdown 到文件。"""
    out_dir = Path(save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename
    out_file.write_text(md_text, encoding="utf-8")
    logger.info("Saved: %s", out_file)


# ─── 主入口 ───────────────────────────────────────────────────────────────────


def _ensure_layout_model() -> Path:
    """下载 PP-DocLayoutV3 ONNX 模型到扁平目录 (避免 symlink 导致 onnxruntime 路径校验失败)。"""
    from huggingface_hub import snapshot_download

    local_dir = Path.home() / ".cache" / "nopaddle" / "models" / "PP-DocLayoutV3-ONNX"
    if not local_dir.exists() or not (local_dir / "PP-DocLayoutV3.onnx").exists():
        logger.info("Downloading PP-DocLayoutV3-ONNX model...")
        snapshot_download(
            repo_id="Bei0001/PP-DocLayoutV3-ONNX",
            local_dir=str(local_dir),
        )
    return local_dir / "PP-DocLayoutV3.onnx"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="轻量 OCR: PP-DocLayoutV3 + llama-cpp-server")
    parser.add_argument("input", help="输入文件路径 (PNG/JPG/PDF)")
    parser.add_argument("-o", "--output", default="./output", help="输出目录 (默认: ./output)")
    parser.add_argument("--server-url", default=LLAMA_CPP_BASE_URL, help="llama-cpp-server URL")
    parser.add_argument("--model", default=LLAMA_CPP_MODEL, help="模型名称")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="布局检测置信度阈值")
    args = parser.parse_args()

    start = time.time()

    # 初始化检测器 (首次自动下载 ~138MB ONNX 模型)
    logger.info("Loading PP-DocLayoutV3 detector...")
    model_path = _ensure_layout_model()
    detector = PPDocLayoutDetector(model_path, conf_threshold=args.conf_threshold)

    # 初始化 llama-cpp backend
    backend = LlamaCppBackend(base_url=args.server_url, model=args.model)

    # 加载输入
    images = load_images(args.input)
    logger.info("Loaded %d page(s) from %s", len(images), args.input)

    input_stem = Path(args.input).stem
    imgs_dir = Path(args.output) / "imgs"

    try:
        for img, page_idx in images:
            logger.info("Processing page %d...", page_idx + 1)
            result = process_image(img, page_idx, detector, backend, imgs_dir=imgs_dir)

            # 生成 Markdown
            md = regions_to_markdown(result, page_idx)

            # 保存
            if len(images) == 1:
                filename = f"{input_stem}.md"
            else:
                filename = f"{input_stem}_p{page_idx + 1:03d}.md"

            save_markdown(md, args.output, filename)
            logger.info("Page %d: %d regions processed", page_idx + 1, len(result["regions"]))
    finally:
        backend.close()

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()

import base64
import time
import argparse
import logging
from pathlib import Path
from functools import lru_cache

import cv2
import niquests
import numpy as np
from rapid_layout import EngineType, ModelType, RapidLayout, RapidLayoutInput


@lru_cache(maxsize=1)
def get_layout_engine() -> RapidLayout:
    """初始化并缓存 RapidLayout 引擎，保证全局只加载一次模型"""
    return RapidLayout(
        cfg=RapidLayoutInput(
            model_type=ModelType.PP_DOC_LAYOUTV3,
            engine_type=EngineType.ONNXRUNTIME,
            conf_thresh=0.35,
            iou_thresh=0.35,
        )
    )


class OCR2MDProcessor:
    """
    轻量级 OCR 转 Markdown 处理器。
    结合了 PP-DocLayoutV3 的本地极速版面分析与远端大模型 (如 llama.cpp) 的并发 OCR 能力。
    """
    
    # 针对不同排版元素的专用提示词映射
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

    def __init__(self, input_path: str, server_url: str, model_name: str, ignore_labels: list[str], api_key: str = None):
        """初始化处理器，绑定所有上下文状态"""
        self.input_path = input_path
        self.server_url = server_url
        self.model_name = model_name
        self.ignore_labels = ignore_labels
        self.api_key = api_key
        
        # 路径与目录初始化
        self.path_obj = Path(input_path)
        self.base_stem = self.path_obj.stem
        self.output_dir = Path(f"output_{self.base_stem}")
        self.imgs_dir = self.output_dir / "imgs"
        
        # 提前检查文件是否存在以避免生成空目录
        if not self.path_obj.exists() or not self.path_obj.is_file():
            raise FileNotFoundError(f"输入文件不存在或非文件: {input_path}")
            
        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志和版面分析模型
        self.logger = self._setup_logger()
        self.layout_engine = get_layout_engine()

    def _setup_logger(self) -> logging.Logger:
        """配置双路输出日志 (控制台 + 文件)"""
        log_file = self.output_dir / "ocr2md.log"
        logger = logging.getLogger(self.base_stem)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def _crop_to_base64(img_cv: np.ndarray, bbox: list) -> str:
        """将 numpy 图片矩阵根据边界框裁剪，并返回高质量 JPEG 的 Base64 编码"""
        x1, y1, x2, y2 = map(int, bbox)
        cropped = img_cv[y1:y2, x1:x2]
        # 使用 JPG 95 质量，肉眼无损且比 PNG base64 体积小 50% 以上
        _, buffer = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    @staticmethod
    def _save_crop_image(img_cv: np.ndarray, bbox: list, save_path: Path):
        """将图片裁剪并保存到本地"""
        x1, y1, x2, y2 = map(int, bbox)
        cropped = img_cv[y1:y2, x1:x2]
        cv2.imwrite(str(save_path), cropped)

    def _build_vlm_payload(self, img_cv: np.ndarray, box: list, label: str) -> dict:
        """构建用于发送给大模型的 JSON Payload"""
        prompt = self.PROMPT_MAPPING.get(label, "OCR:")
        base64_image = self._crop_to_base64(img_cv, box)
        return {
            "model": self.model_name,
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

    def _process_page(self, img_cv: np.ndarray, image_stem: str, client: niquests.Session, global_start: float = None, current_page_idx: int = None):
        """处理单一页面：版面分析 -> 切割图表保存 -> 并发 OCR 识别 -> 生成 Markdown"""
        page_start = time.time()

        # 1. 运行本地版面分析模型
        results = self.layout_engine(img_cv)
        if results.boxes is None:
            self.logger.warning(f"[{image_stem}] 未检测到任何版面元素")
            return

        items_to_process = []
        chart_counter = 0

        # 2. 利用传入的复用 Session，并发抛出所有网络 IO 请求
        for box, label, _ in zip(results.boxes, results.class_names, results.scores):
            if label in self.ignore_labels:
                continue

            # 提取图表并保存
            if label in ["chart", "figure"]:
                chart_counter += 1
                img_name = f"{image_stem}_{label}_{chart_counter}.jpg"
                img_save_path = self.imgs_dir / img_name

                self._save_crop_image(img_cv, box, img_save_path)
                items_to_process.append(("chart", f"![{label}](imgs/{img_name})"))
                self.logger.info(f"已成功提取图表并保存至: {img_save_path}")
                continue

            # 其他文本/表格/公式区块，并发丢给大模型
            payload = self._build_vlm_payload(img_cv, box, label)
            
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # multiplexed=True 时 post 不阻塞，立即返回 Lazy Response
            response = client.post(f"{self.server_url}/chat/completions", headers=headers, json=payload, timeout=120.0)
            items_to_process.append(("vlm", label, response))

        # 阻塞统一等待这页的并发请求全部结束
        client.gather()

        # 3. 按原始版面顺序组合 Markdown
        markdown_chunks = []
        for item in items_to_process:
            if item[0] == "chart":
                markdown_chunks.append(item[1])
            elif item[0] == "vlm":
                _, label, response = item
                try:
                    if not response.ok:
                        self.logger.error(f"Server Error: {response.text}")
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    if not content:
                        continue

                    if label in ["doc_title", "paragraph_title"]:
                        markdown_chunks.append(f"## {content}")
                    else:
                        markdown_chunks.append(content)
                except Exception as e:
                    self.logger.error(f"处理区块 [{label}] 失败: {e}")

        # 4. 保存并记录耗时日志
        output_file = self.output_dir / f"{image_stem}.md"
        final_markdown = "\n\n".join(markdown_chunks)
        output_file.write_text(final_markdown, encoding="utf-8")

        page_elapsed = time.time() - page_start
        log_msg = f"[{image_stem}] 处理完成, 本页耗时: {page_elapsed:.2f}s"
        if global_start is not None and current_page_idx is not None and current_page_idx > 0:
            total_elapsed = time.time() - global_start
            avg_speed = total_elapsed / current_page_idx
            log_msg += f", 累计平均速度: {avg_speed:.2f} s/page"
        log_msg += f", 结果已保存至 {output_file}"
        self.logger.info(log_msg)

    def run(self):
        """启动文档处理流程"""
        start = time.time()
        page_stems = []

        # 核心优化：在整个文档处理生命周期内复用同一个 HTTP/2 Session，避免跨页 TLS 握手开销
        with niquests.Session(multiplexed=True) as client:
            client.trust_env = False

            # 处理 PDF 文件 (懒加载，不占用过多内存)
            if self.path_obj.suffix.lower() == ".pdf":
                try:
                    import fitz
                except ImportError:
                    self.logger.error("处理 PDF 需要安装 pymupdf: uv add pymupdf")
                    return
                    
                self.logger.info("正在逐页加载并处理 PDF (节约内存)...")
                try:
                    doc = fitz.open(self.input_path)
                    for i in range(len(doc)):
                        page = doc[i]
                        # 强制 alpha=False，直接抛弃透明通道，提升性能
                        pix = page.get_pixmap(alpha=False)
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                        
                        if pix.n == 3:
                            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        else:
                            img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

                        current_stem = f"{self.base_stem}_page{i + 1}"
                        page_stems.append(current_stem)
                        
                        self._process_page(img_cv, current_stem, client=client, global_start=start, current_page_idx=i + 1)
                except Exception as e:
                    self.logger.error(f"打开或处理 PDF 失败: {e}")
                    return
                    
            # 处理普通单张图片
            else:
                img_cv = cv2.imread(self.input_path)
                if img_cv is None:
                    self.logger.error(f"读取图片失败: {self.input_path}")
                    return
                    
                page_stems.append(self.base_stem)
                self._process_page(img_cv, self.base_stem, client=client, global_start=start, current_page_idx=1)

        # === 合并多页 Markdown ===
        if len(page_stems) > 1:
            combined_md_path = self.output_dir / f"{self.base_stem}_full.md"
            combined_content = []
            for stem in page_stems:
                page_md = self.output_dir / f"{stem}.md"
                if page_md.exists():
                    combined_content.append(page_md.read_text(encoding="utf-8"))

            if combined_content:
                combined_md_path.write_text("\n\n---\n\n".join(combined_content), encoding="utf-8")
                self.logger.info(f"多页 Markdown 已合并至: {combined_md_path}")

        self.logger.info(f"全部处理完成！总计耗时: {time.time() - start:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR2MD: Lightweight document processing pipeline")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="input/formula-handwritten.webp",
        help="Path to the input image or PDF file",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://172.31.80.1:8080/v1",
        help="VLM server url (e.g. llama.cpp server)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="PaddleOCR-Q4_K_M",
        help="VLM model name",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for the VLM server",
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

    try:
        processor = OCR2MDProcessor(
            input_path=args.input,
            server_url=args.server_url,
            model_name=args.model_name,
            ignore_labels=markdown_ignore_labels,
            api_key=args.api_key,
        )
        processor.run()
    except Exception as e:
        print(f"Error: {e}")

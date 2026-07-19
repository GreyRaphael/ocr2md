import time
import argparse
from paddleocr import PaddleOCRVL


def main():
    parser = argparse.ArgumentParser(description="Run PaddleOCRVL")
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        default="input/formula-handwritten.webp",
        help="Path to the input image or pdf",
    )
    args = parser.parse_args()

    start = time.time()
    pipeline = PaddleOCRVL(
        vl_rec_backend="llama-cpp-server",
        vl_rec_server_url="http://172.31.80.1:8080/v1",
        vl_rec_api_model_name="PaddleOCR-Q4_K_M",
        use_chart_recognition=False,
        use_doc_unwarping=False,
        use_doc_orientation_classify=False,
        markdown_ignore_labels=[
            "number",
            # "footnote",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "aside_text",
        ],
    )  # GPU本地推理
    for res in pipeline.predict(input=args.input):
        res.save_to_markdown(save_path="./output_raw")

    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()

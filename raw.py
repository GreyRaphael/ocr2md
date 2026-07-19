import time
from paddleocr import PaddleOCRVL

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
for res in pipeline.predict(input="input/formula-handwritten.webp"):
    # for res in pipeline.predict(input="input/test_footnote.png"):
    # for res in pipeline.predict(input="input/test_table_1.png"):
    # for res in pipeline.predict(input="input/fupeng3_1.png"):
    res.save_to_markdown(save_path="./output_raw")

print(f"Total time: {time.time() - start:.2f} seconds")

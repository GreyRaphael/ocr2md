# OCR2MD

A high-performance, lightweight tool to convert complex PDF documents and images into clean, structured Markdown files.

## Architecture

`ocr2md` is built for maximum speed and memory safety:
1. **Local Layout Analysis**: Uses `rapid-layout` (ONNX `pp_doc_layoutv3`) to analyze document layout (titles, paragraphs, formulas, tables, charts) entirely offline and instantly on CPU.
2. **Network I/O Optimization**: Extracts snippets directly from memory buffers (Zero Disk I/O) and uses highly optimized JPEG compression to minimize network transmission payloads. 
3. **Multiplexed VLM Concurrency**: Utilizes `niquests` HTTP/2 multiplexing to fire all layout block OCR requests asynchronously, fully exploiting the throughput of the target VLM server (e.g., `llama.cpp`).
4. **Lazy Loading**: Processes enormous PDFs (hundreds of pages) efficiently by loading and releasing one page at a time to prevent Out-Of-Memory (OOM) crashes.

## Quickstart

Run via `uv`:
```bash
uv run main.py input/your_file.pdf
```

The output will be saved into the `output_your_file` directory, containing:
- `imgs/`: Auto-extracted charts and figures (lossless format).
- `pageX.md`: Formatted markdown output for each individual page.
- `your_file_full.md`: The combined markdown file representing the entire document.
- `ocr2md.log`: The runtime execution logs with speed profiling.

## Configuration

You can override the target Vision-Language Model server details via CLI parameters:

```bash
uv run main.py input/example.pdf \
    --server-url http://172.31.80.1:8080/v1 \
    --api-key sk-xxxxxx \
    --model-name PaddleOCR-Q4_K_M
```

---

## Benchmark Notes
*Performance measurements using the optimized OOP architecture (multiplexed VLM concurrency + lazy loading):*

**Current (ocr2md):**
- **fupeng3.pdf** (6 pages): 31.02s (~5.17s/page)
- **fupeng3_1.png**: 31.31s
- **test_footnote.png**: 4.94s
- **test_formula.jpg**: 9.22s
- **test_table_1.png**: 5.69s
- **formula-handwritten.webp**: 16.16s

# Document Extraction with LLMs

This project demonstrates how to extract structured information from a PDF document using a pre-trained language model (`MiniCPM-Llama3-V-2_5`) with PyTorch, PyMuPDF, and the Python Imaging Library (PIL). The code extracts text data such as questions, lists, and tables from PDF images, processing them as markdown tables where applicable. 

## Prerequisites

To run this code, you need to have the following libraries installed:

- `torch`
- `PIL` (Python Imaging Library)
- `transformers`
- `fitz` (PyMuPDF)

To install these dependencies, you can use pip:

```bash
pip install torch transformers pillow pymupdf

# Document Extraction Using LLMs

This project demonstrates how to extract textual information, including questions and tables, from PDF documents using Large Language Models (LLMs). The script leverages the [MiniCPM-Llama3-V-2_5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) model from the OpenBMB repository with PyTorch and Transformers libraries, along with PyMuPDF for PDF processing.

## Requirements

- Python 3.x
- PyTorch
- Transformers
- PIL (Pillow)
- PyMuPDF

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install torch transformers pillow pymupdf
```

## Usage
The script processes a PDF file, extracts images from each page, and then uses a Large Language Model (LLM) to extract questions and tables from the images. The output is printed directly to the console.

## Steps
Model Loading: The script loads the MiniCPM-Llama3-V-2_5 model and tokenizer from Hugging Face's model hub. The model is set to evaluation mode and moved to GPU for faster processing.

```bash
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16
)
model = model.to(device="cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True
)
model.eval()
```
PDF Processing: The script opens a PDF file, extracts images from each page, and stores them in a list.

```bash 
pdf_path = "2021.pdf"
pdf_document = fitz.open(pdf_path)

images = []

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    images.append(img)

pdf_document.close()

Text Extraction: For each image, the script sends a query to the LLM to extract questions and tables, ignoring headers, footers, and front pages.

question = """Extract all the questions as text in this image.
if there are no questions skip it
If there is a header or a footer, or a front page, just ignore it.
Extract tables as markdown tables if there are any.
Don't use the subtitles for the list items, just return the list as text.
if there is something that you cannot record as text add description
"""
msgs = [{"question": "question", "content": question}]

res = model.chat(
    image=images[0],  # loop through all images in images list here
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)
```
## Customization
PDF Path: Change the pdf_path variable to the location of your PDF file.
LLM Configuration: Adjust the model parameters such as temperature for sampling behavior and add any system_prompt if required for specific instructions.
More Information
For more information on how this script works and a detailed explanation of document extraction using LLMs, visit this blog post.

## Author
Eammon Kiprotich.

## References
For a more Detailed explanation visit this link:
https://www.pondhouse-data.com/blog/document-extraction-with-llms

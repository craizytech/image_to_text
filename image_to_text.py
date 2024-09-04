import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF

model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True, torch_dtype=torch.float16
)
model = model.to(device="cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-Llama3-V-2_5", trust_remote_code=True
)
model.eval()

pdf_path = "2021.pdf"
pdf_document = fitz.open(pdf_path)

images = []

for page_number in range(len(pdf_document)):
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    images.append(img)

pdf_document.close()

question = """Extract all the questions as text in this image.
if there are no questions skip it
If there is a header or a footer, or a front page, just ignore it.
Extract tables as markdown tables if there are any.
Don't use the subtitles for the list items, just return the list as text.
if there is something that you cannot record as text add description
"""
msgs = [{"question": "qustion", "content": question}]

res = model.chat(
    image=images[0], # loop through all images in images list here
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True,
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)
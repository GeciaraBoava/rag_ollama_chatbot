# test_env.py
import llama_index
import torch
import faiss
from langchain_community.chat_models import ChatOllama
from docx import Document
from pypdf import PdfReader

print("✅ Tudo instalado e funcionando!")
print("Torch version:", torch.__version__)
print("FAISS version:", faiss.__version__)

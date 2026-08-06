from pathlib import Path

import torch
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "src" / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODEL_DIR = PROJECT_ROOT / "Mistral-7B-Instruct-v0.2"
MODEL_NAME = str(LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else "mistralai/Mistral-7B-Instruct-v0.2")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    return tokenizer, model


def build_rag_index():
    documents = []
    for md_file in DB_DIR.glob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        return None

    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = splitter.split_documents(documents)
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding)


def generate_text(prompt, tokenizer, model, max_length=500, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs][0]


def ask_question(query, vectordb, tokenizer, model):
    docs = vectordb.similarity_search(query, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"You are a helpful assistant. Use the following context to answer the question. Question: {query} Context: {context}"
    return generate_text(prompt, tokenizer, model).strip()


def main():
    tokenizer, model = load_model()
    vectordb = build_rag_index()
    if vectordb is not None:
        print(ask_question("What is the most detailed dream that I had?", vectordb, tokenizer, model))


if __name__ == "__main__":
    main()

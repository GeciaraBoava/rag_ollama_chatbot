import sys
from pathlib import Path
from typing import List
import requests
import warnings

# Suprimir avisos do NLTK
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*punkt_tab.*')

# llama-index 0.12.x (nova estrutura de imports)
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama

import faiss
from docx import Document as DocxDocument
from pypdf import PdfReader

DEFAULT_DOCUMENTS_FOLDER = "documentos"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "deepseek-coder"
EXIT_COMMANDS = ["sair", "exit", "quit"]

def check_ollama_running():
    """Verifica se o Ollama está rodando."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def load_documents(folder_path: str) -> List[Document]:
    """Carrega documentos da pasta especificada."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"A pasta '{folder_path}' não existe.")
    
    files = list(folder.glob("*"))
    if not files:
        raise ValueError(f"A pasta '{folder_path}' está vazia.")

    docs = []
    for f in files:
        try:
            if f.suffix.lower() == ".txt":
                content = f.read_text(encoding="utf-8")
            elif f.suffix.lower() == ".docx":
                docx = DocxDocument(str(f))
                content = "\n".join([p.text for p in docx.paragraphs if p.text.strip()])
            elif f.suffix.lower() == ".pdf":
                reader = PdfReader(str(f))
                content = "\n".join([p.extract_text() or "" for p in reader.pages])
            else:
                print(f"⚠️  Ignorando {f.name} (tipo não suportado)")
                continue

            if content and content.strip():
                docs.append(
                    Document(
                        text=content, 
                        doc_id=str(f.name), 
                        metadata={"source": str(f)}
                    )
                )
                print(f"✅ {f.name} carregado ({len(content)} chars)")
            else:
                print(f"⚠️  Arquivo vazio: {f.name}")
        except Exception as e:
            print(f"❌ Erro lendo {f.name}: {e}")

    if not docs:
        raise ValueError("Nenhum documento válido encontrado.")
    
    print(f"✅ Documentos carregados: {len(docs)} arquivo(s).")
    return docs

def create_faiss_vector_store(embedding_dim: int = 384):
    """Cria um vector store FAISS."""
    index = faiss.IndexFlatL2(embedding_dim)
    return FaissVectorStore(faiss_index=index)

def setup_rag_system(documents_folder: str = DEFAULT_DOCUMENTS_FOLDER):
    """Configura o sistema RAG completo."""
    print("📦 Configurando RAG...")
    documents = load_documents(documents_folder)

    print("🔧 Configurando embeddings...")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)

    print("🔧 Configurando LLM (Ollama)...")
    llm = Ollama(
        model=OLLAMA_MODEL_NAME, 
        request_timeout=120.0, 
        temperature=0.1,
        system_prompt="""
Você é um assistente que responde **somente em português**, de forma clara, objetiva e direta.
Não escreva em outro idioma.
Não peça mais contexto — sempre responda com base nas informações fornecidas.
Sempre explique com exemplos práticos quando possível.
Evite respostas vagas ou genéricas.
        """
    )

    # Configuração global do LlamaIndex 0.12.x (substitui ServiceContext)
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    print("🔧 Configurando FAISS...")
    vector_store = create_faiss_vector_store(embedding_dim=384)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🔧 Criando índice...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    query_engine = index.as_query_engine(similarity_top_k=3, streaming=False)
    print("✅ Sistema RAG configurado!")
    return query_engine

def run_chat_loop(query_engine):
    """Loop principal do chatbot."""
    print("\n" + "="*40)
    print("🤖 Chatbot RAG — digite sua pergunta")
    print("Digite 'sair' para encerrar.")
    print("="*40 + "\n")
    
    while True:
        try:
            pergunta = input("👤 Você: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in EXIT_COMMANDS:
                print("👋 Encerrando.")
                break
            
            print("🔍 Processando...")
            resposta = query_engine.query(pergunta)
            print(f"\n🤖 Bot: {resposta}\n")
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

def main():
    """Função principal."""
    if not check_ollama_running():
        print("❌ Ollama não está rodando. Rode: `ollama serve`")
        sys.exit(1)
    else:
        print("✅ Ollama rodando (checado /api/tags)")

    try:
        query_engine = setup_rag_system()
        run_chat_loop(query_engine)
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        raise

if __name__ == "__main__":
    main()
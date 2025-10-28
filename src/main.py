import sys
from pathlib import Path
from typing import List

# LlamaIndex imports - CORRETO
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama 

import faiss
from docx import Document as DocxDocument
from pypdf import PdfReader
import requests

# Constantes
DEFAULT_DOCUMENTS_FOLDER = "documentos"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # entende o texto e o transforma em vetores pra busca (um modelo da biblioteca Sentence-Transformers (HuggingFace))
OLLAMA_MODEL_NAME = "deepseek-coder" # gera a resposta usando o contexto recuperado (modelo carregado no Ollama)
EXIT_COMMANDS = ["sair", "exit", "quit"]

def check_ollama_running():
    """Verifica se o Ollama está rodando."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def load_documents(folder_path: str) -> List[Document]:
    """Carrega documentos TXT, DOCX e PDF."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"A pasta '{folder_path}' não existe ou não é um diretório.")
   
    files = list(folder.glob("*"))
    if not files:
        raise ValueError(f"A pasta '{folder_path}' está vazia.")

    documents = []
    for file in files:
        try:
            if file.suffix.lower() == ".txt":
                content = file.read_text(encoding="utf-8")
            elif file.suffix.lower() == ".docx":
                doc = DocxDocument(str(file))  
                content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            elif file.suffix.lower() == ".pdf":
                reader = PdfReader(str(file))  
                content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            else:
                print(f"⚠️ Ignorando arquivo não suportado: {file.name}")
                continue

            if content.strip():  # Só adiciona se tiver conteúdo
                documents.append(Document(text=content, id_=str(file)))
                print(f"✅ {file.name} carregado ({len(content)} caracteres)")
            else:
                print(f"⚠️ Arquivo vazio: {file.name}")

        except Exception as e:
            print(f"❌ Erro ao ler arquivo {file.name}: {e}")

    if not documents:
        raise ValueError(f"Nenhum documento válido encontrado em '{folder_path}'.")

    print(f"✅ Documentos carregados: {len(documents)} arquivo(s).")
    return documents

def create_faiss_vector_store(embedding_dim: int = 384):
    """Cria e configura o vector store FAISS."""
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    return vector_store

def setup_rag_system(documents_folder: str = DEFAULT_DOCUMENTS_FOLDER):
    """Configura todo o sistema RAG."""
    
    print("📦 Configurando sistema RAG...")
    
    # 1. Carregar documentos
    documents = load_documents(documents_folder)
    
    # 2. Configurar modelo de embeddings
    print("🔧 Configurando modelo de embeddings...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # 3. Configurar FAISS
    print("🔧 Configurando FAISS...")
    vector_store = create_faiss_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # 4. Configurar LLM
    print("🔧 Configurando LLM...")
    llm = Ollama(
        model=OLLAMA_MODEL_NAME, 
        request_timeout=120.0,
        temperature=0.1
    )
    
    # 5. Configurar settings globais
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50
    
    # 6. Criar índice
    print("🔧 Criando índice vetorial...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    # 7. Criar query engine
    print("🔧 Configurando query engine...")
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        streaming=False
    )
    
    print("✅ Sistema RAG configurado com sucesso!")
    return query_engine

def run_chat_loop(query_engine):
    print("\n" + "="*60)
    print("🤖 Chatbot RAG iniciado!")
    print(f"💡 Digite sua pergunta ou '{EXIT_COMMANDS[0]}' para encerrar.")
    print("="*60 + "\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in EXIT_COMMANDS:
                print("\n👋 Encerrando chatbot. Até logo!")
                break

            print("🔍 Processando sua pergunta...")
            resposta = query_engine.query(pergunta)
            print(f"\n🤖 Bot: {resposta}\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Chatbot interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro ao processar pergunta: {e}\n")

def main():
    print("\n" + "="*60)
    print("🚀 INICIALIZANDO CHATBOT RAG")
    print("="*60)

    try:
        # Verificar se Ollama está rodando
        if not check_ollama_running():
            print("❌ Ollama não está rodando.")
            print("💡 Execute 'ollama serve' em outro terminal ou verifique se já está rodando em segundo plano.")
            print("💡 Se já estiver rodando, aguarde alguns segundos e tente novamente.")
            sys.exit(1)
        else:
            print("✅ Ollama está rodando!")
            
        # Verificar se o modelo está disponível
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]
            
            if OLLAMA_MODEL_NAME not in model_names and f"{OLLAMA_MODEL_NAME}:latest" not in model_names:
                print(f"❌ Modelo '{OLLAMA_MODEL_NAME}' não encontrado.")
                print(f"💡 Execute: ollama pull {OLLAMA_MODEL_NAME}")
                sys.exit(1)
            else:
                print(f"✅ Modelo '{OLLAMA_MODEL_NAME}' disponível!")
                
        except Exception as e:
            print(f"⚠️ Não foi possível verificar os modelos: {e}")
            
        query_engine = setup_rag_system()
        run_chat_loop(query_engine)

    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print(f"💡 Crie a pasta '{DEFAULT_DOCUMENTS_FOLDER}' com seus documentos.")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Erro: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
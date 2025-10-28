import sys
from pathlib import Path
from typing import List
import requests
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*punkt_tab.*')
warnings.filterwarnings('ignore', message='.*validate_default.*')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

# llama-index 0.11.x (estrutura compatível)
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.ollama import Ollama

import faiss
from docx import Document as DocxDocument
from pypdf import PdfReader

DEFAULT_DOCUMENTS_FOLDER = "documentos"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "llama3.2:3b"  # Modelo mais rápido (antes: deepseek-coder)
EXIT_COMMANDS = ["sair", "exit", "quit"]

# Configurações de otimização (ajustadas para melhor precisão)
CHUNK_SIZE = 512  # Aumentado de 256 para capturar mais contexto
CHUNK_OVERLAP = 50  # Aumentado de 25 para melhor continuidade
SIMILARITY_TOP_K = 5  # Aumentado de 2 para recuperar mais chunks relevantes
MAX_TOKENS = 1024  # Aumentado de 512 para respostas mais completas
FAISS_INDEX_DIR = "./storage"  # Diretório para persistir índice

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
    
    # Verifica se existe índice salvo
    if Path(FAISS_INDEX_DIR).exists():
        print("🔄 Carregando índice existente...")
        try:
            from llama_index.core import load_index_from_storage
            
            # Configurar embeddings e LLM antes de carregar
            embed_model = HuggingFaceEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                cache_folder="./.cache/embeddings"
            )
            
            llm = Ollama(
                model=OLLAMA_MODEL_NAME, 
                request_timeout=60.0,
                temperature=0.1,
                additional_kwargs={
                    "num_predict": MAX_TOKENS,
                    "num_ctx": 2048,
                },
                system_prompt="""
Você é um assistente que responde **somente em português**, de forma clara, objetiva e direta.
Seja conciso. Não escreva em outro idioma.
Responda com base nas informações fornecidas.
Use exemplos práticos quando possível.
                """
            )
            
            Settings.llm = llm
            Settings.embed_model = embed_model
            Settings.chunk_size = CHUNK_SIZE
            Settings.chunk_overlap = CHUNK_OVERLAP
            
            storage_context = StorageContext.from_defaults(persist_dir=FAISS_INDEX_DIR)
            index = load_index_from_storage(storage_context)
            
            query_engine = index.as_query_engine(
                similarity_top_k=SIMILARITY_TOP_K,
                streaming=False,
                response_mode="compact"
            )
            print("✅ Índice carregado com sucesso!")
            return query_engine
        except Exception as e:
            print(f"⚠️  Erro ao carregar índice: {e}")
            print("🔄 Criando novo índice...")
    
    # Criar novo índice
    documents = load_documents(documents_folder)

    print("🔧 Configurando embeddings...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder="./.cache/embeddings"
    )

    print("🔧 Configurando LLM (Ollama)...")
    llm = Ollama(
        model=OLLAMA_MODEL_NAME, 
        request_timeout=60.0,
        temperature=0.1,
        additional_kwargs={
            "num_predict": MAX_TOKENS,
            "num_ctx": 2048,
        },
        system_prompt="""
Você é um assistente que responde **somente em português**, de forma clara, objetiva e direta.
Seja conciso. Não escreva em outro idioma.
Responda com base nas informações fornecidas.
Use exemplos práticos quando possível.
        """
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP

    print("🔧 Configurando FAISS...")
    vector_store = create_faiss_vector_store(embedding_dim=384)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🔧 Criando índice...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    # Salvar índice para uso futuro
    print("💾 Salvando índice...")
    index.storage_context.persist(persist_dir=FAISS_INDEX_DIR)

    query_engine = index.as_query_engine(
        similarity_top_k=SIMILARITY_TOP_K,
        streaming=False,
        response_mode="compact"
    )
    print("✅ Sistema RAG configurado e salvo!")
    return query_engine

def run_chat_loop(query_engine):
    """Loop principal do chatbot."""
    print("\n" + "="*40)
    print("🤖 Chatbot RAG — digite sua pergunta")
    print("Digite 'sair' para encerrar.")
    print("="*40 + "\n")
    
    import time
    
    while True:
        try:
            pergunta = input("👤 Você: ").strip()
            if not pergunta:
                continue
            if pergunta.lower() in EXIT_COMMANDS:
                print("👋 Encerrando.")
                break
            
            print("🔍 Processando...")
            start_time = time.time()
            
            resposta = query_engine.query(pergunta)
            
            elapsed = time.time() - start_time
            print(f"\n🤖 Bot: {resposta}")
            print(f"⏱️  Tempo: {elapsed:.2f}s\n")
        except KeyboardInterrupt:
            print("\n👋 Interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")

def main():
    """Função principal."""
    print("="*60)
    print("🤖 RAG Chatbot - Inicializando...")
    print("="*60 + "\n")
    
    # Verifica se o modelo está disponível
    print(f"📋 Verificando modelo: {OLLAMA_MODEL_NAME}")
    if not check_ollama_running():
        print("❌ Ollama não está rodando. Rode: `ollama serve`")
        sys.exit(1)
    
    # Verifica se o modelo está instalado
    try:
        import ollama
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        
        if not any(OLLAMA_MODEL_NAME in name for name in model_names):
            print(f"⚠️  Modelo '{OLLAMA_MODEL_NAME}' não encontrado.")
            print(f"📥 Baixando modelo... (isso pode levar alguns minutos)")
            ollama.pull(OLLAMA_MODEL_NAME)
            print(f"✅ Modelo '{OLLAMA_MODEL_NAME}' baixado!")
        else:
            print(f"✅ Modelo '{OLLAMA_MODEL_NAME}' disponível!")
    except Exception as e:
        print(f"⚠️  Não foi possível verificar modelo: {e}")
        print("   Continuando mesmo assim...")

    try:
        query_engine = setup_rag_system()
        run_chat_loop(query_engine)
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        raise

if __name__ == "__main__":
    main()
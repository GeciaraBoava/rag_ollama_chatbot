import os
import sys
import subprocess
import venv
import platform
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, ".venv")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")
MAIN_SCRIPT = os.path.join(BASE_DIR, "src", "main.py")

# Pastas de cache e índices
FAISS_INDEX_DIR = os.path.join(BASE_DIR, "storage")
CACHE_DIR = os.path.join(BASE_DIR, ".cache")


def create_virtualenv():
    """Cria o ambiente virtual se não existir."""
    if not os.path.exists(VENV_DIR):
        print("📦 Criando ambiente virtual (.venv)...")
        venv.create(VENV_DIR, with_pip=True)
        print("✅ Ambiente virtual criado com sucesso.")
    else:
        print("✅ Ambiente virtual já existe.")


def get_venv_paths():
    """Retorna os executáveis python e pip dentro do venv."""
    if platform.system() == "Windows":
        python_exec = os.path.join(VENV_DIR, "Scripts", "python.exe")
        pip_exec = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        python_exec = os.path.join(VENV_DIR, "bin", "python")
        pip_exec = os.path.join(VENV_DIR, "bin", "pip")
    return python_exec, pip_exec


def run_in_venv(command):
    """Executa comandos dentro do ambiente virtual."""
    python_exec, pip_exec = get_venv_paths()
    if command[0] == "python":
        command[0] = python_exec
    elif command[0] == "pip":
        command[0] = pip_exec
    subprocess.check_call(command)


def install_requirements():
    """Atualiza pip e instala dependências do projeto."""
    print("📥 Atualizando pip...")
    run_in_venv(["python", "-m", "pip", "install", "--upgrade", "pip"])
    
    print("📥 Instalando dependências (isso pode levar alguns minutos)...")
    run_in_venv(["pip", "install", "-r", REQUIREMENTS_FILE])
    
    print("✅ Dependências instaladas com sucesso.")


def check_ollama():
    """Verifica se o Ollama está rodando."""
    python_exec, _ = get_venv_paths()
    check_code = """
import requests
try:
    # Endpoint correto: /api/tags (não /api/models)
    response = requests.get('http://localhost:11434/api/tags', timeout=5)
    if response.status_code == 200:
        exit(0)
    else:
        exit(1)
except Exception:
    exit(1)
"""
    try:
        result = subprocess.run(
            [python_exec, "-c", check_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def start_ollama():
    """Tenta iniciar o Ollama em background."""
    print("🚀 Tentando iniciar Ollama...")
    
    try:
        if platform.system() == "Windows":
            # Windows: inicia em nova janela
            subprocess.Popen(
                ["ollama", "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Linux/Mac: inicia em background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        
        # Aguarda alguns segundos para o Ollama inicializar
        print("⏳ Aguardando Ollama inicializar...")
        import time
        for i in range(10):
            time.sleep(1)
            if check_ollama():
                return True
            print(f"   Tentativa {i+1}/10...")
        
        print("⚠️  Ollama não respondeu após 10 segundos")
        return False
        
    except FileNotFoundError:
        print("❌ Comando 'ollama' não encontrado no sistema")
        print("   Instale o Ollama: https://ollama.ai/download")
        return False
    except Exception as e:
        print(f"❌ Erro ao iniciar Ollama: {e}")
        return False


def check_cache_status():
    """Verifica o status do cache e índice."""
    print("\n📊 Status do Cache e Índice")
    print("="*60)
    
    if os.path.exists(FAISS_INDEX_DIR):
        # Conta arquivos no índice
        index_files = len([f for f in os.listdir(FAISS_INDEX_DIR) if os.path.isfile(os.path.join(FAISS_INDEX_DIR, f))])
        index_size = sum(os.path.getsize(os.path.join(FAISS_INDEX_DIR, f)) for f in os.listdir(FAISS_INDEX_DIR) if os.path.isfile(os.path.join(FAISS_INDEX_DIR, f)))
        print(f"✅ Índice FAISS encontrado: {index_files} arquivo(s), {index_size / (1024*1024):.2f} MB")
    else:
        print("⚠️  Índice FAISS não encontrado (será criado na primeira execução)")
    
    if os.path.exists(CACHE_DIR):
        cache_size = sum(os.path.getsize(os.path.join(dirpath, f)) 
                        for dirpath, _, filenames in os.walk(CACHE_DIR) 
                        for f in filenames)
        print(f"✅ Cache de embeddings: {cache_size / (1024*1024):.2f} MB")
    else:
        print("⚠️  Cache de embeddings não encontrado")
    
    print("="*60 + "\n")


def rebuild_faiss_index():
    """Apaga o índice FAISS antigo, se existir."""
    deleted = False
    if os.path.exists(FAISS_INDEX_DIR):
        print("🗑️  Removendo índice FAISS antigo...")
        shutil.rmtree(FAISS_INDEX_DIR)
        deleted = True
    
    if os.path.exists(CACHE_DIR):
        print("🗑️  Removendo cache de embeddings...")
        shutil.rmtree(CACHE_DIR)
        deleted = True
    
    if deleted:
        print("✅ Índice e cache removidos.")
    else:
        print("ℹ️  Nenhum índice ou cache encontrado.")


def check_python_version():
    """Verifica a versão do Python."""
    version_info = sys.version_info
    version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    print(f"🐍 Python {version_str} detectado")
    
    if version_info < (3, 9):
        print("❌ Python 3.9+ é necessário. Por favor, atualize seu Python.")
        sys.exit(1)
    elif version_info >= (3, 12):
        print("⚠️  Python 3.12+ pode ter problemas de compatibilidade com algumas bibliotecas.")
        print("   Recomendado: Python 3.11.x")


def run_main():
    """Executa o arquivo principal (src/main.py)."""
    if not os.path.exists(MAIN_SCRIPT):
        print(f"❌ Arquivo principal não encontrado: {MAIN_SCRIPT}")
        sys.exit(1)

    print("\n" + "="*60)
    print("🚀 Iniciando aplicação RAG Chatbot...")
    print("="*60 + "\n")
    
    try:
        run_in_venv(["python", MAIN_SCRIPT])
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar aplicação: {e}")
        sys.exit(1)


def main():
    rebuild_flag = "--rebuild" in sys.argv
    status_flag = "--status" in sys.argv

    print("="*60)
    print("🚀 Inicializando ambiente RAG Chatbot")
    print("="*60 + "\n")

    # Se for apenas para ver status
    if status_flag:
        check_cache_status()
        sys.exit(0)

    check_python_version()
    create_virtualenv()
    install_requirements()

    if rebuild_flag:
        rebuild_faiss_index()
    else:
        check_cache_status()

    # Verifica se Ollama está rodando
    print("\n🔍 Verificando Ollama...")
    if check_ollama():
        print("✅ Ollama está rodando corretamente.")
    else:
        print("⚠️  Ollama não está ativo.")
        
        # Tenta iniciar automaticamente
        if start_ollama():
            print("✅ Ollama iniciado com sucesso!")
        else:
            print("\n❌ Não foi possível iniciar o Ollama automaticamente.")
            print("   Por favor, inicie manualmente com: ollama serve")
            sys.exit(1)

    run_main()


if __name__ == "__main__":
    main()
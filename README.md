# 🤖 Chatbot RAG com Ollama e FAISS

Sistema de chatbot com Retrieval-Augmented Generation (RAG) que permite fazer perguntas sobre documentos locais usando IA.

---

## 🎯 Características

- 📚 Carrega documentos de uma pasta local
- 🔍 Busca semântica usando FAISS
- 🤖 Modelo de linguagem local via Ollama (DeepSeek)
- 🔒 100% local - seus dados não saem do computador
- 💬 Interface de chat interativa

---

## 🛠️ Pré-requisitos

### 1. Python 3.8 ou superior

Verifique sua versão:
```bash
python --version
```

### 2. Ollama instalado

Instale o Ollama: https://ollama.ai/download

Baixe o modelo DeepSeek:
```bash
ollama pull deepseek-coder
```

---

## 📦 Instalação

### Linux/Mac

```bash
# Clone ou baixe o projeto
cd chatbot-rag

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
source venv/bin/activate

# Instale dependências
pip install -r requirements.txt
```

### Windows

```cmd
# Clone ou baixe o projeto
cd chatbot-rag

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt
```

---

## 🚀 Uso

### 1. Adicione seus documentos

Coloque arquivos (.txt, .pdf, .docx, etc.) na pasta `documentos/`

### 2. Execute o chatbot

```bash
python src/chatbot_rag.py
```

### 3. Faça perguntas

```
Você: Qual o tema principal dos documentos?
🤖 Bot: [Resposta baseada nos documentos]

Você: sair
👋 Encerrando chatbot. Até logo!
```

---

## 📝 Formatos Suportados

- `.txt` - Arquivos de texto
- `.pdf` - Documentos PDF
- `.docx` - Documentos Word
- `.md` - Markdown
- E outros formatos compatíveis com LlamaIndex

---

## 🔧 Personalização

### Trocar o modelo de IA

Edite em `chatbot_rag.py`:

```python
OLLAMA_MODEL_NAME = "llama2"  # ou outro modelo
```

Modelos disponíveis: https://ollama.ai/library

### Trocar modelo de embeddings

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

## ⚠️ Solução de Problemas

### Erro: "Ollama not found"
```bash
# Verifique se o Ollama está rodando
ollama list
```

### Erro: "Pasta documentos vazia"
- Adicione pelo menos um arquivo na pasta `documentos/`

### Erro de memória
- Use documentos menores
- Ou considere `faiss-gpu` se tiver GPU NVIDIA

---

## 📚 Tecnologias

- **LlamaIndex** - Framework RAG
- **LangChain** - Integração com LLMs
- **FAISS** - Busca vetorial eficiente
- **Ollama** - Modelos locais de IA
- **HuggingFace** - Modelos de embeddings

---

## 📄 Licença

Este projeto é de código aberto. Use livremente!

---

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas!
```

---

## 📄 Arquivo: `setup.sh` (Linux/Mac)

```bash
#!/bin/bash

echo "🚀 Configurando Chatbot RAG..."

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Instale antes de continuar."
    exit 1
fi

# Verifica Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama não encontrado. Instale em: https://ollama.ai/download"
    exit 1
fi

# Cria ambiente virtual
echo "📦 Criando ambiente virtual..."
python3 -m venv venv

# Ativa ambiente
echo "✅ Ativando ambiente virtual..."
source venv/bin/activate

# Instala dependências
echo "📥 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

# Cria pasta documentos
mkdir -p documentos
touch documentos/.gitkeep

# Baixa modelo Ollama
echo "🤖 Baixando modelo DeepSeek..."
ollama pull deepseek-coder

echo ""
echo "✅ Instalação concluída!"
echo ""
echo "Para usar:"
echo "  1. source venv/bin/activate"
echo "  2. Adicione documentos na pasta 'documentos/'"
echo "  3. python src/chatbot_rag.py"
```

---

## 🎯 Próximos Passos

1. **Crie a estrutura de pastas** conforme o diagrama
2. **Copie o código Python** para `src/chatbot_rag.py`
3. **Crie os arquivos de configuração** (requirements.txt, .gitignore, README.md)
4. **Execute a instalação**
5. **Adicione documentos** e teste o chatbot

---

## 💡 Dicas de Uso

- **Documentos menores** = respostas mais rápidas
- **Perguntas específicas** = respostas mais precisas
- **Múltiplos arquivos** = contexto mais rico
- **Organize por tema** = melhor organização

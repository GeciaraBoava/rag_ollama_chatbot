# RAG Chatbot com Ollama e LlamaIndex

Sistema de chatbot inteligente baseado em RAG (Retrieval-Augmented Generation) que permite consultar documentos locais utilizando modelos de linguagem via Ollama.

## 📋 Características

- **Processamento local**: Todos os dados permanecem na sua máquina
- **Múltiplos formatos**: Suporta PDF, DOCX e TXT
- **Alta performance**: Otimizado para respostas rápidas (2-8s)
- **Persistência**: Índice vetorial salvo para reutilização
- **Interface CLI**: Interação via terminal

## 🔧 Requisitos

### Software necessário

- **Python 3.11.9** (ou superior)
- **Ollama** instalado e configurado
- **Git** (opcional, para clonar o repositório)

### Instalação do Ollama

#### Windows/Mac/Linux
```bash
# Acesse: https://ollama.ai/download
# Ou use o instalador oficial para seu sistema operacional
```

Após instalar, inicie o servidor:
```bash
ollama serve
```

## 🚀 Instalação

### 1. Clone o repositório
```bash
git clone <url-do-repositorio>
cd rag_ollama_chatbot
```

### 2. Estrutura de pastas
Crie a pasta para seus documentos:
```bash
mkdir documentos
```

A estrutura final deve ser:
```
rag_ollama_chatbot/
├── src/
│   └── main.py
├── documentos/          # Seus arquivos PDF, DOCX, TXT
├── requirements.txt
├── setup.py
└── README.md
```

### 3. Configure o ambiente
Execute o script de setup que criará o ambiente virtual e instalará as dependências:

```bash
python setup.py
```

O script irá:
- Criar ambiente virtual (`.venv`)
- Instalar todas as dependências
- Verificar se o Ollama está ativo
- Baixar o modelo `llama3.2:3b` automaticamente (primeira execução)

## 📚 Preparação dos Documentos

1. Adicione seus documentos na pasta `documentos/`
2. Formatos suportados: `.pdf`, `.docx`, `.txt`
3. Não há limite de quantidade, mas mais documentos = mais tempo de indexação inicial

Exemplo:
```
documentos/
├── manual_tecnico.pdf
├── relatorio_2024.docx
├── notas.txt
└── apresentacao.pdf
```

## 💬 Uso

### Iniciar o chatbot
```bash
python setup.py
```

### Comandos disponíveis
- Digite sua pergunta e pressione Enter
- `sair`, `exit` ou `quit` para encerrar
- `Ctrl+C` para interromper

### Exemplo de interação
```
🤖 Chatbot RAG — digite sua pergunta
========================================

Você: Qual o conteúdo principal do manual técnico?
🔍 Processando...

🤖 Bot: O manual técnico aborda os seguintes tópicos principais:
1. Instalação do sistema
2. Configuração inicial
3. Manutenção preventiva
⏱️  Tempo: 4.23s

Você: sair
👋 Encerrando.
```

## 🔄 Reconstruir Índice

Se você adicionar, remover ou modificar documentos, reconstrua o índice:

```bash
python setup.py --rebuild
```

Isso irá:
- Remover índice FAISS antigo
- Limpar cache de embeddings
- Reprocessar todos os documentos

## ⚙️ Configurações Avançadas

### Alterar modelo LLM

Edite `src/main.py` e modifique:
```python
OLLAMA_MODEL_NAME = "llama3.2:3b"  # Modelo padrão (rápido)
```

Outros modelos disponíveis:
```bash
ollama pull qwen2.5:3b    # Alternativa rápida
ollama pull phi3:mini     # Ultra compacto
ollama pull llama3.2:1b   # Mais rápido (menos preciso)
```

### Ajustar parâmetros de busca

Em `src/main.py`:
```python
CHUNK_SIZE = 256          # Tamanho dos chunks de texto
SIMILARITY_TOP_K = 2      # Quantidade de chunks recuperados
MAX_TOKENS = 512          # Tamanho máximo da resposta
```

### Usar GPU (se disponível)

Modifique o embedding model:
```python
embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,
    device="cuda",  # Usar GPU
    cache_folder="./.cache/embeddings"
)
```

## 📊 Performance

| Cenário | Tempo Esperado |
|---------|----------------|
| Primeira indexação | 30s - 2min (depende da quantidade de documentos) |
| Carregamento de índice existente | 2-5s |
| Resposta a pergunta | 3-8s |

## 🐛 Solução de Problemas

### Erro: "Ollama não está ativo"
```bash
# Inicie o servidor Ollama
ollama serve
```

### Erro: "Modelo não encontrado"
```bash
# Baixe o modelo manualmente
ollama pull llama3.2:3b
```

### Erro: "ImportError: cannot import name..."
```bash
# Reinstale as dependências
python setup.py
```

### Respostas lentas
- Use um modelo menor: `llama3.2:1b`
- Reduza `SIMILARITY_TOP_K` para 1
- Verifique se há muitos documentos (considere dividir em categorias)

### Pasta `documentos/` vazia
```bash
# O sistema irá gerar erro. Adicione pelo menos um documento antes de executar
```

## 🏗️ Arquitetura

```
┌─────────────────┐
│   Documentos    │
│  (PDF/DOCX/TXT) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processamento  │
│  (Chunking +    │
│   Embeddings)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FAISS Vector   │
│     Store       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────┐
│  Query Engine   │────▶│   Ollama    │
│  (Retrieval)    │     │    LLM      │
└─────────────────┘     └─────────────┘
         │
         ▼
┌─────────────────┐
│    Resposta     │
└─────────────────┘
```

## 📦 Dependências Principais

- **llama-index**: Framework RAG
- **ollama**: Cliente Python para Ollama
- **faiss-cpu**: Busca vetorial eficiente
- **sentence-transformers**: Geração de embeddings
- **pypdf/python-docx**: Leitura de documentos

## 📝 Licença

Este projeto é fornecido como está, sem garantias. Use por sua conta e risco.

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas. Abra uma issue ou pull request.

## 📧 Suporte

Para problemas técnicos:
1. Verifique a seção "Solução de Problemas"
2. Consulte a documentação do Ollama: https://ollama.ai/docs
3. Consulte a documentação do LlamaIndex: https://docs.llamaindex.ai

---

**Versão**: 1.0.0  
**Última atualização**: Outubro 2025
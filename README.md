# Assistente RAG — Consulta inteligente a documentos PDF

Um assistente de perguntas e respostas baseado em **Retrieval-Augmented Generation (RAG)**, desenvolvido em Python. A aplicação permite enviar documentos PDF, extrair seu conteúdo, armazená-lo em um banco vetorial e realizar consultas em linguagem natural, utilizando o contexto recuperado para gerar respostas fundamentadas nos documentos.

O projeto utiliza a API do Google Gemini para geração de embeddings e respostas, ChromaDB para armazenamento vetorial e Streamlit para a interface.

## Objetivo do projeto

Este projeto foi desenvolvido com o objetivo de aplicar conceitos de Inteligência Artificial Generativa, processamento de documentos, embeddings, busca semântica e bancos de dados vetoriais em uma aplicação prática.

Além de explorar o funcionamento de arquiteturas RAG, o projeto busca oferecer uma maneira simples de consultar documentos extensos por meio de perguntas em linguagem natural.

## Funcionalidades

- **Upload de documentos:** envio de um ou vários arquivos PDF.
- **Validação de arquivos:** verificação de formato, tamanho, integridade e quantidade de páginas.
- **Extração de texto:** leitura do conteúdo dos PDFs com PyMuPDF.
- **OCR:** reconhecimento de texto em documentos digitalizados utilizando Tesseract.
- **Divisão em chunks:** fragmentação do conteúdo em trechos menores para facilitar a recuperação de informações.
- **Geração de embeddings:** representação vetorial dos textos utilizando Gemini.
- **Armazenamento vetorial:** persistência dos documentos e embeddings no ChromaDB.
- **Busca semântica:** recuperação dos trechos mais relevantes para cada pergunta.
- **Geração de respostas:** utilização do Gemini para responder com base no contexto recuperado.
- **Referências às fontes:** indicação do documento e da página utilizados na resposta.
- **Histórico de conversa:** visualização das perguntas e respostas durante a sessão.
- **Detecção de duplicados:** identificação de documentos com conteúdo já armazenado por meio de hash SHA-256.

## Tecnologias utilizadas

| Tecnologia | Finalidade |
|---|---|
| Python | Linguagem principal |
| Streamlit | Interface web |
| Google Gemini API | Embeddings e geração de respostas |
| ChromaDB | Banco de dados vetorial |
| PyMuPDF | Extração de texto de PDFs |
| Tesseract OCR | Reconhecimento de texto em documentos digitalizados |
| Pytesseract | Integração entre Python e Tesseract |
| Pillow | Manipulação de imagens |
| python-dotenv | Gerenciamento de variáveis de ambiente |

## Arquitetura do projeto

```text
assistente-rag/
│
├── app.py
├── requirements.txt
├── .env
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── documents/
│
├── chroma_db/
│
└── src/
    ├── document_loader.py
    ├── text_splitter.py
    ├── embeddings.py
    ├── vector_store.py
    ├── generator.py
    └── document_processor.py
```

### Organização dos módulos

- `app.py`: aplicação principal, interface Streamlit, upload de documentos e interação com o usuário.
- `document_loader.py`: extração de texto dos PDFs e aplicação de OCR quando necessário.
- `text_splitter.py`: divisão do conteúdo extraído em chunks.
- `embeddings.py`: geração dos embeddings utilizando a API Gemini.
- `vector_store.py`: armazenamento e recuperação de informações no ChromaDB.
- `generator.py`: construção do contexto e geração das respostas com o Gemini.
- `document_processor.py`: integração das etapas de processamento dos documentos.

## Como funciona

O funcionamento do assistente é dividido em duas etapas principais.

### 1. Processamento dos documentos

1. O usuário envia um ou mais arquivos PDF.
2. A aplicação valida os arquivos recebidos.
3. O conteúdo textual é extraído utilizando PyMuPDF. Quando necessário, o OCR é utilizado para reconhecer textos em páginas digitalizadas.
4. O conteúdo é dividido em chunks.
5. São gerados embeddings para representar semanticamente cada trecho.
6. Os textos, vetores e metadados são armazenados no ChromaDB.

### 2. Consulta e geração de respostas

1. O usuário envia uma pergunta pela interface.
2. A pergunta é convertida em um embedding.
3. O ChromaDB realiza uma busca por similaridade e recupera os trechos mais relevantes.
4. Os trechos recuperados são enviados como contexto para o modelo Gemini.
5. O modelo gera uma resposta fundamentada nas informações disponíveis.
6. A aplicação apresenta a resposta e suas respectivas referências aos documentos e páginas.

## Requisitos

Antes de executar o projeto, é necessário ter:

- Python instalado.
- Uma chave de API do Google Gemini.
- Tesseract OCR instalado para o reconhecimento de texto em PDFs digitalizados.

A chave de API pode ser obtida no [Google AI Studio](https://aistudio.google.com/).

O Tesseract OCR pode ser obtido em seu [repositório oficial](https://github.com/tesseract-ocr/tesseract).

Para o reconhecimento de documentos em português, é necessário disponibilizar o pacote de idioma `por` do Tesseract.

## Instalação e execução

### 1. Clone o repositório

```bash
git clone URL_DO_REPOSITORIO
```

Entre na pasta do projeto:

```bash
cd assistente-rag
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows:

```powershell
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure a chave da API

Crie um arquivo `.env` na raiz do projeto e adicione sua chave:

```env
GEMINI_API_KEY=sua_chave_aqui
```

**Importante:** não compartilhe sua chave de API nem envie o arquivo `.env` para o GitHub.

### 5. Execute a aplicação

```bash
streamlit run app.py
```

Após a inicialização, o Streamlit disponibilizará o endereço local da aplicação no terminal.

## Armazenamento dos documentos

Os documentos processados e seus respectivos embeddings são armazenados localmente no diretório `chroma_db/`, utilizando o recurso de persistência do ChromaDB.

Isso permite que os documentos continuem disponíveis em execuções posteriores da aplicação, sem a necessidade de processá-los novamente.

Os arquivos PDF enviados são armazenados em `data/documents/`.

Esses diretórios contêm dados locais e não devem ser enviados ao repositório público.

## Segurança e limitações

- A chave da API é configurada por variável de ambiente.
- A aplicação valida os arquivos PDF antes do processamento.
- As respostas são orientadas pelo contexto recuperado, buscando reduzir respostas inventadas ou sem fundamentação documental.
- A qualidade das respostas depende da qualidade do documento, da extração textual e da relevância dos trechos recuperados.
- O OCR pode apresentar imprecisões em documentos com baixa resolução, tabelas ou formatações complexas.
- O histórico de conversa é mantido durante a sessão da aplicação, mas não é persistido como histórico permanente.
- O banco vetorial é local e compartilhado entre as sessões que utilizam a mesma instalação.

## Demonstração

![Interface do Assistente RAG](assets/assistente-rag.png)

![Vídeo demonstrativo do Assistente RAG](https://youtu.be/z8UHmTgUR4M)
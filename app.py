import os
from pathlib import Path

import pymupdf
import logging

import streamlit as st
from dotenv import load_dotenv
from google import genai

from src.document_processor import processar_documento
from src.vector_store import conectar_banco, buscar_chunks
from src.embeddings import gerar_embedding
from src.generator import gerar_resposta

# Configuração do registro de erros
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validar_pdf(nome_arquivo, conteudo):
    """
    Valida o formato, tamanho e estrutura de um arquivo PDF.
    """
    if not nome_arquivo.lower().endswith(".pdf"):
        return False, "O arquivo precisa estar no formato PDF."

    tamanho_maximo = 20 * 1024 * 1024

    if len(conteudo) > tamanho_maximo:
        return False, "O arquivo ultrapassa o limite de 20 MB."

    try:
        with pymupdf.open(
            stream=conteudo,
            filetype="pdf"
        ) as documento:

            if documento.is_encrypted:
                return False, "PDF protegido por senha não é permitido."

            if documento.page_count == 0:
                return False, "O PDF não possui páginas."

            if documento.page_count > 100:
                return False, "O PDF ultrapassa o limite de 100 páginas."

    except Exception:
        return False, "O arquivo não é um PDF válido ou está corrompido."

    return True, "PDF válido."

st.set_page_config(
    page_title="Assistente RAG",
    page_icon=":material/library_books:",
    layout="wide"
)


# Cabeçalho da aplicação
st.title(":material/library_books: Assistente RAG")

st.write(
    "Envie seus documentos PDF e faça perguntas "
    "com base no conteúdo deles."
)

st.divider()


# Área de envio de documentos
st.subheader(":material/document_search: Documentos")

LIMITE_PDF_MB = 10
LIMITE_PDF_BYTES = LIMITE_PDF_MB * 1024 * 1024

arquivos = st.file_uploader(
    "Selecione um ou mais arquivos PDF",
    type=["pdf"],
    accept_multiple_files=True
)


# Exibe os documentos selecionados
if arquivos:
    st.success(f"{len(arquivos)} documento(s) selecionado(s).")

    for arquivo in arquivos:
        tamanho_kb = arquivo.size / 1024

        if arquivo.size > LIMITE_PDF_BYTES:
            st.error(
                f"**{arquivo.name}** — {tamanho_kb:.2f} KB. "
                f"Arquivo excede o limite de {LIMITE_PDF_MB} MB."
            )
        else:
            st.write(
                f":material/attach_file: **{arquivo.name}** — "
                f"{tamanho_kb:.2f} KB"
            )

else:
    st.info("Selecione arquivos PDF para começar.")

arquivos_validos = (
    bool(arquivos)
    and all(
        arquivo.size <= LIMITE_PDF_BYTES
        for arquivo in arquivos
    )
)


# Botão para processar os documentos
if st.button(
    "Processar documentos",
    icon=":material/cloud_upload:",
    disabled=not arquivos_validos
):

    pasta_documentos = Path("data/documents")
    pasta_documentos.mkdir(parents=True, exist_ok=True)

    client = None

    try:
        # Carrega a chave da API
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Chave GEMINI_API_KEY não encontrada."
            )

        # Conecta ao Gemini e ao ChromaDB
        client = genai.Client(api_key=api_key)

        _, colecao = conectar_banco()

        total_processados = 0

        for arquivo in arquivos:
            nome_documento = Path(arquivo.name).name

            caminho_pdf = pasta_documentos / nome_documento

            conteudo = arquivo.getvalue()

            # Valida o arquivo antes de salvá-lo
            valido, mensagem = validar_pdf(
                nome_arquivo=nome_documento,
                conteudo=conteudo
            )

            if not valido:
                st.error(f"{nome_documento}: {mensagem}")
                continue

            # Evita sobrescrever um documento diferente
            if caminho_pdf.exists():

                conteudo_existente = caminho_pdf.read_bytes()

                if conteudo_existente != conteudo:
                    st.error(
                        f"Já existe um arquivo chamado "
                        f"{nome_documento}, mas seu conteúdo "
                        "é diferente. Renomeie o novo arquivo "
                        "antes de enviá-lo."
                    )
                    continue

            else:
                caminho_pdf.write_bytes(conteudo)

            # Processa o documento
            with st.spinner(
                f"Processando {nome_documento}..."
            ):
                resultado = processar_documento(
                    caminho_pdf=caminho_pdf,
                    colecao=colecao,
                    client=client
                )

            total_processados += 1

            if resultado["total_chunks"] == 0:
                st.warning(
                    f"{nome_documento}: nenhum texto encontrado."
                )

            elif resultado["novos_chunks"] == 0:
                st.info(
                    f"{nome_documento}: documento já processado. "
                    "Nenhum novo embedding foi gerado."
                )

            else:
                st.success(
                    f"{nome_documento}: "
                    f"{resultado['novos_chunks']} novos chunks "
                    "armazenados."
                )

        st.divider()

        st.success("Processamento finalizado!")

        st.write(
            f"Total de documentos processados: "
            f"{total_processados}"
        )

        st.write(
            f"Total de chunks no ChromaDB: {colecao.count()}"
        )

    except Exception:
        logging.exception("Erro ao processar documentos.")

        st.error(
            "Ocorreu um erro ao processar os documentos. "
            "Tente novamente."
        )

    finally:
        if client is not None:
            client.close()


st.divider()


# Área de perguntas
st.subheader(":material/comment: Converse com seus documentos")


pergunta = st.chat_input("Digite sua pergunta sobre os documentos...")

if pergunta:

    client = None
    
    try:
        # Carrega a chave da API
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "Chave GEMINI_API_KEY não encontrada."
            )

        #Conecta ao Gemini e ao ChromaDB
        client = genai.Client(api_key=api_key)

        _, colecao = conectar_banco()

        if colecao.count() == 0:
            st.warning(
                "Nenhum documento foi processado ainda."
            )

        else:
            with st.spinner("Buscando informações nos documentos..."):
                # 1. Gera o embedding da pergunta
                embedding_pergunta = gerar_embedding(
                    texto=pergunta.strip(),
                    client=client,
                    tipo_tarefa="RETRIEVAL_QUERY"
                )

                # 2. Recupera os chunks mais semelhantes
                quantidade = min(3, colecao.count())

                chunks = buscar_chunks(
                    colecao=colecao,
                    embedding_pergunta=embedding_pergunta,
                    quantidade=quantidade
                )

            if not chunks:
                st.warning(
                    "Não foram encontrados trechos relevantes."
                )
            else:
                with st.spinner("Elaborando resposta..."):
                    #3. Gera a resposta utilizando os chunks
                    resposta = gerar_resposta(
                        pergunta=pergunta.strip(),
                        chunks=chunks,
                        client=client
                    )

                #4. Exibe a resposta
                st.subheader("Resposta")
                st.markdown(resposta)

                #Exibe os trechos utilizados
                with st.expander(
                    "Consultar trechos utilizados como contexto"
                ):
                    for i, chunk in enumerate(chunks, start=1):
                        documento = chunk["metadados"]["documento"]
                        pagina = chunk["metadados"]["pagina"]

                        st.markdown(
                            f"**Trecho {i} — {documento} "
                            f"(Página {pagina})**"
                        )

                        st.write(chunk["texto"])

                        st.divider()

    except Exception:
        logging.exception("Erro ao consultar documentos.")

        st.error(
            "Ocorreu um erro ao consultar os documentos. "
            "Tente novamente."
        )

    finally:
        if client is not None:
            client.close()
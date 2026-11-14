
import hashlib

from pathlib import Path

from src.document_loader import extrair_texto_pdf
from src.text_splitter import dividir_texto
from src.embeddings import gerar_embeddings
from src.vector_store import armazenar_chunks


def processar_documento(caminho_pdf, colecao, client):
    """
    Processa um PDF, gera embeddings dos chunks novos
    e armazena os resultados no ChromaDB.
    """

    caminho_pdf = Path(caminho_pdf)

    if not caminho_pdf.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {caminho_pdf}"
        )

    nome_documento = caminho_pdf.name

    # Calcula o hash SHA-256 do documento
    conteudo = caminho_pdf.read_bytes()
    hash_documento = hashlib.sha256(conteudo).hexdigest()

    # Verifica se o mesmo conteúdo já está armazenado
    documentos_existentes = colecao.get(
        where={"hash_sha256": hash_documento},
        include=["metadatas"]
    )

    if documentos_existentes["ids"]:
        documento_original = (
            documentos_existentes["metadatas"][0]["documento"]
        )

        return {
            "documento": nome_documento,
            "total_chunks": 0,
            "novos_chunks": 0,
            "ja_armazenados": 0,
            "duplicado": True,
            "documento_original": documento_original
        }

    # Identifica chunks anteriormente armazenados com esse nome
    registros_anteriores = colecao.get(
        where={"documento": nome_documento}
    )

    ids_anteriores = registros_anteriores["ids"]

    # 1. Extrai o texto do PDF
    paginas = extrair_texto_pdf(str(caminho_pdf))

    # 2. Divide o conteúdo em chunks
    chunks = dividir_texto(paginas)

    if not chunks:
        return {
            "documento": nome_documento,
            "total_chunks": 0,
            "novos_chunks": 0,
            "ja_armazenados": len(ids_anteriores)
        }

    # 3. Gera embeddings para todos os chunks
    resultados = gerar_embeddings(
        chunks,
        client
    )

    # 4. Atualiza os registros no ChromaDB
    total_novos = armazenar_chunks(
        colecao=colecao,
        resultados=resultados,
        documento=nome_documento,
        hash_documento=hash_documento
    )

    # 5. Remove chunks antigos que não existem mais
    ids_novos = {
        f"{nome_documento}_{chunk['chunk_id']}"
        for chunk in chunks
    }

    ids_obsoletos = [
        identificador
        for identificador in ids_anteriores
        if identificador not in ids_novos
    ]

    if ids_obsoletos:
        colecao.delete(ids=ids_obsoletos)

    return {
        "documento": nome_documento,
        "total_chunks": len(chunks),
        "novos_chunks": total_novos,
        "ja_armazenados": 0
    }

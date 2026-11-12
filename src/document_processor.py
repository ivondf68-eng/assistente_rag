
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
    nome_documento = caminho_pdf.name

    # 1. Extrai o texto do PDF
    paginas = extrair_texto_pdf(str(caminho_pdf))

    # 2. Divide o conteúdo em chunks
    chunks = dividir_texto(paginas)

    if not chunks:
        return {
            "documento": nome_documento,
            "total_chunks": 0,
            "novos_chunks": 0,
            "ja_armazenados": 0
        }

    # 3. Identifica os chunks já armazenados
    ids = [
        f"{nome_documento}_{chunk['chunk_id']}"
        for chunk in chunks
    ]

    ids_existentes = set(colecao.get(ids=ids)["ids"])

    # 4. Separa somente os chunks pendentes
    chunks_pendentes = [
        chunk for chunk in chunks
        if f"{nome_documento}_{chunk['chunk_id']}"
        not in ids_existentes
    ]

    total_novos = 0

    # 5. Gera embeddings somente para os chunks novos
    if chunks_pendentes:
        resultados = gerar_embeddings(
            chunks_pendentes,
            client
        )

        # 6. Armazena os resultados
        total_novos = armazenar_chunks(
            colecao=colecao,
            resultados=resultados,
            documento=nome_documento
        )

    return {
        "documento": nome_documento,
        "total_chunks": len(chunks),
        "novos_chunks": total_novos,
        "ja_armazenados": len(ids_existentes)
    }

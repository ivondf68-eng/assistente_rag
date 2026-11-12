
import os
from dotenv import load_dotenv
from google import genai

from src.document_loader import extrair_texto_pdf
from src.text_splitter import dividir_texto
from src.embeddings import gerar_embeddings
from src.vector_store import conectar_banco, armazenar_chunks


caminho = "data/documents/exemplo.pdf"
nome_documento = "exemplo.pdf"

# Extrai e divide o documento
paginas = extrair_texto_pdf(caminho)
chunks = dividir_texto(paginas)

# Conecta ao ChromaDB
chroma_client, colecao = conectar_banco()

# Identifica os chunks que já estão armazenados
ids = [
    f"{nome_documento}_{chunk['chunk_id']}"
    for chunk in chunks
]

ids_existentes = set(colecao.get(ids=ids)["ids"])

# Mantém somente os chunks ainda não armazenados
chunks_pendentes = [
    chunk for chunk in chunks
    if f"{nome_documento}_{chunk['chunk_id']}" not in ids_existentes
]

print(f"Total de chunks: {len(chunks)}")
print(f"Chunks já armazenados: {len(ids_existentes)}")
print(f"Chunks pendentes: {len(chunks_pendentes)}")

if chunks_pendentes:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Chave GEMINI_API_KEY não encontrada.")

    client = genai.Client(api_key=api_key)

    try:
        print("\nGerando embeddings dos chunks pendentes...")

        resultados = gerar_embeddings(chunks_pendentes, client)

        total = armazenar_chunks(
            colecao=colecao,
            resultados=resultados,
            documento=nome_documento
        )

        print(f"Novos chunks armazenados: {total}")

    finally:
        client.close()

else:
    print("\nNenhum embedding precisa ser gerado.")

print(f"\nTotal de registros no ChromaDB: {colecao.count()}")

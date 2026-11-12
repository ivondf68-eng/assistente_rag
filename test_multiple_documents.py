
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from src.document_loader import extrair_texto_pdf
from src.text_splitter import dividir_texto
from src.embeddings import gerar_embeddings
from src.vector_store import conectar_banco, armazenar_chunks


# Localiza todos os PDFs da pasta
pasta_documentos = Path("data/documents")
arquivos_pdf = list(pasta_documentos.glob("*.pdf"))

if not arquivos_pdf:
    print("Nenhum arquivo PDF encontrado.")
    exit()

# Conecta ao ChromaDB
chroma_client, colecao = conectar_banco()

# A conexão com o Gemini será criada somente se necessária
client = None

try:
    print(f"Documentos encontrados: {len(arquivos_pdf)}")

    for caminho_pdf in arquivos_pdf:
        nome_documento = caminho_pdf.name

        print("\n" + "=" * 50)
        print(f"Processando: {nome_documento}")

        # 1. Extrai o texto
        paginas = extrair_texto_pdf(str(caminho_pdf))

        # 2. Divide o texto em chunks
        chunks = dividir_texto(paginas)

        if not chunks:
            print("Nenhum texto encontrado no documento.")
            continue

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

        print(f"Total de chunks: {len(chunks)}")
        print(f"Já armazenados: {len(ids_existentes)}")
        print(f"Pendentes: {len(chunks_pendentes)}")

        if not chunks_pendentes:
            print("Documento já processado. Pulando...")
            continue

        # 5. Gera embeddings somente quando necessário
        if client is None:
            load_dotenv()

            api_key = os.getenv("GEMINI_API_KEY")

            if not api_key:
                raise ValueError(
                    "Chave GEMINI_API_KEY não encontrada."
                )

            client = genai.Client(api_key=api_key)

        resultados = gerar_embeddings(
            chunks_pendentes,
            client
        )

        # 6. Armazena os resultados
        total = armazenar_chunks(
            colecao=colecao,
            resultados=resultados,
            documento=nome_documento
        )

        print(f"Novos chunks armazenados: {total}")

    print("\n" + "=" * 50)
    print("PROCESSAMENTO FINALIZADO!")
    print(f"Total de registros no ChromaDB: {colecao.count()}")

finally:
    if client is not None:
        client.close()

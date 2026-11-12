
import os
from dotenv import load_dotenv
from google import genai

from src.embeddings import gerar_embedding
from src.vector_store import conectar_banco, buscar_chunks


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Chave GEMINI_API_KEY não encontrada.")

client = genai.Client(api_key=api_key)

try:
    pergunta = "Quais são os principais riscos das alucinações em Inteligência Artificial?"

    print(f"Pergunta: {pergunta}\n")

    # Gera o embedding da pergunta
    embedding_pergunta = gerar_embedding(
        texto=pergunta,
        client=client,
        tipo_tarefa="RETRIEVAL_QUERY"
    )

    # Conecta ao ChromaDB
    chroma_client, colecao = conectar_banco()

    # Busca os três chunks mais semelhantes
    resultados = buscar_chunks(
        colecao=colecao,
        embedding_pergunta=embedding_pergunta,
        quantidade=3
    )

    print("Trechos encontrados:\n")

    for i, resultado in enumerate(resultados, start=1):
        print("=" * 50)
        print(f"Resultado {i}")
        print(f"Página: {resultado['metadados']['pagina']}")
        print(f"Distância: {resultado['distancia']:.4f}")
        print("\nTexto:")
        print(resultado["texto"])

finally:
    client.close()

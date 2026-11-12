
import os
from dotenv import load_dotenv
from google import genai

from src.embeddings import gerar_embedding
from src.vector_store import conectar_banco, buscar_chunks
from src.generator import gerar_resposta


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Chave GEMINI_API_KEY não encontrada.")

client = genai.Client(api_key=api_key)

try:
    pergunta = "Na rede 200.245.29.0, utilizando a máscara 255.255.255.192, quais são as sub-redes possíveis e quantos equipamentos cada uma comporta?"

    print(f"Pergunta: {pergunta}\n")

    # 1. Gera o embedding da pergunta
    embedding_pergunta = gerar_embedding(
        texto=pergunta,
        client=client,
        tipo_tarefa="RETRIEVAL_QUERY"
    )

    # 2. Recupera os trechos relevantes
    chroma_client, colecao = conectar_banco()

    chunks = buscar_chunks(
        colecao=colecao,
        embedding_pergunta=embedding_pergunta,
        quantidade=3
    )

    # 3. Gera a resposta com o Gemini
    print("Gerando resposta...\n")

    resposta = gerar_resposta(
        pergunta=pergunta,
        chunks=chunks,
        client=client
    )

    print("=" * 50)
    print("RESPOSTA DO ASSISTENTE")
    print("=" * 50)
    print(resposta)

finally:
    client.close()

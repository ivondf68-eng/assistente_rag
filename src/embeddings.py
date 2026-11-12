
from google import genai
from google.genai import types


MODELO_EMBEDDING = "gemini-embedding-001"


def gerar_embedding(texto, client, tipo_tarefa="RETRIEVAL_DOCUMENT"):
    """
    Converte um texto em um vetor numérico (embedding).
    """

    resposta = client.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config=types.EmbedContentConfig(
            task_type=tipo_tarefa
        )
    )

    return resposta.embeddings[0].values


def gerar_embeddings(chunks, client):
    """
    Gera embeddings para todos os chunks de um documento.
    Preserva os dados de identificação de cada chunk.
    """

    resultados = []

    for chunk in chunks:
        vetor = gerar_embedding(
            texto=chunk["texto"],
            client=client
        )

        resultados.append({
            "chunk_id": chunk["chunk_id"],
            "pagina": chunk["pagina"],
            "indice_chunk": chunk["indice_chunk"],
            "texto": chunk["texto"],
            "embedding": vetor
        })

    return resultados


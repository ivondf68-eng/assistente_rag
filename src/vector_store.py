
import chromadb


def conectar_banco(nome_colecao="documentos"):
    """
    Cria uma conexão persistente com o ChromaDB
    e obtém ou cria uma coleção.
    """

    cliente = chromadb.PersistentClient(
        path="./chroma_db"
    )

    colecao = cliente.get_or_create_collection(
        name=nome_colecao,
        metadata={"hnsw:space": "cosine"}
    )

    return cliente, colecao


def armazenar_chunks(colecao, resultados, documento="exemplo.pdf", hash_documento=None):
    """
    Armazena os textos, embeddings e metadados no ChromaDB.
    """

    if not resultados:
        return 0

    colecao.upsert(
        ids=[
            f"{documento}_{item['chunk_id']}"
            for item in resultados
        ],

        embeddings=[
            item["embedding"]
            for item in resultados
        ],

        documents=[
            item["texto"]
            for item in resultados
        ],

        metadatas=[
            {
                "pagina": item["pagina"],
                "indice_chunk": item["indice_chunk"],
                "documento": documento,
                "hash_sha256": hash_documento or ""
            }
            for item in resultados
        ]
    )

    return len(resultados)

def atualizar_hash_documento(colecao, documento, hash_documento):
    """
    Atualiza o hash SHA-256 nos metadados dos chunks
    de um documento já armazenado.
    """

    resultado = colecao.get(
        where={"documento": documento},
        include=["metadatas"]
    )

    if not resultado["ids"]:
        return 0

    metadados_atualizados = []

    for metadados in resultado["metadatas"]:
        metadados["hash_sha256"] = hash_documento
        metadados_atualizados.append(metadados)

    colecao.update(
        ids=resultado["ids"],
        metadatas=metadados_atualizados
    )

    return len(resultado["ids"])


def buscar_chunks(colecao, embedding_pergunta, quantidade=3):
    """
    Busca os chunks mais semelhantes à pergunta do usuário.
    """

    resultado = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=quantidade,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    chunks_encontrados = []

    for i in range(len(resultado["ids"][0])):
        chunks_encontrados.append({
            "id": resultado["ids"][0][i],
            "texto": resultado["documents"][0][i],
            "metadados": resultado["metadatas"][0][i],
            "distancia": resultado["distances"][0][i]
        })

    return chunks_encontrados


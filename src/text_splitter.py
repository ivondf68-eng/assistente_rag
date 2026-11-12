
def dividir_texto(paginas, tamanho_chunk=1000, sobreposicao=200):
    """
    Divide o texto extraído dos PDFs em chunks,
    preservando a página de origem.
    """

    if tamanho_chunk <= 0:
        raise ValueError("O tamanho do chunk deve ser maior que zero.")

    if sobreposicao < 0 or sobreposicao >= tamanho_chunk:
        raise ValueError(
            "A sobreposição deve ser maior ou igual a zero "
            "e menor que o tamanho do chunk."
        )

    chunks = []

    for pagina in paginas:
        texto = pagina["texto"].strip()
        numero_pagina = pagina["pagina"]

        if not texto:
            continue

        inicio = 0
        indice_chunk = 0

        while inicio < len(texto):

            fim = min(inicio + tamanho_chunk, len(texto))

            # Procura um espaço para evitar cortar palavras
            if fim < len(texto):
                limite = texto.rfind(" ", inicio, fim)

                if limite > inicio + tamanho_chunk * 0.7:
                    fim = limite

            trecho = texto[inicio:fim].strip()

            if trecho:
                chunks.append({
                    "chunk_id": len(chunks),
                    "pagina": numero_pagina,
                    "indice_chunk": indice_chunk,
                    "texto": trecho
                })

                indice_chunk += 1

            # Encerra quando chega ao final da página
            if fim >= len(texto):
                break

            # Avança considerando a sobreposição
            inicio = fim - sobreposicao

    return chunks

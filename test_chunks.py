from src.document_loader import extrair_texto_pdf
from src.text_splitter import dividir_texto

caminho = "data/documents/exemplo.pdf"

paginas = extrair_texto_pdf(caminho)

print(f"Total de páginas: {len(paginas)}")

chunks = dividir_texto(paginas)

for chunk in chunks:
    print("\n" + "=" * 40)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Página: {chunk['pagina']}")
    print(f"Índice: {chunk['indice_chunk']}")
    print(f"Caracteres: {len(chunk['texto'])}")
    print(chunk["texto"])

print(f"\nTotal de chunks: {len(chunks)}")
print("Teste finalizado!")
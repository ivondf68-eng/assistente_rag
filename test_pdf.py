from src.document_loader import extrair_texto_pdf

caminho = "data/documents/ferramenta_idosos.pdf"

paginas = extrair_texto_pdf(caminho)

for pagina in paginas:
    print(f"\n--- Página {pagina['pagina']} ---")
    print(pagina["texto"])

print(f"\nTotal de páginas: {len(paginas)}")
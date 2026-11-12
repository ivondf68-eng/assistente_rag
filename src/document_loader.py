import io

import pymupdf
import pytesseract

from PIL import Image


def extrair_texto_pdf(caminho_pdf):
    """
    Extrai o texto de cada página de um PDF.

    Primeiro tenta a extração convencional com PyMuPDF.
    Caso não encontre texto, utiliza OCR com Tesseract.
    """

    paginas = []

    with pymupdf.open(caminho_pdf) as documento:

        for numero, pagina in enumerate(documento, start=1):

            # 1. Tenta extrair o texto diretamente
            texto = pagina.get_text().strip()

            # 2. Se não encontrar texto, utiliza OCR
            if not texto:

                # Renderiza a página em uma imagem de 300 DPI
                pixmap = pagina.get_pixmap(
                    dpi=300,
                    alpha=False
                )

                imagem = Image.open(
                    io.BytesIO(pixmap.tobytes("png"))
                )

                # Reconhece o texto em português
                texto = pytesseract.image_to_string(
                    imagem,
                    lang="por"
                ).strip()

            paginas.append({
                "pagina": numero,
                "texto": texto
            })

    return paginas

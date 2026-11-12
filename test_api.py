import os 
from dotenv import load_dotenv 
from google import genai

#Carrega as variáveis de ambiente do .env
load_dotenv()

#Recupera a chave da API
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("A chave da API não foi encontrada.")

#inicializa o cliente do Gemini
client = genai.Client(api_key=api_key)

try:
    print("Conectando a API do Gemini...\n")

    response = client.models.generate_content(
        model = "gemini-3.5-flash-lite",
        contents = "Explique brevemente o que é IA."
    )

    print("Resposta do Gemini:\n")
    print(response.text)

    print("\nConexão realizada com sucesso!")

except Exception as error:
    print(f"Erro ao conectar à API do Gemini: {error}")

finally:
    cliente.close()
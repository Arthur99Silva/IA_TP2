from transformers import pipeline

# Carrega pipeline do modelo distilBERT
print("Carregando o modelo de análise de sentimentos...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Modelo carregado com sucesso!")

print("\n--- Iniciando os Testes ---")

# Exemplo da Amazon de uma avaliação positiva
review_positiva = "This is a fantastic product! I am very happy with my purchase and would recommend it to everyone."
resultado_pos = classifier(review_positiva)
print(f"Avaliação: '{review_positiva}'")
print(f"Resultado do Modelo: {resultado_pos}")

print("-" * 30)

# Exemplo da Amazon de uma avaliação negativa
review_negativa = "Worst purchase of my life. The item broke after just one use. It's a complete waste of money."
resultado_neg = classifier(review_negativa)
print(f"Avaliação: '{review_negativa}'")
print(f"Resultado do Modelo: {resultado_neg}")
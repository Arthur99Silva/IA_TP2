import gradio as gr
from transformers import pipeline
from datasets import load_dataset
# Carregando distilBERT
print("Carregando o modelo de análise de sentimentos...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
print("Modelo carregado com sucesso!")

# Carregando amazon_polarity para transfer learning e testa com um exemplo
print("\nCarregando o dataset amazon_polarity...")
try:
    dataset = load_dataset("amazon_polarity")
    print("Dataset carregado com sucesso!")

    print("\n--- Testando o modelo com um exemplo do dataset ---")
    exemplo_teste = dataset['test'][0]
    texto_exemplo = exemplo_teste['content']
    sentimento_real = 'POSITIVO' if exemplo_teste['label'] == 1 else 'NEGATIVO'

    print(f"Texto da avaliação: '{texto_exemplo}'")
    print(f"Sentimento Real (do dataset): {sentimento_real}")

    # Analisa o exemplo
    resultado_exemplo = classifier(texto_exemplo)[0]
    label_modelo = 'POSITIVO' if resultado_exemplo['label'] == 'POSITIVE' else 'NEGATIVO'
    score_modelo = resultado_exemplo['score']

    print(f"Resultado do Modelo: {label_modelo} com confiança de {score_modelo:.2%}")
    print("-------------------------------------------------")

except Exception as e:
    print(f"Falha ao carregar o dataset: {e}")


# Interface p/ o modelo distilBERT
def analisar_sentimento(texto_da_avaliacao):
    if not texto_da_avaliacao:
        return "Por favor, insira um texto para ser analisado."

    resultado = classifier(texto_da_avaliacao)[0]
    label = resultado['label']
    score = resultado['score']

    if label == 'POSITIVE':
        label_traduzido = 'POSITIVO 😊'
    else:
        label_traduzido = 'NEGATIVO 😠'

    return f"Sentimento: {label_traduzido}\nConfiança: {score:.2%}"


# Interface
print("\nIniciando a interface web com Gradio...")

iface = gr.Interface(
    fn=analisar_sentimento,
    inputs=gr.Textbox(
        lines=8,
        label="Escreva a Avaliação de um Produto ou Serviço em inglês",
        placeholder="Ex: 'Very good product!'"
    ),
    outputs=gr.Textbox(
        label="Resultado da Análise"
    ),
    title="🤖 Analisador de Sentimentos com IA",
    description="Escreva um texto no campo abaixo e o modelo de Inteligência Artificial (DistilBERT) irá classificá-lo como Positivo ou Negativo."
)

iface.launch()
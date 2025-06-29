import gradio as gr
from transformers import pipeline

# CARREGANDO O MODELO
print("Carregando o modelo de análise de sentimentos...")
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
) # 
print("Modelo carregado com sucesso!")


# Interface
def analisar_sentimento(texto_da_avaliacao):
    # TExto inserido
    if not texto_da_avaliacao:
        return "Por favor, insira um texto para ser analisado."

    # Analisa o texto
    resultado = classifier(texto_da_avaliacao)[0]
    label = resultado['label']
    score = resultado['score']

    if label == 'POSITIVE':
        label_traduzido = 'POSITIVO 😊'
    else:
        label_traduzido = 'NEGATIVO 😠'

    return f"Sentimento: {label_traduzido}\nConfiança: {score:.2%}"


# Gradio
print("\nIniciando a interface web com Gradio...")

iface = gr.Interface(
    fn=analisar_sentimento,
    inputs=gr.Textbox(
        lines=8,
        label="Escreva a Avaliação de um Produto ou Serviço",
        placeholder="Ex: 'Adorei o produto, a entrega foi muito rápida e a qualidade é excelente!'"
    ),
    outputs=gr.Textbox(
        label="Resultado da Análise"
    ),
    title="🤖 Analisador de Sentimentos com IA",
    description="Escreva um texto no campo abaixo e o modelo de Inteligência Artificial (DistilBERT) irá classificá-lo como Positivo ou Negativo."
)

iface.launch()
from datasets import load_dataset

# Carrega o dataset
dataset = load_dataset("amazon_polarity")

print("Estrutura do Dataset:")
print(dataset)

# Verificar com exemplo
print("\nExemplo de avaliação:")
exemplo_treino = dataset['train'][0]
print(exemplo_treino)
# 1 = positivo

print("\nExemplo de avaliação (Teste):")
exemplo_teste = dataset['test'][0]
print(exemplo_teste)
# 0 = negativo
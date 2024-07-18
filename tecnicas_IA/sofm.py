import pandas as pd
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

# Carregar o arquivo CSV
df = pd.read_csv('./datasets_SP/datasets_sp_filtered/sp_filtered_clean.csv')

# Remover a coluna alvo se houver
df = df.drop(columns=['id_recorrido', 'fecha_origen_recorrido', 'fecha_destino_recorrido', 'id_estacion_destino', 'Data Medicao'])

# Normalizar os dados
data = df.values.astype(float)
data = (data - np.min(data)) / (np.max(data) - np.min(data))  # normalização entre 0 e 1

# Definir o tamanho do mapa (grid)
grid_size = 20 # tamanho da grade bidimensional do mapa

# Inicializar e treinar o SOM
som = MiniSom(grid_size, grid_size, data.shape[1], sigma=1.0, learning_rate=0.5)
som.train_random(data, 1000)  # treinamento com 1000 iterações

# Mapear os dados para os neurônios vencedores
mapped = np.array([som.winner(x) for x in data])

# Adicionar os rótulos ao DataFrame original
df['Neuron_X'] = mapped[:, 0]
df['Neuron_Y'] = mapped[:, 1]

# Criar um dicionário para contar os elementos em cada neurônio
neuron_counts = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}

# Contar os elementos em cada neurônio
for neuron_x, neuron_y in zip(mapped[:, 0], mapped[:, 1]):
    neuron_counts[(neuron_x, neuron_y)] += 1

# Definir o tamanho da figura e a largura das linhas do gráfico
plt.figure(figsize=(20, 20))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Ajuste a largura das linhas
plt.colorbar()

# Adicionar marcadores para os neurônios com a contagem de elementos
for (i, j), count in neuron_counts.items():
    plt.text(i + 0.5, j + 0.5, str(count), color='black', fontsize=10, horizontalalignment='center', verticalalignment='center')

plt.title('Mapa Auto-Organizável de Características (SOFM)')
plt.show()

# # Exportar cada cluster (neurônio) para um arquivo CSV específico
# for neuron_x in range(grid_size):
#     for neuron_y in range(grid_size):
#         neuron_df = df[(df['Neuron_X'] == neuron_x) & (df['Neuron_Y'] == neuron_y)]
#         if not neuron_df.empty:
#             neuron_df.drop(columns=['Neuron_X', 'Neuron_Y'], inplace=True)  # remover colunas de neurônio
#             neuron_df.to_csv(f'neuron_{neuron_x}_{neuron_y}.csv', index=False)

print("Clusters exportados com sucesso!")

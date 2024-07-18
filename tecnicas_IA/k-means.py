import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Carregar o arquivo CSV normalizado
df = pd.read_csv('dataset_clean_normalized.csv')
df = df.drop(columns=['id_recorrido', 'fecha_origen_recorrido', 'fecha_destino_recorrido', 'id_estacion_destino', 'Data_Medicao', 'tipo_bicicleta_ELECTRIC'])

# Diretório onde os scalers foram salvos
scaler_directory = 'scalers'

# Carregar os scalers salvos
scalers = {}
columns = ["duracion_recorrido","EVAPORACAO_DO_PICHE","DIARIA(mm)","INSOLACAO_TOTAL","DIARIO(h)","PRECIPITACAO_TOTAL","DIARIO(mm)","TEMPERATURA_MAXIMA","DIARIA(°C)","TEMPERATURA_MEDIA_COMPENSADA","DIARIA(°C).1","TEMPERATURA_MINIMA","DIARIA(°C).2","UMIDADE_RELATIVA_DO_AR","MEDIA_DIARIA(%)","UMIDADE_RELATIVA_DO_AR.1","MINIMA_DIARIA(%)","VENTO","VELOCIDADE_MEDIA_DIARIA(m_por_s)"]
for col in columns:
    with open(os.path.join(scaler_directory, f'scaler_{col}.pkl'), 'rb') as file:
        scalers[col] = pickle.load(file)

# Calcular o WCSS para diferentes números de clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

# Plotar o gráfico do método do cotovelo
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

# Definir o número de clusters adequado
num_clusters = int(input("Insira o número adequado de clusters com base no gráfico do cotovelo: "))

# Inicializar o modelo k-means com o número de clusters adequado
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)

# Ajustar o modelo aos dados
kmeans.fit(df)

# Adicionar a coluna de rótulos ao DataFrame original
df['Cluster'] = kmeans.labels_

# Exportar cada cluster para um arquivo CSV específico
for cluster_num in range(num_clusters):
    cluster_df = df[df['Cluster'] == cluster_num].copy()
    for col in columns:
        cluster_df[col] = scalers[col].inverse_transform(cluster_df[[col]])
    cluster_df.to_csv(f'cluster_{cluster_num}.csv', index=False)

print("Clusters exportados com sucesso!")

# Plotar os clusters
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.drop(columns=['Cluster']))

colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

plt.figure(figsize=(10, 5))
for cluster_num in range(num_clusters):
    plt.scatter(df_pca[df['Cluster'] == cluster_num, 0], df_pca[df['Cluster'] == cluster_num, 1], 
                s=50, c=[colors[cluster_num]], label=f'Cluster {cluster_num + 1}')
plt.title('Clusters Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
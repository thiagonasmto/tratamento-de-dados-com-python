import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tratamento_dados.tratament_df import preprocess_data, save_dataframe_to_csv

# Carregar o DataFrame
file_path = './datasets_SP/datasets_sp_filtered/sp_filtered_clean.csv'
df = pd.read_csv(file_path)

df_processed, removed_percentage = preprocess_data(df, columns=["duracion_recorrido","long_estacion_origen","lat_estacion_origen","long_estacion_destino","lat_estacion_destino","EVAPORACAO DO PICHE","DIARIA(mm)","INSOLACAO TOTAL","DIARIO(h)","PRECIPITACAO TOTAL","DIARIO(mm)","TEMPERATURA MAXIMA","DIARIA(°C)","TEMPERATURA MEDIA COMPENSADA","DIARIA(°C).1","TEMPERATURA MINIMA","DIARIA(°C).2","UMIDADE RELATIVA DO AR","MEDIA DIARIA(%)","UMIDADE RELATIVA DO AR.1","MINIMA DIARIA(%)","VENTO","VELOCIDADE MEDIA DIARIA(m/s)","genero_female","genero_male","perfil_viagem_Recreativa","perfil_viagem_Serviço","perfil_viagem_Utilitária","sexo_FEMALE","sexo_MALE","sexo_OTHER"], treatment='select_columns')

# Pré-processamento
df_numeric = df_processed.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df_numeric)

# Configurar e aplicar o T-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(df_normalized)

# Adicionar os resultados ao DataFrame
df['TSNE1'] = tsne_results[:,0]
df['TSNE2'] = tsne_results[:,1]

# Visualizar os resultados
plt.figure(figsize=(10, 7))
plt.scatter(df['TSNE1'], df['TSNE2'], c='blue', alpha=0.5)
plt.title('T-SNE')
plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.show()

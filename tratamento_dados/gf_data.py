import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('marge_trips_BikeSampa_2023_maio_dez_droped.csv')

# Padronizar os valores da coluna 'gênero'
# df['genero'] = df['genero'].str.lower().replace({'female': 'Feminino', 'FEMALE': 'Feminino',
#                                                  'male': 'Masculino', 'MALE': 'Masculino',
#                                                  'other': 'Other'})

# Padronizar os valores da coluna 'gênero'
# df['tipo_bicicleta'] = df['tipo_bicicleta'].str.lower().replace({'electric': 'Elétrica', 'mechanical': 'Mecânica'})

# Padronizar os valores da coluna 'gênero'
# df['perfil_viagem'] = df['perfil_viagem'].str.lower().replace({'electric': 'Elétrica', 'mechanical': 'Mecânica'})

# Contar as ocorrências dos gêneros
genero_counts = df['perfil_viagem'].value_counts()

# Configurações de estilo
sns.set(style='whitegrid')

def plot_grafico(tipo, titulo, numero):
    plt.figure(figsize=(10, 6))
    
    if tipo == 'barras':
        ax = genero_counts.plot(kind='bar', color=sns.color_palette('muted', len(genero_counts)))
        plt.title(f'Gráfico {numero} - {titulo}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Gênero', fontsize=12)
        plt.ylabel('Quantidade', fontsize=12)
        plt.xticks(rotation=0)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)

    elif tipo == 'pizza':
        genero_counts.plot(kind='pie', autopct='%1.1f%%', colors=sns.color_palette('muted', len(genero_counts)), startangle=90, textprops={'fontsize': 10})
        plt.title(f'Gráfico {numero} - {titulo}', fontsize=14, fontweight='bold', pad=15)
        plt.ylabel('')
        
    elif tipo == 'histograma':
        ax = df['genero'].value_counts().plot(kind='hist', bins=3, color=sns.color_palette('muted')[0])
        plt.title(f'Gráfico {numero} - {titulo}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Gênero', fontsize=12)
        plt.ylabel('Frequência', fontsize=12)
        for p in ax.patches:
            ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
    
    elif tipo == 'colunas':
        ax = genero_counts.plot(kind='barh', color=sns.color_palette('muted', len(genero_counts)))
        plt.title(f'Gráfico {numero} - {titulo}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Quantidade', fontsize=12)
        plt.ylabel('Gênero', fontsize=12)
        for p in ax.patches:
            ax.annotate(str(p.get_width()), (p.get_width(), p.get_y() + p.get_height() / 2.), ha='center', va='center', xytext=(5, 0), textcoords='offset points', fontsize=10)
    
    else:
        print("Tipo de gráfico não suportado. Escolha entre 'barras', 'pizza', 'histograma', 'colunas'.")
        return

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.figtext(0.5, 0.01, 'Fonte: Tembici, 2023', ha='center', va='center', fontsize=10)
    plt.show()

# Exemplo de uso: escolha o tipo de gráfico que deseja plotar
plot_grafico('pizza', 'Distribuição do tipo de viagem no Dataset', 3)  # você pode mudar para 'barras', 'histograma', ou 'colunas'

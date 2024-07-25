import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
sns.set()

# Função para carregar dados de um arquivo CSV
def carregar_dados(caminho_arquivo):
    dados = pd.read_csv(caminho_arquivo)
    return dados

# Função para plotar histogramas de uma ou mais colunas
def plotar_histograma(dados, colunas, unico_grafico=False):
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.histplot(dados[coluna], kde=False, label=coluna, element="step")
        plt.title('Histograma de múltiplas colunas')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.histplot(dados[coluna], kde=False)
            plt.title(f'Histograma da coluna {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')
            plt.show()

# Função para plotar gráficos KDE de uma ou mais colunas
def plotar_kde(dados, colunas, unico_grafico=False):
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.kdeplot(dados[coluna], shade=True, label=coluna)
        plt.title('Gráfico KDE de múltiplas colunas')
        plt.xlabel('Valor')
        plt.ylabel('Densidade')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(dados[coluna], fill=True)
            plt.title(f'Gráfico KDE da coluna {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Densidade')
            plt.show()

# Função para plotar combinação de histograma e KDE de uma ou mais colunas
def plotar_histograma_kde(dados, colunas, unico_grafico=False):
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.histplot(dados[coluna], kde=True, label=coluna, element="step")
        plt.title('Histograma + KDE de múltiplas colunas')
        plt.xlabel('Valor')
        plt.ylabel('Frequência / Densidade')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.histplot(dados[coluna], kde=True)
            plt.title(f'Histograma + KDE da coluna {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Frequência / Densidade')
            plt.show()

# Função para plotar rug plots de uma ou mais colunas
def plotar_rug(dados, colunas, unico_grafico=False):
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.rugplot(dados[coluna], label=coluna)
        plt.title('Rug plot de múltiplas colunas')
        plt.xlabel('Valor')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.rugplot(dados[coluna])
            plt.title(f'Rug plot da coluna {coluna}')
            plt.xlabel(coluna)
            plt.show()

def plotar_fdp_kde(dados, pares_colunas, unico_grafico=False):
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna_x, coluna_y in pares_colunas:
            sns.kdeplot(x=dados[coluna_x], y=dados[coluna_y], fill=True, label=f'{coluna_x} vs {coluna_y}')
        plt.title('Densidade de Probabilidade (FDP) via KDE de múltiplos pares de colunas')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    else:
        for coluna_x, coluna_y in pares_colunas:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(x=dados[coluna_x], y=dados[coluna_y], fill=True)
            plt.title(f'Densidade de Probabilidade (FDP) via KDE das colunas {coluna_x} e {coluna_y}')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.show()

# Nova função para plotar jointplot com estilo 'white' e tipo 'hex'
def plotar_jointplot_hex(dados, coluna_x, coluna_y):
    with sns.axes_style('white'):
        sns.jointplot(x=coluna_x, y=coluna_y, data=dados, kind='hex', cmap='viridis')
        plt.show()

def plotar_pairplot(dados, colunas=None, hue_coluna=None, tamanho=2.5): 
    if colunas is not None:
        dados = dados[colunas]
    sns.pairplot(dados, hue=hue_coluna, height=tamanho)
    plt.show()

def plotar_boxplot(dados, coluna_x, coluna_y, hue_coluna=None, estilo='ticks'):
    """
    Plota um gráfico de BoxPlot para as colunas especificadas de um DataFrame.

    :param dados: DataFrame contendo os dados.
    :param coluna_x: Nome da coluna a ser usada no eixo x.
    :param coluna_y: Nome da coluna a ser usada no eixo y.
    :param hue_coluna: Nome da coluna para ser usada como base para diferenciar as cores dos boxplots. Pode ser None.
    :param estilo: Estilo dos eixos a ser aplicado ao gráfico.
    :param tamanho: Tupla representando o tamanho da figura (largura, altura).
    """
    with sns.axes_style(style=estilo):
        g = sns.catplot(x=coluna_x, y=coluna_y, hue=hue_coluna, data=dados, kind="box")
        g.set_axis_labels(coluna_x, coluna_y)
        plt.show()

def plotar_pairgrid(dados, colunas, hue_coluna=None, palette='RdBu_r', alpha=0.8):
    """
    Plota uma grade de gráficos de dispersão (PairGrid) para as colunas especificadas de um DataFrame,
    colorindo os pontos de acordo com a coluna de categorias (hue_coluna).

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de nomes de colunas a serem incluídas no PairGrid.
    :param hue_coluna: (Opcional) Nome da coluna para diferenciar as cores dos pontos no gráfico.
    :param palette: Paleta de cores a ser usada para as categorias especificadas em hue_coluna.
    :param alpha: Transparência dos pontos no gráfico.
    """
    # Criação da Grade de Pares
    g = sns.PairGrid(dados, vars=colunas, hue=hue_coluna, palette=palette)
    g.map(plt.scatter, alpha=alpha)
    g.add_legend()
    plt.show()

def plotar_heatmap_correlacao(dados, colunas, annot=True, cmap='coolwarm'):
    """
    Calcula a matriz de correlação para as colunas selecionadas e plota um heatmap dessa matriz.

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de nomes de colunas para calcular a matriz de correlação.
    :param annot: Se True, exibe os valores de correlação no heatmap.
    :param cmap: Paleta de cores a ser usada no heatmap.
    """
    # Cálculo da matriz de correlação
    matriz_correlacao = dados[colunas].corr()
    
    # Plot do heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacao, annot=annot, cmap=cmap, fmt=".2f")
    plt.title('Heatmap da Matriz de Correlação')
    plt.show()

def plotar_violinplot(dados, coluna_x, coluna_y, hue_coluna=None, split=False, inner="quartile", palette=["lightblue", "lightpink"], estilo=None):
    """
    Plota um gráfico de violino para visualizar distribuições de uma variável contínua (coluna_y)
    em relação a uma variável categórica (coluna_x), com a opção de dividir por uma terceira variável categórica (hue_coluna).

    :param dados: DataFrame contendo os dados.
    :param coluna_x: Nome da coluna categórica para o eixo x.
    :param coluna_y: Nome da coluna contínua para o eixo y.
    :param hue_coluna: Nome da coluna para dividir os violinos (se aplicável). Pode ser None.
    :param split: Se True, divide os violinos ao longo da categoria em hue_coluna.
    :param inner: Especifica o tipo de representação interna nos violinos. Valores comuns são 'box', 'quartile', 'point', 'stick', ou None.
    :param palette: Paleta de cores a ser usada para os violinos.
    :param estilo: Estilo dos eixos a ser aplicado ao gráfico (exemplo: 'white', 'dark', 'ticks', etc.).
    """
    with sns.axes_style(style=estilo):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=coluna_x, y=coluna_y, hue=hue_coluna, data=dados, split=split, inner=inner, palette=palette)
        plt.title(f'Gráfico de Violino: {coluna_y} por {coluna_x}')
        plt.show()

def gerar_relatorio_profiling(dados, titulo="Pandas Profiling Report", arquivo_saida=None):
    """
    Gera um relatório de perfil dos dados utilizando pandas-profiling e, opcionalmente, salva o relatório como um arquivo HTML.

    :param dados: DataFrame contendo os dados a serem analisados.
    :param titulo: Título do relatório.
    :param arquivo_saida: Caminho do arquivo para salvar o relatório em formato HTML. Se None, o relatório não será salvo.
    """
    # Gerando o relatório
    profile = ProfileReport(dados, title=titulo, explorative=True)

    # Exibindo o relatório no notebook (não funcional fora de um ambiente interativo)
    profile.to_notebook_iframe()

    # Salvando o relatório em um arquivo HTML, se especificado
    if arquivo_saida:
        profile.to_file(output_file=arquivo_saida)


# Exemplo de uso das funções
if __name__ == "__main__":
    caminho_arquivo = 'heart_attack_prediction_dataset.csv'
    dados = pd.read_csv(caminho_arquivo)
    
    # Exemplo de chamadas de função
    colunas = ['Age', 'Cholesterol']  # Substitua pelas suas colunas
    pares_colunas = [('Age', 'Cholesterol'), ('Heart Rate', 'Exercise Hours Per Week')]
    
    # Plota gráficos usando as funções definidas
    # plotar_histograma(dados, colunas)
    # plotar_kde(dados, colunas)
    # plotar_histograma_kde(dados, colunas)
    # plotar_rug(dados, colunas)
    # plotar_fdp_kde(dados, pares_colunas)
    # plotar_jointplot_hex(dados, 'Age', 'Cholesterol')  # Novo exemplo de uso
    # plotar_pairplot(dados, colunas=['Age', 'Cholesterol','Heart Rate', 'Exercise Hours Per Week'], hue_coluna='Age', tamanho=2.5)

    # Exemplo de uso da função plotar_pairgrid
    # plotar_pairgrid(dados, colunas=['Age','Heart Attack Risk'], hue_coluna='Heart Attack Risk', palette='RdBu_r', alpha=0.7)
    
    # Exemplo de uso da função plotar_heatmap_correlacao
    # plotar_heatmap_correlacao(dados, colunas=['Age','Heart Attack Risk'])

    # Exemplo de uso da função plotar_violinplot
    # plotar_violinplot(dados, coluna_x='Diet', coluna_y='Age', hue_coluna='Sex', split=True, inner="quartile", palette=["lightblue", "lightpink"], estilo='white')

    # Gerar relatório de profiling e salvar como HTML
    gerar_relatorio_profiling(dados, titulo="Relatório de Profiling", arquivo_saida="Relatório do dataset")
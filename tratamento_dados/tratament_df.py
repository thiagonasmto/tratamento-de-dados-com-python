import pandas as pd
from datetime import datetime
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
import pickle

def statistics_results(df, nome_do_arquivo, caminho):
    statistics = df.describe(include='all')
    statistics = statistics.transpose()
    save_dataframe_to_csv(statistics, "statistics_dataframe_"+nome_do_arquivo, caminho)
    print("Estatísticas calculadas e salvas com sucesso em", caminho+"statistics_dataframe_"+nome_do_arquivo)

def converter_data(data):
    try:
        return datetime.strptime(data, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
    except ValueError:
        return None

def save_dataframe_to_csv(df, nome_arquivo, caminho):
    caminho_completo = caminho + nome_arquivo
    df.to_csv(caminho_completo, index=False)
    print(f"DataFrame salvo como '{nome_arquivo}' em '{caminho}'.")

def save_dataframe_to_excel(df, nome_arquivo, caminho):
    caminho_completo = caminho + nome_arquivo
    df.to_excel(caminho_completo, index=False)
    print(f"DataFrame salvo como '{nome_arquivo}' em '{caminho}'.")

def preprocess_data(df, columns, treatment='one-hot', fillna_value=None, impute_method=None, keep_value=None):
    df_processed = df.copy()
    removed_percentage = 0.0
    count_electric = 0
    
    for col in columns:
        if treatment == 'numeric':
            unique_values = df_processed[col].unique()
            mapping = {val: i for i, val in enumerate(unique_values)}
            df_processed[col] = df_processed[col].map(mapping)
            
        elif treatment == 'normalize':
            scaler = StandardScaler()
            df_processed[col] = scaler.fit_transform(df_processed[[col]])
            # Salvar o scaler em um arquivo pickle
            with open(f'scaler_{col}.pkl', 'wb') as file:
                pickle.dump(scaler, file)
            
        elif treatment == 'one-hot':
            df_processed = pd.get_dummies(df_processed, columns=[col])
            
        elif treatment == 'fillna':
            df_processed[col].fillna(fillna_value, inplace=True)
            
        elif treatment == 'impute':
            if impute_method == 'knn':
                imputer = KNNImputer()
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
            elif impute_method == 'mice':
                imputer = IterativeImputer()
                df_processed[col] = imputer.fit_transform(df_processed[[col]])
        
        elif treatment == 'dropna':
            initial_rows = df_processed.shape[0]
            df_processed.dropna(subset=[col], inplace=True)
            removed_rows = initial_rows - df_processed.shape[0]
            removed_percentage = (removed_rows / initial_rows) * 100
            print("Porcentagem de linhas retiradas pelo Drop na coluna '{}': {:.2f}%".format(col, removed_percentage))

        elif treatment == 'keep_nan':
            df_processed = df_processed[df_processed[columns].isnull().any(axis=1)]

        elif treatment == 'keep_value':
            initial_rows = df_processed.shape[0]
            df_processed = df_processed[df_processed[col] == keep_value]
            removed_rows = initial_rows - df_processed.shape[0]
            removed_percentage = (removed_rows * 100) / initial_rows
            print("Porcentagem de linhas retiradas pelo Keep value: {:.2f}%".format(removed_percentage))
            
        elif treatment == 'count_value':
            count_electric = df_processed[df_processed[col] == keep_value].shape[0]
            print("Quantidade de linhas com o valor 'ELECTRIC' na coluna '{}': {}".format(col, count_electric))

        elif treatment == 'count_nan':
            nan_count = df_processed[col].isnull().sum()
            total_count = len(df_processed[col])
            nan_percentage = (nan_count / total_count) * 100

            if nan_count > 0:
                print("Existem {} valores NaN na coluna '{}', que representa {:.2f}% do total.".format(nan_count, col, nan_percentage))
            else:
                print("Não existem valores NaN na coluna '{}'.".format(col))

        elif treatment == 'select_columns':
            df_processed = df_processed[columns]

        elif treatment == 'convert_date':
            df_processed[col] = df_processed[col].apply(lambda x: converter_data(x))
        
        elif treatment == 'boolean_transform':
            df_processed[col] = df_processed[col].astype(int)

        elif treatment == 'delete_column':
            df_processed.drop(columns=[col], inplace=True)
            print("A coluna '{}' foi deletada.".format(col))
            
    return df_processed, removed_percentage

import pandas as pd
import numpy as np
from preprocessamento import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

# configurações plot
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")  

def main():
    
    # 1. Carregamento dos dados
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(os.path.dirname(script_dir), 'train_mod.csv')
    df = pd.read_csv(data_file)
    print(f"Dimensões do dataset: {df.shape}")
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nInformações do dataset:")
    print(df.info())

    # 2. Análise de valores faltantes
    print("\n=== Análise de Valores Faltantes ===")
    missing_data = analisar_valores_faltantes(df)
    df_with_outliers = df.copy()

    # 3. Remoção de registros sem preço
    print("\n=== Tratamento da Variável Target ===")
    n_antes = len(df)
    df = df.dropna(subset=['Preco'])
    n_depois = len(df)
    print(f"Registros removidos por preço faltante: {n_antes - n_depois} ({((n_antes - n_depois)/n_antes)*100:.2f}%)")

    # 4. Conversão de tipos de dados
    # Convertendo Volume_motor para string e criando coluna turbo
    df['Volume_motor'] = df['Volume_motor'].astype(str)
    # Detecta turbo baseado em variações da palavra "turbo" no texto
    df['turbo'] = df['Volume_motor'].str.lower().str.contains('turbo|urbo')
    # Limpa a coluna Volume_motor removendo a parte do turbo
    df['Volume_motor'] = df['Volume_motor'].str.lower().str.replace('turbo', '').str.replace('urbo', '').str.strip()
    df['Volume_motor'] = df['Volume_motor'].str.replace(',', '.').str.strip()
    
    # Converte Volume_motor para float após limpeza
    df['Volume_motor'] = pd.to_numeric(df['Volume_motor'], errors='coerce')
    
    # Limpeza dos dados de cilindros
    print("\n=== Limpeza da coluna Cilindros ===")
    print("Valores únicos antes da limpeza:", sorted(df['Cilindros'].unique()))
    
    # Valores válidos de cilindros (1 a 16)
    cilindros_validos = list(range(1, 17))
    # Substitui valores inválidos por NaN
    df.loc[~df['Cilindros'].isin(cilindros_validos), 'Cilindros'] = np.nan
    
    print("Valores únicos após limpeza:", sorted(df['Cilindros'].dropna().unique()))
    print("Número de valores NaN em Cilindros:", df['Cilindros'].isna().sum())
    
    print("\n=== Análise da coluna Volume_motor e turbo ===")
    print("\nValores únicos em Volume_motor:")
    print(df['Volume_motor'].value_counts().head(10))
    print("\nDistribuição da coluna turbo:")
    print(df['turbo'].value_counts())
    print("\nExemplos de registros com turbo:")
    print(df[df['turbo']][['Volume_motor', 'turbo', 'Fabricante', 'Modelo']].head())
    
    # Aplicar as mesmas transformações ao df_with_outliers
    df_with_outliers['Volume_motor'] = df_with_outliers['Volume_motor'].astype(str)
    df_with_outliers['turbo'] = df_with_outliers['Volume_motor'].str.lower().str.contains('turbo|urbo')
    df_with_outliers['Volume_motor'] = df_with_outliers['Volume_motor'].str.lower().str.replace('turbo', '').str.replace('urbo', '').str.strip()
    df_with_outliers['Volume_motor'] = df_with_outliers['Volume_motor'].str.replace(',', '.').str.strip()
    df_with_outliers['Volume_motor'] = pd.to_numeric(df_with_outliers['Volume_motor'], errors='coerce')
    df_with_outliers.loc[~df_with_outliers['Cilindros'].isin(cilindros_validos), 'Cilindros'] = np.nan

    # 5. Tratamento de outliers de preço
    print("\n=== Análise de Outliers de Preço ===")
    print("Estatísticas dos preços antes da remoção de outliers:")
    print(df['Preco'].describe())
    
    precos_originais = df['Preco'].copy()
    
    Q1_preco = df['Preco'].quantile(0.25)
    Q3_preco = df['Preco'].quantile(0.75)
    IQR_preco = Q3_preco - Q1_preco
    limite_inferior_preco = Q1_preco - 1.5 * IQR_preco
    limite_superior_preco = Q3_preco + 1.5 * IQR_preco
    
    outliers_preco = df[(df['Preco'] < limite_inferior_preco) | (df['Preco'] > limite_superior_preco)]
    print(f"\nNúmero de outliers de preço detectados: {len(outliers_preco)}")
    print("\nExemplos de preços considerados outliers:")
    print(outliers_preco['Preco'].sort_values().head())
    print(outliers_preco['Preco'].sort_values().tail())
    
    n_antes = len(df)
    df = df[(df['Preco'] >= limite_inferior_preco) & (df['Preco'] <= limite_superior_preco)]
    n_depois = len(df)
    print(f"\nRegistros removidos por outliers de preço: {n_antes - n_depois} ({((n_antes - n_depois)/n_antes)*100:.2f}%)")
    
    print("\nEstatísticas dos preços após remoção de outliers:")
    print(df['Preco'].describe())
    
    print("Valores únicos de cilindros após carregar dados:", df['Cilindros'].unique())

    # plot preços
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(y=precos_originais)
    plt.title('Distribuição dos Preços\nAntes da Remoção de Outliers')
    plt.ylabel('Preço')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Preco'])
    plt.title('Distribuição dos Preços\nApós Remoção de Outliers')
    plt.ylabel('Preço')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=precos_originais, bins=50)
    plt.title('Histograma dos Preços\nAntes da Remoção de Outliers')
    plt.xlabel('Preço')
    plt.ylabel('Contagem')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=df['Preco'], bins=50)
    plt.title('Histograma dos Preços\nApós Remoção de Outliers')
    plt.xlabel('Preço')
    plt.ylabel('Contagem')
    
    plt.tight_layout()
    plt.show()

    # 6. Tratamento de inconsistências
    print("\n=== Análise de Inconsistências ===")
    analisar_inconsistencias(df)
    
    n_antes = len(df)
    df = corrigir_inconsistencias(df)
    n_depois = len(df)
    print(f"Registros removidos por inconsistências: {n_antes - n_depois} ({((n_antes - n_depois)/n_antes)*100:.2f}%)")
    
    print("\n=== Verificação após correções ===")
    analisar_inconsistencias(df)

    # 7. Tratamento de valores faltantes e outliers
    print("\n=== Tratamento de Valores Faltantes ===")
    colunas_numericas = ['Volume_motor', 'Km', 'Rodas', 'Airbags', 'Preco', 'Idade_Carro']
    df = tratar_valores_faltantes(df, strategy='median')
    
    outliers_info = detectar_outliers(df, colunas_numericas)
    
    n_antes = len(df)
    for col in colunas_numericas:
        if col != 'Preco':  # verifica apenas preço
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR)))]
    
    n_depois = len(df)
    print(f"\nRegistros removidos por outliers: {n_antes - n_depois} ({((n_antes - n_depois)/n_antes)*100:.2f}%)")
    print(f"Dimensões após remoção de outliers extremos: {df.shape}")

    # 8. Análise de correlações
    print("\n=== Análise de Correlações ===")
    # inclui turbo
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'bool']).columns
    numeric_cols = numeric_cols.drop(['Ano'])  # Removendo apenas Ano pois agora usamos Idade_Carro
    
    corr_matrix = df[numeric_cols].corr()
    
    # matriz triangular superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # plot correlações
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0, 
                fmt='.2f',
                square=True)
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()
    
    # Análise específica da correlação com turbo
    print("\nCorrelações com a coluna 'turbo':")
    turbo_corr = corr_matrix['turbo'].sort_values(ascending=False)
    print(turbo_corr)
    
    # Análise e correlação
    print("\nEstatísticas de Preço por número de Airbags:")
    preco_por_airbags = df.groupby('Airbags')['Preco'].agg(['count', 'mean', 'std']).round(2)
    print(preco_por_airbags)
    
    print("\nMédia de Idade por número de Airbags:")
    idade_por_airbags = df.groupby('Airbags')['Idade_Carro'].mean().round(2).sort_index()
    print(idade_por_airbags)
    
    print("\nEstatísticas de Preço por Categoria de Idade:")
    preco_por_idade = df.groupby('Idade_Categoria')['Preco'].describe().round(2)
    print(preco_por_idade)

    # 9. Codificação das variáveis categóricas
    print("\n=== Codificação de Variáveis Categóricas ===")
    colunas_categoricas = ['Fabricante', 'Modelo', 'Categoria', 'Combustivel', 
                          'Tipo_cambio', 'Tracao', 'Cor', 'Cilindros', 'Portas',
                          'Volume_motor', 'turbo']  # Adicionando Volume_motor e turbo às colunas categóricas
    df_encoded, encoders = codificar_categoricas(df, colunas_categoricas)

    # 10. Normalizar dados
    print("\n=== Normalização dos Dados ===")
    colunas_para_normalizar = df_encoded.select_dtypes(include=['int64', 'float64']).columns
    colunas_para_normalizar = colunas_para_normalizar.drop(['Preco', 'Ano'])  # Removendo Ano pois agora usamos Idade_Carro
    df_normalized, scaler = normalizar_dados(df_encoded, colunas_para_normalizar)
    
    if df_normalized.isnull().any().any():
        print("\nAtenção: Ainda existem valores NaN após normalização!")
        print(df_normalized.isnull().sum()[df_normalized.isnull().sum() > 0])
        df_normalized = df_normalized.fillna(df_normalized.mean())

    # 11. PCA
    print("\n=== Aplicação de PCA ===")
    X = df_normalized.drop('Preco', axis=1)
    print(f"Número de features antes do PCA: {X.shape[1]}")
    print("\nFeatures originais:")
    print(list(X.columns))
    
    data_pca, pca, pca_scaler = aplicar_pca(X, var_ratio=0.95)
    print(f"\nNúmero de componentes PCA selecionados: {pca.n_components_}")
    print(f"Percentual da variância explicada: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Análise detalhada dos componentes PCA
    feature_names = X.columns
    importancia_features = analisar_componentes_pca(X, pca, feature_names)
    
    print("\nImportância das features originais nos primeiros componentes:")
    for _, row in importancia_features.iterrows():
        print(f"{row['Feature']}: {row['Importância']:.4f}")

    # 12. salva dos dados processados
    print("\n=== Salvando Dados Processados ===")
    X_processado = df_normalized.drop('Preco', axis=1)
    y_processado = df_normalized['Preco']
    
    # Salvando os arquivos no diretório do script
    X_path = os.path.join(script_dir, 'ProcessedDatabase_SEM_outliers.csv')
    y_path = os.path.join(script_dir, 'ProcessedDatabase_target_SEM_outliers.csv')
    
    X_processado.to_csv(X_path, index=False)
    y_processado.to_csv(y_path, index=False)
    print(f"Dados processados salvos em:\n{X_path}\n{y_path}")
    
    return df_normalized, X_processado, y_processado, encoders, scaler

if __name__ == "__main__":
    try:
        print("=== Iniciando Preprocessamento dos Dados ===")
        df_normalized, X_processado, y_processado, encoders, scaler = main()
        print("\nPreprocessamento finalizado com sucesso!")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        import traceback
        print("\nDetalhes do erro:")
        print(traceback.format_exc()) 

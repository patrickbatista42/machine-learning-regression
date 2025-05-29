import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def analisar_valores_faltantes(df):
    missing = pd.DataFrame({
        'Total': df.isnull().sum(),
        'Percentual': (df.isnull().sum() / len(df)) * 100
    })
    missing = missing[missing['Total'] > 0].sort_values('Total', ascending=False)
    
    if not missing.empty:
        print("\n=== Análise de Valores Faltantes ===")
        print(missing)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Mapa de Valores Faltantes')
        plt.show()
    
    return missing

def analisar_inconsistencias(df):
    print("\n=== Análise de Inconsistências ===")
    
    for col in df.columns:
        n_unique = df[col].nunique()
        print(f"\n{col}:")
        print(f"Valores únicos: {n_unique}")
        if n_unique < 50:
            print("Valores:", df[col].unique())
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nRanges de valores numéricos:")
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"Min: {df[col].min()}")
        print(f"Max: {df[col].max()}")
        print(f"Média: {df[col].mean():.2f}")
        print(f"Mediana: {df[col].median():.2f}")

def detectar_outliers(df, colunas):
    outliers_info = {}
    
    for col in colunas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1 
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        outliers_info[col] = {
            'quantidade': len(outliers),
            'percentual': (len(outliers) / len(df)) * 100,
            'limite_inferior': lower_bound,
            'limite_superior': upper_bound
        }
        
        print(f"\nOutliers em {col}:")
        print(f"Quantidade: {len(outliers)}")
        print(f"Percentual: {(len(outliers) / len(df)) * 100:.2f}%")
        print(f"Limite inferior: {lower_bound:.2f}")
        print(f"Limite superior: {upper_bound:.2f}")
    
    return outliers_info

def analisar_correlacoes(df):
    if not all(df.dtypes.isin(['int64', 'float64'])):
        df = df.select_dtypes(include=['int64', 'float64'])
    
    corr_matrix = df.corr()
    
    if corr_matrix.isnull().any().any():
        print("\nAtenção: Existem valores NaN na matriz de correlação!")
        print("\nColunas com NaN:")
        print(corr_matrix.columns[corr_matrix.isnull().any()].tolist())
    
    zero_var_cols = df.columns[df.std() == 0]
    if len(zero_var_cols) > 0:
        print("\nAtenção: As seguintes colunas têm variância zero:")
        print(zero_var_cols.tolist())
    
    plt.figure(figsize=(12, 8))
    mask = np.zeros_like(corr_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5)
    
    plt.title('Matriz de Correlação')
    plt.tight_layout()
    plt.show()
    
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                strong_corr.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'corr': corr_matrix.iloc[i, j]
                })
    
    return strong_corr

def codificar_categoricas(df, colunas):
    df_encoded = df.copy()
    encoders = {}
    
    for col in colunas:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders

def aplicar_pca(df, n_components=None, var_ratio=0.95):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    
    if n_components is None:
        pca = PCA(n_components=var_ratio, svd_solver='full')
    else:
        pca = PCA(n_components=n_components)
    
    data_pca = pca.fit_transform(data_scaled)
    
    var_ratio = pca.explained_variance_ratio_
    var_cumsum = np.cumsum(var_ratio)
    
    print("\n=== Análise PCA ===")
    print(f"Número de features originais: {df.shape[1]}")
    print(f"Número de componentes selecionados: {pca.n_components_}")
    print(f"Variância total explicada: {sum(var_ratio):.4f}")
    
    print("\nVariância explicada por componente:")
    for i, (var, cum_var) in enumerate(zip(var_ratio, var_cumsum)):
        print(f"PC{i+1}: {var:.4f} ({cum_var:.4f} acumulado)")
    
    return data_pca, pca, scaler

def normalizar_dados(df, colunas):
    scaler = StandardScaler()
    df_normalized = df.copy()
    df_normalized[colunas] = scaler.fit_transform(df[colunas])
    return df_normalized, scaler

def tratar_valores_faltantes(df, strategy='mean'):
    df_treated = df.copy()
    
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    imputer = SimpleImputer(strategy=strategy)
    df_treated[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    if len(categorical_columns) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_treated[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    return df_treated

def corrigir_inconsistencias(df):
    df_clean = df.copy()
    
    # Correções de valores válidos
    valores_validos_cilindros = [2, 3, 4, 5, 6, 8, 10, 12]
    df_clean = df_clean[df_clean['Cilindros'].isin(valores_validos_cilindros)]
    df_clean = df_clean[df_clean['Km'] <= 1000000]
    df_clean = df_clean[df_clean['Numero_proprietarios'] <= 20]
    df_clean = df_clean[~df_clean['Categoria'].isin(['2001', '2018'])]
    df_clean = df_clean[df_clean['Rodas'] <= 24]
    df_clean = df_clean[~df_clean['Cor'].isin(['11', '23'])]
    
    # Tratamento do ano e criação de features relacionadas à idade
    ano_atual = 2024
    df_clean['Ano'] = df_clean['Ano'].fillna(df_clean['Ano'].median())
    
    # Mantendo apenas Idade_Carro e Idade_Categoria
    df_clean['Idade_Carro'] = ano_atual - df_clean['Ano'].astype(int)
    
    # Idade categorizada (mantida por ser uma representação diferente e útil)
    def categorizar_idade(idade):
        if idade <= 3:
            return 0  # Novo
        elif idade <= 7:
            return 1  # Seminovo
        elif idade <= 15:
            return 2  # Usado
        else:
            return 3  # Antigo
    
    df_clean['Idade_Categoria'] = df_clean['Idade_Carro'].apply(categorizar_idade)
    
    # Mapeamentos de padronização
    mapa_couro = {
        'Sim': 1, 'sim': 1, 'SIM': 1,
        'o': 0, 'nao': 0, 'Não': 0, 'NAO': 0,
        'SUV': 0, 'Sedan': 0
    }
    df_clean['Couro'] = df_clean['Couro'].map(mapa_couro)
    
    mapa_combustivel = {
        'Hibrido': 'HIBRIDO',
        'GASOLI': 'GASOLINA', 'Gasoli': 'GASOLINA', 'gasoli': 'GASOLINA', 'Gasol.': 'GASOLINA',
        'diesel': 'DIESEL', 'Diesel': 'DIESEL', 'DIESEL': 'DIESEL', 'Dies.': 'DIESEL',
        'Gás tural': 'GAS_NATURAL',
        'Sim': 'NAO_INFORMADO'
    }
    df_clean['Combustivel'] = df_clean['Combustivel'].map(mapa_combustivel)
    
    mapa_cambio = {
        'Automatico': 'AUTOMATICO', 'Tiptronic': 'AUTOMATICO', 'Variator': 'AUTOMATICO', '8': 'AUTOMATICO',
        'Manual': 'MANUAL', '6': 'MANUAL'
    }
    df_clean['Tipo_cambio'] = df_clean['Tipo_cambio'].map(mapa_cambio)
    
    mapa_tracao = {
        '4x4': '4X4',
        'Dianteira': 'DIANTEIRA', 'Tiptronic': 'DIANTEIRA', 'Automatico': 'DIANTEIRA',
        'Traseira': 'TRASEIRA'
    }
    df_clean['Tracao'] = df_clean['Tracao'].map(mapa_tracao)
    
    return df_clean

def analisar_componentes_pca(X, pca, feature_names):
    var_ratio = pca.explained_variance_ratio_
    var_cumsum = np.cumsum(var_ratio)
    
    # Plot da variância explicada
    plt.figure(figsize=(12, 4))
    
    # Plot variância individual
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(var_ratio) + 1), var_ratio)
    plt.xlabel('Componente Principal')
    plt.ylabel('Proporção da Variância Explicada')
    plt.title('Variância Explicada por Componente')
    plt.grid(True)
    
    # Gráfico de variância acumulada
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(var_cumsum) + 1), var_cumsum, 'bo-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Acumulada')
    plt.title('Variância Explicada Acumulada')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% da variância')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # comparacao features originais
    componentes = pd.DataFrame(pca.components_, columns=feature_names)
    
    # plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(componentes, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f')
    plt.title('Contribuição das Features para cada Componente Principal')
    plt.xlabel('Features Originais')
    plt.ylabel('Componentes Principais')
    plt.show()
    
    # Importância das features
    feature_importance = np.abs(pca.components_).mean(axis=0)
    feature_importance_normalized = feature_importance / feature_importance.sum()
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importância': feature_importance_normalized
    }).sort_values('Importância', ascending=False)
    
    # plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importância', y='Feature', data=importance_df)
    plt.title('Importância Relativa das Features Originais')
    plt.grid(True)
    plt.show()
    
    print("\n=== Análise dos Componentes PCA ===")
    print("\nVariância explicada por componente:")
    for i, v in enumerate(var_ratio):
        print(f"Componente {i+1}: {v:.4f}")
    
    print("\nVariância explicada acumulada:")
    for i, v in enumerate(var_cumsum):
        print(f"Até componente {i+1}: {v:.4f}")
    
    print("\n=== Importância relativa das features originais ===")
    print(importance_df)
    
    # Análise específica da relação entre Airbags e outras features
    print("\n=== Análise da relação entre Airbags e outras features ===")
    print("\nContribuição dos Airbags em cada componente principal:")
    airbags_contrib = pd.DataFrame({
        'Componente': [f'PC{i+1}' for i in range(len(pca.components_))],
        'Contribuição': pca.components_[:, feature_names.tolist().index('Airbags')]
    })
    print(airbags_contrib)
    
    # Verificando em quais componentes os Airbags têm maior peso
    print("\nComponentes onde Airbags têm maior contribuição:")
    for i, comp in enumerate(pca.components_):
        if abs(comp[feature_names.tolist().index('Airbags')]) > 0.3:  # threshold arbitrário
            print(f"\nComponente {i+1}:")
            contrib = pd.DataFrame({
                'Feature': feature_names,
                'Contribuição': comp
            }).sort_values('Contribuição', key=abs, ascending=False)
            print(contrib.head())
    
    return importance_df

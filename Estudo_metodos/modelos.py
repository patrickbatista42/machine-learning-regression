import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from feature_engineering import aplicar_feature_engineering
from otimizacao import otimizar_random_forest, SVROptimization, otimizar_svr_grid, comparar_otimizacoes_svr, comparar_hyperparametros_svr
from avaliacao import Validacao

class Modelo:
    def __init__(self, nome):
        self.nome = nome
        self.model = None
        self.resultado = None
    
    def treinar(self, X_train, y_train):
        raise NotImplementedError("Implemente este método na classe filha")
    
    def predizer(self, X_test):
        """Realiza predições com o modelo treinado."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        return self.model.predict(X_test)
    
    def avaliar(self, X_test, y_test):
        """Avalia o modelo com dados de teste."""
        y_pred = self.predizer(X_test)

        #usa métodos estáticos da classe validacao
        self.resultado = Validacao.avaliar_modelo(y_test, y_pred, self.nome)
        
        # Plot
        Validacao.plotar_predicoes(y_test, y_pred, self.nome)
        Validacao.plotar_residuos(y_test, y_pred, self.nome)
        
        return self.resultado
    
    def validacao_cruzada(self, X_train, y_train, cv=10):
        """Realiza validação cruzada no modelo."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='r2')
        print(f"\nResultados Cross-Validation - {self.nome}:")
        print(f"R2 médio: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores

class RegressaoLinear(Modelo):
    """Implementação do modelo de Regressão Linear."""
    
    def __init__(self):
        super().__init__("Regressão Linear")
        self.model = LinearRegression()
    
    def treinar(self, X_train, y_train):
        print(f"\n=== Modelo {self.nome} ===")
        self.model.fit(X_train, y_train)
        return self

class SVR_Model(Modelo):
    """Implementação do modelo Support Vector Regression com otimização NSGA-II ou Grid Search"""
    
    def __init__(self):
        super().__init__("SVR Otimizado")
        self.predefined_params = { # dados treinados antes com nsga para 100 individuos e 50 gerações
            'C': 999.851075,
            'epsilon': 0.981079,
            'gamma': 0.046943
        }
        self.best_params = None
        self.best_objectives = None
        self.model = None
    
    def treinar(self, X_train, y_train, X_test=None, y_test=None, ParametersSearch=3):
        """
        Treina o modelo SVR 
        seleção manual do ParametersSearch!  (por aqui)
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste (opcional, não usado)
            y_test: Target de teste (opcional, não usado)
            ParametersSearch: Método de otimização (1=NSGA-II, 2=Grid Search, 3=Comparação)
        """
        print(f"\nTreinando {self.nome}...")
        
        if ParametersSearch == 1:
            # Otimização com NSGA-II
            print("Usando NSGA-II para otimização multi-objetivo...")
            optimizer = SVROptimization(X_train, y_train)
            self.best_params, self.best_objectives = optimizer.otimizar()
            self.model = optimizer.criar_modelo(self.best_params)
            self.model.fit(X_train, y_train)
            
            print("\nMelhores parâmetros encontrados:")
            for param, value in self.best_params.items():
                print(f"{param}: {value:.6f}")
            print("\nObjetivos alcançados:")
            for metric, value in self.best_objectives.items():
                print(f"{metric}: {value:.6f}")
                
        elif ParametersSearch == 2:
            # grid search
            print("Usando Grid Search para otimização...")
            self.model, self.best_params, best_score = otimizar_svr_grid(X_train, y_train)
            
        elif ParametersSearch == 3:
            # Comparação entre parâmetros predefinidos e Grid Search usando validação cruzada
            print("Comparando modelo com parâmetros predefinidos vs Grid Search...")
            self.model, self.best_params, metrics = comparar_hyperparametros_svr(
                X_train, y_train, self.predefined_params
            )
            
            # Armazenar métricas
            self.best_objectives = {
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'MAE_std': metrics['mae_std'],
                'R2_std': metrics['r2_std']
            }
            
        else:
            raise ValueError("ParametersSearch deve ser 1 (NSGA-II), 2 (Grid Search) ou 3 (Comparação)")
            
        return self.model

    def predizer(self, X):
        """Realiza predições com o modelo treinado"""
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de fazer predições")
        return self.model.predict(X)

class RandomForest(Modelo):
    """Implementação do modelo Random Forest"""
    
    def __init__(self):
        super().__init__("Random Forest Otimizado")
        self.feature_importances = None
    
    def treinar(self, X_train, y_train):
        print(f"\n=== Modelo {self.nome} ===")
        print("\nIniciando otimização de hiperparâmetros...")
        
        self.model, melhores_params, melhor_score = otimizar_random_forest(
            X_train, y_train, cv=10, n_iter=50
        )
        self.model.fit(X_train, y_train)
        return self
    
    def plotar_importancia_features(self, feature_names):
        """Plota a importância das features do Random Forest."""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        importancia_rf = pd.DataFrame({
            'Feature': feature_names,
            'Importância': self.model.feature_importances_
        }).sort_values('Importância', ascending=False)
        
        print("\nTop 15 Features mais importantes (Random Forest Otimizado):")
        print(importancia_rf.head(15))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=importancia_rf.head(15), x='Importância', y='Feature')
        plt.title('Top 15 Features Mais Importantes - Random Forest')
        plt.xlabel('Importância Relativa')
        plt.tight_layout()
        plt.show()

def carregar_e_preparar_dados():
    """Carrega e prepara os dados para modelagem."""
    print("Carregando dados...")
    X = pd.read_csv('ProcessedDatabase_SEM_outliers.csv')
    y = pd.read_csv('ProcessedDatabase_target_SEM_outliers.csv')
    
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    
    print("\nAplicando feature engineering...")
    X_engineered, novas_features = aplicar_feature_engineering(X)
    print(f"Novas features criadas: {len(novas_features)}")
    print("Exemplos de novas features:", novas_features[:5])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, X_engineered.columns

def comparar_modelos(modelos):
    """Compara os resultados dos diferentes modelos."""
    resultados = [modelo.resultado for modelo in modelos]
    
    print("\n=== Comparação dos Modelos ===")
    comparacao = pd.DataFrame(resultados).set_index('modelo')
    print("\nMétricas comparativas:")
    print(comparacao)
    
    plt.figure(figsize=(12, 6))
    metricas = ['mse', 'rmse', 'mae', 'r2']
    
    for i, metrica in enumerate(metricas, 1):
        plt.subplot(2, 2, i)
        valores = [resultado[metrica] for resultado in resultados]
        modelos_nomes = [resultado['modelo'] for resultado in resultados]
        plt.bar(modelos_nomes, valores)
        plt.title(f'Comparação - {metrica.upper()}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return comparacao


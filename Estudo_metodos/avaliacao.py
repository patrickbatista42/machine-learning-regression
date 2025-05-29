import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns

class Validacao:
    
    @staticmethod
    def avaliar_modelo(y_true, y_pred, nome_modelo):
        """
        Avalia o modelo usando várias métricas e retorna um dicionário com os resultados.
        """
        # Converter para arrays numpy se necessário
        if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Calcular métricas
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Imprimir resultados
        print(f"\nMétricas de Avaliação - {nome_modelo}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        
        return {
            'modelo': nome_modelo,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    @staticmethod
    def plotar_predicoes(y_true, y_pred, nome_modelo):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Preços Reais')
        plt.ylabel('Preços Preditos')
        plt.title(f'Valores Reais vs Preditos - {nome_modelo}')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plotar_residuos(y_true, y_pred, nome_modelo):

        # Converter para arrays numpy se necessário
        if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
            y_true = y_true.values.ravel()
        if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
            y_pred = y_pred.values.ravel()
        
        residuos = y_true - y_pred
        
        plt.figure(figsize=(12, 5))
        
        # Plot de dispersão dos resíduos
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuos, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Valores Preditos')
        plt.ylabel('Resíduos')
        plt.title(f'Resíduos vs Preditos - {nome_modelo}')
        
        # Histograma dos resíduos
        plt.subplot(1, 2, 2)
        plt.hist(residuos, bins=30, edgecolor='black')
        plt.xlabel('Resíduos')
        plt.ylabel('Frequência')
        plt.title(f'Distribuição dos Resíduos - {nome_modelo}')
        
        plt.tight_layout()
        plt.show() 

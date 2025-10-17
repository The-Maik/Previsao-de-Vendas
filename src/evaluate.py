import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model():
    """
    Carrega o modelo treinado e os dados de teste para avaliar a performance
    e gerar gráficos de resultados.
    """
    print("Iniciando o processo de avaliação...")

    # --- Carregar Modelo e Dados ---
    try:
        model_path = os.path.join(os.path.dirname(
            __file__), '..', 'models', 'sales_predictor_model.joblib')
        model = joblib.load(model_path)
        print("Modelo carregado com sucesso.")

        data_path = os.path.join(os.path.dirname(
            __file__), '..', 'data', 'raw', 'historical_sales.csv')
        df = pd.read_csv(data_path, parse_dates=['Data'])
        print("Dados carregados com sucesso.")
    except FileNotFoundError as e:
        print(
            f"Erro: Arquivo não encontrado. Certifique-se de que os scripts anteriores foram executados. Detalhe: {e}")
        return

    # --- Preparar os Dados ---
    df['Ano'] = df['Data'].dt.year
    df['Mes'] = df['Data'].dt.month
    df['Dia'] = df['Data'].dt.day
    df['Dia_da_semana'] = df['Data'].dt.dayofweek
    df = df.drop('Data', axis=1)

    features = ['Ano', 'Mes', 'Dia', 'Dia_da_semana', 'Promocao_Ativa']
    target = 'Vendas'
    X = df[features]
    y = df[target]

    # --- Usar a MESMA divisão de treino/teste ---
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print("Conjunto de teste preparado.")

    # --- Fazer Previsões ---
    predictions = model.predict(X_test)
    print("Previsões realizadas no conjunto de teste.")

    # --- Calcular Métricas ---
    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\n--- Métricas de Avaliação ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print("-----------------------------\n")

    # --- Gerar Gráficos ---
    # Criar a pasta reports/figures se ela não existir
    figures_dir = os.path.join(os.path.dirname(
        __file__), '..', 'reports', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Gráfico 1 Valores Reais vs. Previsões
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [
             y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Valores Reais vs. Previsões')
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.grid(True)
    real_vs_pred_path = os.path.join(figures_dir, 'reais_vs_previsoes.png')
    plt.savefig(real_vs_pred_path)
    print(f"Gráfico 'Reais vs. Previsões' salvo em: {real_vs_pred_path}")

    # Gráfico 2 Resíduos
    residuals = y_test - predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Gráfico de Resíduos')
    plt.xlabel('Valores Previstos')
    plt.ylabel('Resíduos (Erro)')
    plt.grid(True)
    residuals_path = os.path.join(figures_dir, 'residuos_plot.png')
    plt.savefig(residuals_path)
    print(f"Gráfico 'Resíduos' salvo em: {residuals_path}")



if __name__ == '__main__':
    evaluate_model()

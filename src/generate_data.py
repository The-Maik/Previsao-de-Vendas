import pandas as pd
import numpy as np
import os


def generate_sales_data(start_date="2022-01-01", end_date="2024-12-31", filename="historical_sales.csv"):
    """
    Gera um dataset sintético de vendas diárias para uma franquia.

    O dataset inclui:
    - Uma tendência de crescimento ao longo do tempo.
    - Sazonalidade semanal (vendas maiores nos fins de semana).
    - Sazonalidade anual (picos em datas comemorativas).
    - Ruído aleatório para simular a variabilidade do dia a dia.
    - Efeito de promoções.
    """

    # 1. Criar o range de datas
    dates = pd.to_datetime(pd.date_range(
        start=start_date, end=end_date, freq='D'))
    n_days = len(dates)

    # 2. Criar a base de vendas com uma tendência de crescimento linear
    # Vendas crescendo de 1000 para 2500
    base_sales = np.linspace(start=1000, stop=2500, num=n_days)

    # 3. Adicionar sazonalidade semanal
    # Vendas aumentam na Sexta (4), Sábado (5) e diminuem no Domingo (6)
    day_of_week_effect = np.array([1.0, 1.0, 1.1, 1.2, 1.4, 1.5, 0.9])[
        dates.dayofweek]

    # 4. Adicionar sazonalidade anual (picos em meses específicos)
    month_effect = np.array([
        1.0, 1.0, 1.1, 1.2, 1.3,  # Jan-Mai
        1.4, 1.2, 1.1,  # Jun-Ago
        1.1, 1.2, 1.5, 1.8  # Set-Dez (pico no final do ano)
    ])[dates.month - 1]

    # 5. Adicionar ruído aleatório para simular a imprevisibilidade
    random_noise = np.random.normal(loc=1.0, scale=0.05, size=n_days)

    # 6. Simular o efeito de promoções
    # Promoções acontecem em 20% dos dias de forma aleatória
    is_promotion = np.random.choice([0, 1], size=n_days, p=[0.8, 0.2])
    promotion_effect = np.where(
        is_promotion == 1, 1.25, 1.0)  # Promoções aumentam vendas em 25%

    # 7. Combinar todos os efeitos para calcular as vendas finais
    final_sales = base_sales * day_of_week_effect * \
        month_effect * random_noise * promotion_effect
    final_sales = np.round(final_sales, 2)

    # 8. Criar o DataFrame final
    df = pd.DataFrame({
        'Data': dates,
        'Vendas': final_sales,
        'Promocao_Ativa': is_promotion
    })

    # 9. Salvar os dados em um arquivo CSV
    # Criar a pasta 'data/raw' se ela não existir
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

    print(f"Dataset gerado com sucesso e salvo em: {output_path}")
    print("\nVisualização das 5 primeiras linhas:")
    print(df.head())

    return df


# Executar a função quando o script for rodado diretamente
if __name__ == "__main__":
    generate_sales_data()

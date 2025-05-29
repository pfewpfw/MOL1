import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 1. Чтение датасета
df = pd.read_csv('D:/для работ/MMOlab/titanic_data.csv')

pd.options.display.max_columns = None
# 2. Вывод первых 5 строк датасета
print("Первые 5 строк датасета:")
print(df.head(10))

# 3. Подсчёт количества пропущенных значений для каждого столбца
print("\nКоличество пропущенных значений по столбцам:")
print(df.isnull().sum())

# 4. Заполнение пропущенных значений:
# Для числовых столбцов заполним медианой
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Для категориальных столбцов (тип object или category) заполним модой.
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nПосле заполнения пропусков:")
print(df.isnull().sum())

# 5. Нормализация числовых данных с помощью MinMaxScaler.
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Преобразование категориальных данных в dummy-переменные с параметром drop_first=True.
df = pd.get_dummies(df, drop_first=True)

print("\nДатасет после нормализации и преобразования категориальных данных:")
print(df.head())
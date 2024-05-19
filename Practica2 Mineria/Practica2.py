import pandas as pd
from tabulate import tabulate

#Practica 1
def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns))

df = pd.read_csv("C:/Users/ricar/OneDrive/Documentos/Mineria-Practicas/climate_change_data.csv")
print_tabulate(df)

#Practica 2: estadistica descriptiva
def analisis_temperaturas(df: pd.DataFrame)-> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], format="ISO8601") 
    df["anio"] = df["Date"].dt.year
    df_by_dep = df.groupby(["Country", "anio"]).agg({'Temperature': ['sum', 'count', 'mean', 'min', 'max']})
    df_by_dep = df_by_dep.reset_index()
    df_by_dep.columns = ['Country', 'anio', 'Suma_Total_temperatura', 'Conteo_temperatura', 'Promedio_temperaturas', 'Temperatura_Minima', 'Temperatura_Maxima']
    print_tabulate(df_by_dep.head())
    
    return df_by_dep

analyzed_df = analisis_temperaturas(df)
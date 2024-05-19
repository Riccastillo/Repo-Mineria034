import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from tabulate import tabulate
from scipy import stats
from scipy.stats import mode, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols


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

#Practica 3: crear imagenes de comportamiento
def normalize_distribution(dist: np.array, n: int) -> np.array:
    b = dist - min(dist) + 0.000001
    c = (b / np.sum(b)) * n
    return np.round(c)


def create_distribution(mean: float, size: int) -> pd.Series:
    return normalize_distribution(np.random.standard_normal(size), mean * size)

def generate_df(means: List[Tuple[float, float, str]], n: int) -> pd.DataFrame:
    lists = [
        (create_distribution(_x, n), create_distribution(_y, n), np.repeat(_l, n))
        for _x, _y, _l in means
    ]
    x = np.array([])
    y = np.array([])
    labels = np.array([])
    for _x, _y, _l in lists:
        x = np.concatenate((x, _x), axis=None)
        y = np.concatenate((y, _y))
        labels = np.concatenate((labels, _l))
    return pd.DataFrame({"x": x, "y": y, "label": labels})


def get_cmap(n, name="hsv"):
   
    return matplotlib.colormaps.get_cmap(name, n)

def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: List[np.array], labels: np.array, input_data: List[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]


groups = [(20, 20, "grupo1"), (100, 50, "grupo2"), (200, 430, "grupo3")]
df = generate_df(groups, 50)
scatter_group_by("img/groups.png", df, "x", "y", "label")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df.itertuples(index=False, name=None)
]
input_data = [point for point, _ in list_t]
labels = [label for _, label in list_t]
new_points = [np.array([100, 150]), np.array([1, 1]), np.array([1, 300]), np.array([80, 40]), np.array([400, 400])] # y, x
kn = k_nearest_neightbors(
    input_data,
    labels,
    new_points,
    5,
)
print(f"{new_points}, {kn}")


#Practica 4: Prueba ANOVA
def anova(df_aux: pd.DataFrame, str_ols: str):
    
    modl = ols(str_ols, data=df_aux).fit()
    anova_df = sm.stats.anova_lm(modl, typ=2)
    if anova_df["PR(>F)"][0] < 0.005:
        print("hay diferencias")
        print(anova_df)
        
    else:
        print("No hay diferencias")

def anova_1(file_name: str):
    df_complete = pd.read_csv(file_name)
    df_by_type = df_complete.groupby(["Country", "anio"])[["Temperature"]].aggregate(pd.DataFrame.sum)
    df_by_type.reset_index(inplace=True)
    df_aux = df_by_type.rename(columns={"Temperature": "TotalTemperature"}).drop(['anio'], axis=1)
    df_aux = df_aux.loc[df_aux["Country"].isin(["Afghanistan","Chile"])]
 
    print(df_aux.head())
    anova(df_aux, "TotalTemperature ~ Country")


#Practica 5: Pruebas de correlación
#se busca correlación entre la temperarura y las emisiones de CO2
r, p = stats.pearsonr(df['Temperature'], df['C02 Emissions'])
print(f"Correlación Pearson: r={r}, p-value={p}")

r, p = stats.spearmanr(df['Temperature'], df['C02 Emissions'])
print(f"Correlación Spearman: r={r}, p-value={p}")

r, p = stats.kendalltau(df['Temperature'], df['C02 Emissions'])
print(f"Correlación Pearson: r={r}, p-value={p}")



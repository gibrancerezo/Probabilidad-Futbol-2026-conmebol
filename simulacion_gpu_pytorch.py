
import pandas as pd
import torch
import unicodedata
from itertools import product

## Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Normalizar texto
def normalizar(nombre):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(nombre)) if not unicodedata.combining(c))

# Cargar archivos
tabla_posiciones = pd.read_csv("Tabla_posiciones.csv", encoding="latin1")
partidos_faltantes = pd.read_csv("PartidosFaltantes.csv", encoding="latin1")

# Normalizar nombres
tabla_posiciones["Country"] = tabla_posiciones["Country"].apply(normalizar)
partidos_faltantes["Local"] = partidos_faltantes["Local"].apply(normalizar)
partidos_faltantes["Visitor"] = partidos_faltantes["Visitor"].apply(normalizar)

# Aplicar resultado Bolivia 0-0 Uruguay
tabla_posiciones.loc[tabla_posiciones["Country"] == "Bolivia", "PTS"] += 1
tabla_posiciones.loc[tabla_posiciones["Country"] == "Uruguay", "PTS"] += 1
partidos_faltantes = partidos_faltantes[~(
    (partidos_faltantes["Local"] == "Bolivia") &
    (partidos_faltantes["Visitor"] == "Uruguay")
)]

# Preparar estructuras base
equipos = sorted(tabla_posiciones["Country"].tolist())
partidos = list(zip(partidos_faltantes["Local"], partidos_faltantes["Visitor"]))
num_partidos = len(partidos)
num_equipos = len(equipos)
equipo_idx = {e: i for i, e in enumerate(equipos)}

# Tensor de puntos base
puntos_base = torch.tensor([tabla_posiciones.loc[tabla_posiciones["Country"] == team, "PTS"].values[0]
                            for team in equipos], dtype=torch.int16, device=device)

# Combinaciones posibles por partido
goles = torch.arange(5, device=device)
comb_por_partido = torch.cartesian_prod(goles, goles)
total_combs = 25 ** num_partidos
max_combs = 100000

if total_combs > max_combs:
    print("Demasiadas combinaciones, usando muestra Montecarlo.")
    comb_idx = torch.randint(0, 25, (max_combs, num_partidos), device=device)
else:
    comb_idx = torch.tensor(list(product(range(25), repeat=num_partidos)), device=device)

goles_comb = comb_por_partido[comb_idx]  # [N, num_partidos, 2]

# Inicializar contadores
N = goles_comb.shape[0]
puntos = puntos_base.repeat(N, 1)
gd = torch.zeros_like(puntos)

# Calcular puntos y diferencia
for i, (local, visitante) in enumerate(partidos):
    gl = goles_comb[:, i, 0]
    gv = goles_comb[:, i, 1]
    idx_l = equipo_idx[local]
    idx_v = equipo_idx[visitante]

    gd[:, idx_l] += gl - gv
    gd[:, idx_v] += gv - gl
    puntos[:, idx_l] += (gl > gv) * 3 + (gl == gv)
    puntos[:, idx_v] += (gv > gl) * 3 + (gv == gl)

# Clasificación
ranking = torch.argsort(torch.stack((puntos, gd), dim=2), dim=1, descending=True)
pos_counts = torch.zeros((num_equipos, num_equipos), device=device)
class_counts = torch.zeros((num_equipos, 3), device=device)

for pos in range(num_equipos):
    for t in range(num_equipos):
        pos_counts[t, pos] += (ranking[:, pos] == t).sum()

for i in range(num_equipos):
    class_counts[i, 0] = pos_counts[i, :6].sum()
    class_counts[i, 1] = pos_counts[i, 6]
    class_counts[i, 2] = pos_counts[i, 7:].sum()

# Mostrar resultados
df_pos = pd.DataFrame((pos_counts / N * 100).cpu().numpy(), index=equipos,
                      columns=[f"Pos {i+1}" for i in range(num_equipos)]).round(2)

df_class = pd.DataFrame((class_counts / N * 100).cpu().numpy(), index=equipos,
                        columns=["Clasificado", "Repechaje", "Eliminado"]).round(2)

print("\n=== Probabilidades por Posición (%) ===")
print(df_pos)
print("\n=== Probabilidades de Clasificación (%) ===")
print(df_class)

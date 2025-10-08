#%%Comparador resultados de medidas ESAR sintesis de Elisa 
# Medidas a 300 kHz y 57 kA/m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import chardet
import re
import os
from uncertainties import ufloat
#%% Funciones
def plot_ciclos_promedio(directorio):
    # Buscar recursivamente todos los archivos que coincidan con el patrón
    archivos = glob(os.path.join(directorio, '**', '*ciclo_promedio*.txt'), recursive=True)

    if not archivos:
        print(f"No se encontraron archivos '*ciclo_promedio.txt' en {directorio} o sus subdirectorios")
        return
    fig,ax=plt.subplots(figsize=(8, 6),constrained_layout=True)
    for archivo in archivos:
        try:
            # Leer los metadatos (primeras líneas que comienzan con #)
            metadatos = {}
            with open(archivo, 'r') as f:
                for linea in f:
                    if not linea.startswith('#'):
                        break
                    if '=' in linea:
                        clave, valor = linea.split('=', 1)
                        clave = clave.replace('#', '').strip()
                        metadatos[clave] = valor.strip()

            # Leer los datos numéricos
            datos = np.loadtxt(archivo, skiprows=9)  # Saltar las 8 líneas de encabezado/metadatos

            tiempo = datos[:, 0]
            campo = datos[:, 3]  # Campo en kA/m
            magnetizacion = datos[:, 4]  # Magnetización en A/m

            # Crear etiqueta para la leyenda
            nombre_base = os.path.split(archivo)[-1].split('_')[1]
            #os.path.basename(os.path.dirname(archivo))  # Nombre del subdirectorio
            etiqueta = f"{nombre_base}"

            # Graficar

            ax.plot(campo, magnetizacion, label=etiqueta)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue

    plt.xlabel('H (kA/m)')
    plt.ylabel('M (A/m)')
    plt.title(f'Comparación de ciclos de histéresis {os.path.split(directorio)[-1]}')
    plt.grid(True)
    plt.legend()  # Leyenda fuera del gráfico
    plt.savefig('comparativa_ciclos_'+os.path.split(directorio)[-1]+'.png',dpi=300)
    plt.show()

def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=20,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N

#LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata

#%% Localizo ciclos y resultados
ciclos_autoclave=glob(('250828_autoclave/**/*ciclo_promedio*'),recursive=True)
ciclos_autoclave.sort()

ciclos_balon = glob('250828_balon/**/*ciclo_promedio*', recursive=True)
ciclos_balon.sort()

ciclos_NF = glob('250910_NF/**/*ciclo_promedio*', recursive=True)
ciclos_NF.sort()

labels=['Autoclave','Balon','NF']

#%% comparo los ciclos C 
plot_ciclos_promedio('250828_autoclave')
plot_ciclos_promedio('250828_balon')
plot_ciclos_promedio('250910_NF')

#%% selecciono mejores ciclos
# C1-2 -115711
# c2-1 -120233
# c3-2 -121118
# c4-2 121712 
# [1,0,1,1]
# f1-1- 123146
# f2 -3 - 124049
# f3 - 1 - 124517
# f4 - 1 - 125106
# [0,2,0,0]

#%% SAR y tau 

res_autoclave = glob('250828_autoclave/**/*resultados*',recursive=True)
res_balon = glob('250828_balon/**/*resultados*', recursive=True)
res_NF = glob('250910_NF/**/*resultados*', recursive=True)
res_C=[res_autoclave,res_balon, res_NF]
for r in res_C:
    r.sort()
    
SAR_autoclave,err_SAR_autoclave,tau_autoclave,err_tau_autoclave = [],[],[],[]
SAR_balon,err_SAR_balon,tau_balon,err_tau_balon = [],[],[],[]
SAR_NF,err_SAR_NF,tau_NF,err_tau_NF = [],[],[],[]
    
for C in res_autoclave:
    meta, _,_,_,_,_,_,_,_,_,_,_,SAR,tau,_= lector_resultados(C)
    SAR_autoclave.append(meta['SAR_W/g'].n)
    err_SAR_autoclave.append(meta['SAR_W/g'].s)
    tau_autoclave.append(meta['tau_ns'].n)
    err_tau_autoclave.append(meta['tau_ns'].s)
for C in res_balon:
    meta, _,_,_,_,_,_,_,_,_,_,_,SAR,tau,_= lector_resultados(C)
    SAR_balon.append(meta['SAR_W/g'].n)
    err_SAR_balon.append(meta['SAR_W/g'].s)
    tau_balon.append(meta['tau_ns'].n)
    err_tau_balon.append(meta['tau_ns'].s)    
for C in res_NF:    
    meta, _,_,_,_,_,_,_,_,_,_,_,SAR,tau,_= lector_resultados(C)
    SAR_NF.append(meta['SAR_W/g'].n)
    err_SAR_NF.append(meta['SAR_W/g'].s)
    tau_NF.append(meta['tau_ns'].n)
    err_tau_NF.append(meta['tau_ns'].s)

    
#%% SAR
fig,a=plt.subplots(nrows=1,figsize=(7,5),constrained_layout=True)
categories = ['Autoclave', 'Balon', 'NF']

for i, (sarC, errC) in enumerate(zip([SAR_autoclave, SAR_balon, SAR_NF],
                    [err_SAR_autoclave, err_SAR_balon, err_SAR_NF])):
    x_pos = [i+1]*len(sarC)  # Posición X fija para cada categoría
    a.errorbar(x_pos, sarC, yerr=errC,fmt='.', 
                 label=categories[i], 
                 capsize=5, linestyle='None')

a.set_title('Muestras C - 300 kHz - 57 kA/m',loc='left')
for ax in (a):
    ax.set_xlabel('Categorías')
    ax.set_ylabel('SAR (W/g)')
    ax.legend(ncol=2, loc='upper right')
    ax.grid(True, axis='y', linestyle='--')
# a.set_xticklabels(categories)
#ax.set_xticks([1])
#plt.savefig('comparativa_SAR_C_F.png', dpi=300)
plt.show()
#%%
#%% SAR y tau - Versión mejorada para arrays completos

res_autoclave = glob('250828_autoclave/**/*resultados*', recursive=True)
res_balon = glob('250828_balon/**/*resultados*', recursive=True)
res_NF = glob('250910_NF/**/*resultados*', recursive=True)
res_C = [res_autoclave, res_balon, res_NF]

for r in res_C:
    r.sort()

# Listas para almacenar los arrays completos de SAR
SAR_arrays_autoclave, SAR_arrays_balon, SAR_arrays_NF = [], [], []

# Extraer todos los arrays de SAR para cada categoría
for C in res_autoclave:
    meta, _, _, _, _, _, _, _, _, _, _, _, SAR, tau, _ = lector_resultados(C)
    SAR_arrays_autoclave.append(SAR)

for C in res_balon:
    meta, _, _, _, _, _, _, _, _, _, _, _, SAR, tau, _ = lector_resultados(C)
    SAR_arrays_balon.append(SAR)

for C in res_NF:
    meta, _, _, _, _, _, _, _, _, _, _, _, SAR, tau, _ = lector_resultados(C)
    SAR_arrays_NF.append(SAR)

#%% Gráfico de SAR vs índice (3 filas, 1 columna)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), constrained_layout=True,sharex=True)
categorias = ['Autoclave', 'Balón', 'NF']
colores = ['tab:blue', 'tab:red', 'tab:green']

# Función para graficar cada categoría
def graficar_sar_vs_indice(ax, sar_arrays, categoria, color, idx):
    for i, sar_array in enumerate(sar_arrays):
        indices = np.arange(len(sar_array))  # Índices del array
        ax.plot(indices, sar_array, 'o-', alpha=0.7, linewidth=1, markersize=4,
                label=f'{categoria} {i+1}', color=color)
    
    ax.set_title(f'{categoria} - SAR vs Índice', loc='left')

    ax.set_ylabel('SAR (W/g)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar límites del eje Y para mejor visualización
    # if sar_arrays:  # Si hay datos
    #     all_values = np.concatenate(sar_arrays)
    #     y_min = np.min(all_values) * 0.98
    #     y_max = np.max(all_values) * 1.02
    #     ax.set_ylim(y_min, y_max)
axes[2].set_xlabel('Índice')

# Graficar cada categoría en su propio subplot
graficar_sar_vs_indice(axes[0], SAR_arrays_autoclave, 'Autoclave', 'tab:blue', 0)
graficar_sar_vs_indice(axes[1], SAR_arrays_balon, 'Balón', 'tab:red', 1)
graficar_sar_vs_indice(axes[2], SAR_arrays_NF, 'NF', 'tab:green', 2)

plt.suptitle('Comparativa SAR vs Índice - Muestras C - 300 kHz - 57 kA/m', fontsize=14)
plt.savefig('comparativa_SAR_Eli.png', dpi=300)
plt.show()
#%% Gráfico combinado: SAR vs índice + Ciclos de histéresis (ESTILOS HOMOGENEIZADOS)
# Configurar estilos globales de matplotlib
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 11), constrained_layout=True, sharex='col')
categorias = ['Autoclave', 'Balón', 'NF']
colores = ['tab:blue', 'tab:red', 'tab:green']

# Listas de ciclos por categoría (usando las que ya tienes cargadas)
ciclos_por_categoria = {
    'Autoclave': ciclos_autoclave,
    'Balón': ciclos_balon,
    'NF': ciclos_NF
}

# Función para graficar SAR vs índice (columna izquierda)
def graficar_sar_vs_indice(ax, sar_arrays, categoria, color, idx):
    for i, sar_array in enumerate(sar_arrays):
        indices = np.arange(len(sar_array))
        ax.plot(indices, sar_array, 'o-', alpha=0.7, linewidth=1, markersize=4,
                label=f'{categoria} {i+1}', color=color)
    
    ax.set_title(f'{categoria} - SAR vs Índice', loc='left', fontweight='bold', fontsize=13)
    ax.set_ylabel('SAR (W/g)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Estilo de ticks
    ax.tick_params(axis='both', which='major', labelsize=11)

# Función para graficar ciclos de histéresis usando lector_ciclos (columna derecha)
def graficar_ciclos_mejorado(ax, archivos_ciclos, categoria, color):
    if not archivos_ciclos:
        ax.text(0.5, 0.5, f'No hay ciclos\n{categoria}', 
                transform=ax.transAxes, ha='center', va='center', fontsize=12, fontweight='bold')
        return
    
    # Ordenar archivos para consistencia
    archivos_ciclos.sort()
    
    for j, archivo in enumerate(archivos_ciclos):
        try:
            t, _, _, H_kAm, M_Am, metadata = lector_ciclos(archivo)
            
            # Usar el mismo color para todos los ciclos de esta categoría
            # Etiqueta simple: Autoclave 1, Autoclave 2, etc.
            etiqueta = f"{categoria} {j+1}"
            
            ax.plot(H_kAm/1000, M_Am, label=etiqueta, color=color, linewidth=2, alpha=0.8)

        except Exception as e:
            print(f"Error procesando archivo {archivo}: {str(e)}")
            continue
    
    ax.set_title(f'{categoria} - Ciclos de Histéresis', loc='left', fontweight='bold', fontsize=13)
    ax.set_xlabel('H (kA/m)', fontweight='bold')
    ax.set_ylabel('M (A/m)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=10)
    
    # Estilo de ticks
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Ajustar aspecto para mejor visualización
    ax.set_aspect('auto')

# Graficar cada categoría en filas
for i, categoria in enumerate(categorias):
    # Columna izquierda: SAR vs índice
    if categoria == 'Autoclave':
        graficar_sar_vs_indice(axes[i, 0], SAR_arrays_autoclave, categoria, colores[i], i)
    elif categoria == 'Balón':
        graficar_sar_vs_indice(axes[i, 0], SAR_arrays_balon, categoria, colores[i], i)
    elif categoria == 'NF':
        graficar_sar_vs_indice(axes[i, 0], SAR_arrays_NF, categoria, colores[i], i)
    
    # Columna derecha: Ciclos de histéresis usando lector_ciclos
    archivos_ciclos = ciclos_por_categoria[categoria]
    graficar_ciclos_mejorado(axes[i, 1], archivos_ciclos, categoria, colores[i])

# Configuración adicional para los ejes
axes[2, 0].set_xlabel('Índice', fontweight='bold', fontsize=12)
axes[2, 1].set_xlabel('H (kA/m)', fontweight='bold', fontsize=12)

plt.suptitle('Análisis Completo: SAR vs Índice y Ciclos de Histéresis\nSintesis Elisa - 300 kHz - 57 kA/m', 
             fontsize=16, fontweight='bold')
plt.savefig('analisis_completo_SAR_ciclos.png', dpi=300, bbox_inches='tight')
plt.show()

# Restaurar configuración por defecto (opcional)
plt.rcParams.update(plt.rcParamsDefault)
#%% Corrección del gráfico simple de SAR (el que tenía error)
fig, a = plt.subplots(nrows=1, figsize=(7, 5), constrained_layout=True)
categories = ['Autoclave', 'Balon', 'NF']

for i, (sarC, errC) in enumerate(zip([SAR_autoclave, SAR_balon, SAR_NF],
                    [err_SAR_autoclave, err_SAR_balon, err_SAR_NF])):
    x_pos = [i+1] * len(sarC)  # Posición X fija para cada categoría
    a.errorbar(x_pos, sarC, yerr=errC, fmt='.', 
                 label=categories[i], 
                 capsize=5, linestyle='None')

a.set_title('Muestras C - 300 kHz - 57 kA/m', loc='left')
a.set_xlabel('Categorías')
a.set_ylabel('SAR (W/g)')
a.legend(ncol=2, loc='upper right')
a.grid(True, axis='y', linestyle='--')
plt.show()

#% Información de los ciclos cargados
print("Resumen de ciclos cargados:")
print("=" * 50)
for categoria in categorias:
    archivos = ciclos_por_categoria[categoria]
    print(f"{categoria}: {len(archivos)} archivos de ciclo")
    for archivo in archivos:
        nombre = os.path.basename(archivo)
        print(f"  - {nombre}")
print("=" * 50)
#%% Opcional: Gráfico combinado en una sola figura para comparación directa
fig2, ax2 = plt.subplots(figsize=(8,4), constrained_layout=True)

# Graficar todos juntos para comparación
for i, (sar_arrays, categoria, color) in enumerate(zip(
    [SAR_arrays_autoclave, SAR_arrays_balon, SAR_arrays_NF], 
    categorias, 
    colores
)):
    for j, sar_array in enumerate(sar_arrays):
        indices = np.arange(len(sar_array))
        # Desplazar ligeramente los puntos para evitar superposición
        x_offset = i * 0.1
        ax2.plot(indices + x_offset, sar_array, 'o-', alpha=0.6, markersize=3,
                label=f'{categoria} {j+1}' if j == 0 else "", color=color)

ax2.set_title('Comparativa SAR vs Índice - Todas las muestras\n300 kHz - 57 kA/m', loc='left')
ax2.set_xlabel('Índice')
ax2.set_ylabel('SAR (W/g)')
ax2.legend(ncol=3, loc='upper right')
ax2.grid()

plt.show()

#%% Estadísticas básicas
print("Estadísticas de SAR:")
print("=" * 50)
for categoria, sar_arrays in zip(categorias, [SAR_arrays_autoclave, SAR_arrays_balon, SAR_arrays_NF]):
    if sar_arrays:
        all_values = np.concatenate(sar_arrays)
        print(f"{categoria}:")
        print(f"  Número de muestras: {len(sar_arrays)}")
        print(f"  Total de puntos: {len(all_values)}")
        print(f"  SAR promedio: {np.mean(all_values):.2f} ± {np.std(all_values):.2f} W/g")
        print(f"  Rango: [{np.min(all_values):.2f}, {np.max(all_values):.2f}] W/g")
        print()


#%% Tau
fig2, (a, b) = plt.subplots(nrows=2, figsize=(7, 5), constrained_layout=True)

categories = ['C1', 'C2', 'C3', 'C4']
categories2 = ['F1', 'F2', 'F3', 'F4']

for i, (tauC, err_tauC) in enumerate(zip([tau_C1, tau_C2, tau_C3, tau_C4],
                                         [err_tau_C1, err_tau_C2, err_tau_C3, err_tau_C4])):
    x_pos = [i+1]*len(tauC)  # Posición X fija para cada categoría
    a.errorbar(x_pos, tauC, yerr=err_tauC, fmt='.', 
               label=categories[i], 
               capsize=5, linestyle='None')

for j, (tauF, err_tauF) in enumerate(zip([tau_F1, tau_F2, tau_F3, tau_F4],
                                         [err_tau_F1, err_tau_F2, err_tau_F3, err_tau_F4])):
    x2_pos = [j+1]*len(tauF)  # Posición X fija para cada categoría
    b.errorbar(x2_pos, tauF, yerr=err_tauF, fmt='.', 
               label=categories2[j], 
               capsize=5, linestyle='None')

a.set_title('Muestras C - 300 kHz - 57 kA/m',loc='left')
b.set_title('Muestras F - 300 kHz - 57 kA/m',loc='left')

for ax in (a, b):
    ax.set_ylabel(r'$\tau$ (ns)')
    ax.set_xticks([1, 2, 3, 4])
    # ax.set_xlabel('Categorías')
    ax.legend(ncol=2, loc='upper right')
    ax.grid(True, axis='y', linestyle='--')
a.set_xticklabels(categories)
b.set_xticklabels(categories2)
plt.savefig('comparativa_tau_C_F.png', dpi=300)
plt.show()

# %%

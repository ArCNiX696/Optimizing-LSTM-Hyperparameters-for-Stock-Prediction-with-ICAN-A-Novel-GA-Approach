import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gc
import os
from tabulate import tabulate


def plot_corr_matrix(corr_matrix):
    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.4f') #annot=show corr values ,cmap= colors, fmt=decimals 
    plt.title('Correlation Matrix')
    plt.show()

def boxplot(df, target_variable=str):
    # Create a figure and axes for the plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Create the boxplot for each column with respect to the target variable
    sns.boxplot(data=df.drop(columns=target_variable), ax=ax)
    sns.swarmplot(data=df.drop(columns=target_variable), color=".25", ax=ax)
    
    # Set the title and axes labels
    ax.set_title('Boxplot of distribution with respect to ' + target_variable)
    ax.set_ylabel('Distributon')
    
    # Rotate x-axis labels for better visualization
    plt.xticks(rotation=45)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming df is your DataFrame and 'target' is the target variable
# boxplot_all_columns(df, 'target')

def plot_corr_scatter(df):
    sns.set_theme(style='whitegrid')
    sns.pairplot(df,height=1.6)
    plt.show()


def plot_datetime(df,column=str,column2=str):
    df= df[[column,column2]]

    df[column]= pd.to_datetime(df[column])

    plt.title(f'{column2} with respect to {column}')
    plt.xlabel(column)
    plt.ylabel(column2)
    plt.plot(df[column] , df[column2])

def plot_evolution(data,x_label=str,y_label=str,title=str,file_name=str):
        plt.figure(figsize=(8, 8)) 
        plt.plot(data, color='Red',marker = 'o', linewidth=2,label='Simple GA')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)  
        plt.legend()
        plot_path = os.path.join('./Statistics results/GA statistics',f'{file_name}.png')
        plt.savefig(plot_path)
        plt.close() 
        gc.collect()
        # plt.show()

# Best_fitness = [0.1596245914697647, 0.14490897301584482, 0.13173003681004047, 0.14403520617634058, 0.15283382963389158, 0.12628928137322265, 0.1353702275082469, 0.1405266928486526, 0.10062975925393403, 0.14287359081208706]
# plot_evolution(Best_fitness,'Number of Generations','Fitness score','Chromosome evolution in all generations',f'Evolution till generation {10}, best fitness {0.10062975925393403}')


def plot_comparison(simple_ga, ga_ican):
    # Define the metrics
    metrics = ['MSE', 'MAE', 'MAPE', 'R²']

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # the width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, simple_ga, width, label='Simple GA', color='#1f77b4')
    rects2 = ax.bar(x + width/2, ga_ican, width, label='GA ICAN', color='#ff7f0e')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Genetic Algorithms for LSTM Hyperparameter Tuning')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Add a grid for better readability
    ax.grid(True, linestyle='--', linewidth=0.5)

    # Function to add labels to the bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 4)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    fig.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_violin_comparison(metric_name, simple_ga_value, ga_ican_value):
    # Prepare the data for plotting
    data = {
        'Algorithm': ['Simple GA', 'GA ICAN'],
        'Value': [simple_ga_value, ga_ican_value]
    }
    
    df = pd.DataFrame(data)
    
    # Create the violin plot
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Algorithm', y='Value', data=df, inner='point')
    
    plt.title(f'Comparison of {metric_name}')
    plt.ylabel('Score')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()


def plot_results_table(metrics, simple_ga_results, ga_ican_results):
    # Redondear los valores a 6 decimales
    simple_ga_results = [round(x, 6) for x in simple_ga_results]
    ga_ican_results = [round(x, 6) for x in ga_ican_results]

    # Crear el DataFrame con los resultados
    data = {
        'Metric': metrics,
        'Simple GA': simple_ga_results,
        'GA ICAN': ga_ican_results
    }
    
    df = pd.DataFrame(data)
    
    # Configurar la figura y el axis
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Ocultar el axis
    ax.axis('tight')
    ax.axis('off')
    
    # Crear la tabla
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Estilizar la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    
    # Estilizar encabezados de columnas
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('grey')
        cell.set_linewidth(0.5)
        if key[0] == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f5f5f5')

    # Ajustar el tamaño y la apariencia de la figura
    plt.title('Comparison of Genetic Algorithms for LSTM Hyperparameter Tuning', fontsize=14, weight='bold')
    plt.subplots_adjust(left=0.2, top=0.8, right=0.8, bottom=0.2)
    
    plt.show()

# # Datos para los algoritmos genéticos
# simple_ga_results = [0.003018892777618, 0.070967644453049, 0.191974729299545, 0.743038058280945]
# ga_ican_results = [0.001772226591129, 0.046816967427731, 0.128945827484131, 0.753030300140381]
# metrics = ['MSE', 'MAE', 'MAPE', 'R²']

# # Llamar a la función para mostrar la tabla
# plot_results_table(metrics, simple_ga_results, ga_ican_results)

import plotly.graph_objects as go

def plot_results_table(metrics, simple_ga_results, ga_ican_results):
    # Redondear los valores a 6 decimales
    simple_ga_results = [round(x, 6) for x in simple_ga_results]
    ga_ican_results = [round(x, 6) for x in ga_ican_results]

    # Crear el DataFrame con los resultados
    data = {
        'Metric': metrics,
        'Simple GA': simple_ga_results,
        'GA ICAN': ga_ican_results
    }
    
    df = pd.DataFrame(data)
    
    # Crear la tabla utilizando Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=['<b>Metric</b>', '<b>Simple GA</b>', '<b>GA ICAN</b>'],
                    fill_color='#1f77b4',
                    align='center',
                    font=dict(color='white', size=14),
                    line_color='darkslategray'),
        cells=dict(values=[df.Metric, df['Simple GA'], df['GA ICAN']],
                   fill_color=[['#f8f8f8', '#e2e2e2']*len(df)],
                   align='center',
                   font=dict(color='black', size=12),
                   line_color='darkslategray'))
    ])

    fig.update_layout(
        title_text='Comparison of Genetic Algorithms for LSTM Hyperparameter Tuning',
        title_x=0.5,
        title_font_size=20,
        title_font_family='Arial',
        title_font_color='darkblue',
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white'
    )
    
    fig.show()

# # Datos para los algoritmos genéticos
# simple_ga_results = [0.003018892777618, 0.070967644453049, 0.191974729299545, 0.743038058280945]
# ga_ican_results = [0.001772226591129, 0.046816967427731, 0.128945827484131, 0.753030300140381]
# metrics = ['MSE', 'MAE', 'MAPE', 'R²']

# # Llamar a la función para mostrar la tabla
# plot_results_table(metrics, simple_ga_results, ga_ican_results)

import matplotlib.pyplot as plt

# Datos de tiempo de ejecución en segundos
simple_ga_time = 47 * 3600 + 12 * 60 + 47  # 47 horas, 12 minutos, 47 segundos
ga_ican_time = 33 * 3600 + 54 * 60 + 2  # 33 horas, 54 minutos, 2 segundos

# Convertir los tiempos a horas para una mejor representación
simple_ga_time_hours = simple_ga_time / 3600
ga_ican_time_hours = ga_ican_time / 3600

# Datos para el gráfico
algorithms = ['Simple GA', 'GA ICAN']
execution_times = [simple_ga_time_hours, ga_ican_time_hours]

# Crear el gráfico de barras horizontales
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(algorithms, execution_times, color=['#1f77b4', '#ff7f0e'])

# Añadir etiquetas a las barras
for bar in bars:
    width = bar.get_width()
    label_y_pos = bar.get_y() + bar.get_height() / 2
    ax.text(width, label_y_pos, f'{width:.2f} hours', va='center', ha='left', color='black', fontsize=12)

# Configurar los títulos y etiquetas
ax.set_xlabel('Execution Time (hours)')
ax.set_title('Execution Time Comparison of Genetic Algorithms')
ax.grid(True, linestyle='--', linewidth=0.5)

plt.show()
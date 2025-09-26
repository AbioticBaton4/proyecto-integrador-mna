# Análisis y Pronóstico de la Depresión en México: De la Evidencia Epidemiológica a la Inteligencia Predictiva

Un proyecto de Machine Learning para analizar y predecir patrones de depresión en México utilizando datos epidemiológicos y técnicas de inteligencia artificial.

## Descripción del Proyecto

Este proyecto tiene como objetivo desarrollar modelos predictivos para analizar los factores de riesgo asociados con la depresión en México, utilizando datos epidemiológicos y técnicas avanzadas de Machine Learning para generar insights que puedan contribuir a la salud pública.

## Estructura del Proyecto

```
proyecto-integrador-mna/
├── data/
│   ├── raw/              # Datos originales, inmutables
│   ├── interim/          # Datos transformados, intermedios
│   ├── processed/        # Datos finales para modelado
│   └── external/         # Datos de terceros
├── notebooks/
│   ├── exploratory/      # Análisis exploratorio de datos (EDA)
│   ├── modeling/         # Desarrollo de modelos
│   └── evaluation/       # Evaluación y validación
├── src/
│   ├── data/
│   │   └── make_dataset.py    # Scripts para generar datasets
│   ├── features/
│   │   └── build_features.py  # Transformación de features
│   ├── models/
│   │   ├── train_model.py     # Entrenamiento de modelos
│   │   └── predict_model.py   # Predicciones
│   ├── visualization/
│   │   └── visualize.py       # Gráficos y visualizaciones
│   └── utils/
│       └── helpers.py         # Funciones auxiliares
├── models/               # Modelos entrenados serializados
├── reports/
│   ├── figures/          # Gráficos generados
│   └── final/            # Reportes finales
├── config/               # Archivos de configuración
├── tests/                # Pruebas unitarias
├── docs/                 # Documentación
├── requirements.txt      # Dependencias de Python (pip)
├── environment.yml       # Configuración de entorno conda
├── .gitignore           # Archivos a ignorar por Git
└── README.md            # Este archivo
```

## Inicio Rápido

### Prerrequisitos

- Python 3.8+
- Git
- Anaconda/Miniconda (opcional, para instalación con conda)

### Instalación

#### Opción 1: Instalación con pip (recomendado)

1. Clonar el repositorio:
```bash
git clone https://github.com/AbioticBaton4/proyecto-integrador-mna.git
cd proyecto-integrador-mna
```

2. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

#### Opción 2: Instalación con conda

1. Clonar el repositorio:
```bash
git clone https://github.com/AbioticBaton4/proyecto-integrador-mna.git
cd proyecto-integrador-mna
```

2. Crear el entorno desde el archivo environment.yml:
```bash
conda env create -f environment.yml
```

3. Activar el entorno:
```bash
conda activate depression-analysis
```

## Flujo de Trabajo

### 1. Preparación de Datos
- Los datos originales se almacenan en `data/raw/`
- Los scripts de procesamiento están en `src/data/`
- Los datos procesados se guardan en `data/processed/`

### 2. Análisis Exploratorio
- Notebooks de EDA en `notebooks/exploratory/`
- Visualizaciones generadas en `reports/figures/`

### 3. Ingeniería de Características
- Scripts de feature engineering en `src/features/`
- Transformaciones y escalado de variables

### 4. Modelado
- Desarrollo de modelos en `notebooks/modeling/`
- Scripts de entrenamiento en `src/models/`
- Modelos guardados en `models/`

### 5. Evaluación
- Notebooks de evaluación en `notebooks/evaluation/`
- Métricas y reportes finales en `reports/final/`

## Metodología

El proyecto sigue una metodología estándar de ciencia de datos:

1. **Comprensión del Negocio**: Definición del problema de salud pública
2. **Comprensión de los Datos**: Exploración de datos epidemiológicos
3. **Preparación de Datos**: Limpieza y transformación
4. **Modelado**: Desarrollo de algoritmos predictivos
5. **Evaluación**: Validación de modelos
6. **Despliegue**: Implementación de soluciones

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Autores

- **Salvador Mendoza Pérez** - *Trabajo inicial* - [AbioticBaton4](https://github.com/AbioticBaton4)
- **Stephanie Fortiz de Ita** - *Trabajo inicial* - [StephanieFortiz](https://github.com/StephanieFortiz)
- **Naomi Fabiola Tokunaga Nuñez**  - *Trabajo inicial* - [naomi-fabiolatn](https://github.com/naomi-fabiolatn)
  

## Agradecimientos

- Instituto Nacional de Salud Pública de México
- Instituto Mexicano del Seguro Social
- Tecnológico de Monterrey
- Comunidad de Data Science
- Contribuidores del proyecto

---

*Proyecto desarrollado como parte del programa de Maestría en Inteligencia Artificial Aplicada - Tecnológico de Monterrey*
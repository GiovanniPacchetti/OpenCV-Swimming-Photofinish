## 📋 Descripción

El sistema procesa videos de carreras de natación (vista cenital/lateral) y realiza las siguientes tareas:
1.  **Detección de Carriles:** Selección manual asistida o automática basada en el estilo de nado.
2.  **Tracking de Nadadores:** Detección de movimiento mediante sustracción de fondo (MOG2) y seguimiento con rectángulos adaptativos.
3.  **Lógica de Llegada:** Detección precisa del cruce de la línea de meta con validación temporal para evitar falsos positivos por salpicaduras.
4.  **Generación de Resultados:** Visualización en tiempo real y exportación de videos con la clasificación final.

## 📂 Estructura del Proyecto

El repositorio contiene los siguientes scripts principales:

*   **`proyecto_manual.py`**: 
    *   🔴 **Script Principal (Core).** Contiene toda la lógica del sistema: clases de tracking (`SwimmerTracker`), gestión de carriles (`CarrilesInclinadosDinamicos`), detección MOG2 y funciones de geometría.
    *   Puede ejecutarse directamente para procesar un video y visualizarlo en ventana.
    *   Ajustar las variables del principio del codigo dependiendo del video seleccionado

*   **`video_making.py`**:
    *   🎥 **Script de Grabación.** Utiliza las clases de `proyecto_manual.py` para procesar un video y generar un archivo `.mp4` de salida con una interfaz gráfica (barra lateral con clasificación).
    *   Incluye lógica de detección automática de estilo basada en el nombre del archivo.

*   **`evaluacion_resultados.py`**:
    *   📊 **Script de Benchmarking.** Ejecuta el sistema sobre un conjunto de videos de prueba predefinidos, compara los resultados con un *Ground Truth* manual y genera métricas de precisión y velocidad (FPS).

*   **`Proyecto/`**: Carpeta donde se deben colocar los videos de entrada (ej: `libre.mp4`, `braza.mp4`).


## 🛠️ Uso

### 1. Procesar un video en tiempo real
Para ver el análisis en pantalla sin guardar:
```bash
python proyecto_manual.py
```
*Asegúrate de editar la variable `VIDEO_PATH` dentro del archivo para apuntar a tu video.*

### 2. Generar un video con los resultados
Para crear un video de salida con la interfaz de clasificación:
```bash
python video_making.py
```
*Este script generará un archivo `resultado_profesional.mp4`.*

### 3. Ejecutar evaluación de rendimiento
Para obtener métricas de precisión sobre varios videos:
```bash
python evaluacion_resultados.py
```

## ⚙️ Configuración y Controles

Durante la ejecución, el sistema pedirá interacción del usuario en el primer frame:
1.  **Selección de Carriles:** Haz clic en el borde superior e inferior de cada corchera.
2.  **Línea de Meta:** Haz clic en los dos extremos de la línea de meta.
3.  **Máscara (ROI):** Dibuja un polígono alrededor de la piscina para ignorar las gradas.

**Teclas durante la ejecución:**
- `q`: Salir / Detener grabación.

## 🧠 Tecnologías Utilizadas

-   **Python**
-   **OpenCV:** MOG2 (Background Subtraction), Lucas-Kanade (Optical Flow), Morphological Ops.
-   **NumPy:** Operaciones matriciales.

## 📝 Autor

**Giovanni Pacchetti Astigarraga**  
[giovanni.pacchetti@opendeusto.es](mailto:giovanni.pacchetti@opendeusto.es)

***
*Proyecto desarrollado para la asignatura de Visión por Computador.*

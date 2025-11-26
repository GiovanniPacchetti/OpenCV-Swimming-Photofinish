import cv2 as cv
import numpy as np
import time
from collections import defaultdict

# Importar tus clases y funciones del archivo principal
# Asumiendo que tu archivo principal se llama 'proyecto_natacion.py'
from proyecto_manual import (
    procesar_video_natacion, 
    crear_mascara_piscina, 
    CarrilesInclinadosDinamicos,
    LineaMetaTracker,
    SwimmerTracker
)

# ============================================================================
# DEFINICIÓN DE GROUND TRUTH (LA VERDAD PARA COMPARAR)
# ============================================================================
# Diccionario con los datos reales de cada video para comparar
# Clave: Nombre del archivo
# Valor: { 'ganador_real': carril_ganador, 'num_nadadores': cantidad_total }

GROUND_TRUTH = {
    'Proyecto/libre.mp4': {
        'ganador_real': 6,       # Carril que gana en realidad
        'num_nadadores': 8,      # Nadadores que deberían detectarse
        'estilo': 'Libre'
    },
    'Proyecto/espalda.mp4': {
        'ganador_real': 6,
        'num_nadadores': 8,
        'estilo': 'Espalda'
    },
    'Proyecto/braza.mp4': {
        'ganador_real': 6,
        'num_nadadores': 8,
        'estilo': 'Braza'
    },
    'Proyecto/mariposa1.mp4': {
        'ganador_real': 5,
        'num_nadadores': 7,
        'estilo': 'Mariposa'
    },
    'Proyecto/mariposa2.mp4': {
        'ganador_real': 2,
        'num_nadadores': 8,
        'estilo': 'Mariposa'
    }
}

# ============================================================================
# FUNCIÓN DE EVALUACIÓN
# ============================================================================

def evaluar_rendimiento():
    print("\n" + "="*60)
    print("INICIANDO EVALUACIÓN DE RENDIMIENTO DEL SISTEMA")
    print("="*60 + "\n")
    
    resultados_totales = {
        'aciertos_ranking': 0,
        'total_videos': 0,
        'detecciones_correctas': 0,
        'total_frames_analizados': 0,
        'fps_promedio': []
    }
    
    for video_path, datos_reales in GROUND_TRUTH.items():
        print(f"\n>>> Evaluando: {video_path} ({datos_reales['estilo']})")
        
        # 1. Medir tiempo de ejecución
        start_time = time.time()
        
        # 2. Ejecutar tu algoritmo (versión silenciosa si es posible)
        # Nota: Asegúrate de que 'procesar_video_natacion' devuelva el ranking
        try:
            ranking_obtenido = procesar_video_natacion(
                video_path, 
                direccion='izquierda', 
                seleccionar_roi=True,  # Tendrás que seleccionar ROI manualmente para cada uno
                delay_ms=1,            # Rápido para evaluación
                seleccionar_meta_manual=True
            )
        except Exception as e:
            print(f"❌ Error procesando {video_path}: {e}")
            continue
            
        end_time = time.time()
        duracion = end_time - start_time
        
        # 3. Calcular métricas
        if not ranking_obtenido:
            print("⚠️ No se obtuvo ranking.")
            continue
            
        # Ganador detectado (el primero en la lista de ranking)
        ganador_detectado = ranking_obtenido[0][0]  # (carril, frame) -> carril
        
        # Comparar con Ground Truth
        es_correcto = (ganador_detectado == datos_reales['ganador_real'])
        
        # Calcular FPS
        cap = cv.VideoCapture(video_path)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps_procesamiento = total_frames / duracion
        cap.release()
        
        # Guardar resultados
        resultados_totales['total_videos'] += 1
        if es_correcto:
            resultados_totales['aciertos_ranking'] += 1
        resultados_totales['fps_promedio'].append(fps_procesamiento)
        
        print(f"   - Ganador Real: Carril {datos_reales['ganador_real']}")
        print(f"   - Ganador Detectado: Carril {ganador_detectado}")
        print(f"   - Resultado: {'✅ CORRECTO' if es_correcto else '❌ INCORRECTO'}")
        print(f"   - Velocidad: {fps_procesamiento:.2f} FPS")

    # ============================================================================
    # INFORME FINAL
    # ============================================================================
    
    print("\n" + "="*60)
    print("RESULTADOS FINALES DE LA EVALUACIÓN")
    print("="*60)
    
    if resultados_totales['total_videos'] > 0:
        precision_ranking = (resultados_totales['aciertos_ranking'] / resultados_totales['total_videos']) * 100
        fps_global = sum(resultados_totales['fps_promedio']) / len(resultados_totales['fps_promedio'])
        
        print(f"\n📊 PRECISIÓN DE RANKING (Photo-Finish):")
        print(f"   {precision_ranking:.2f}% ({resultados_totales['aciertos_ranking']}/{resultados_totales['total_videos']} videos correctos)")
        
        print(f"\n⚡ RENDIMIENTO COMPUTACIONAL:")
        print(f"   Velocidad promedio: {fps_global:.2f} FPS")
        
        print(f"\n🎯 DESGLOSE POR ESTILO:")
        # Aquí podrías agregar lógica para imprimir aciertos por estilo si guardas esos datos
        print("   (Revisar logs individuales arriba)")
        
    else:
        print("No se completó ninguna evaluación.")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluar_rendimiento()

import cv2 as cv
import numpy as np

# ============================================================================
# 1. CONFIGURACIÓN DE GRABACIÓN
# ============================================================================

VIDEO_INPUT = 'Proyecto/braza.mp4'       
VIDEO_OUTPUT = 'resultado_Braza.mp4'
NUM_CARRILES = 8
DIRECCION = 'izquierda'
ANCHO_INTERFAZ = 300  # Ancho de la barra lateral en píxeles

# ============================================================================
# 2. COORDENADAS HARDCODEADAS
# ============================================================================

LINEAS_LIBRE = [
    (0, 52, 768, 34), (0, 69, 768, 57), (0, 100, 768, 79),
    (0, 130, 768, 116), (0, 168, 768, 149), (0, 216, 768, 191),
    (0, 271, 768, 239), (0, 330, 768, 321), (0, 431, 768, 405),
]
LINEAS_ESPALDA = [
    (0, 58, 768, 37), (0, 85, 768, 53), (0, 109, 768, 78),
    (0, 141, 768, 107), (0, 177, 768, 137), (0, 217, 768, 178),
    (0, 272, 768, 226), (0, 341, 768, 285), (0, 420, 768, 369),
]
LINEAS_BRAZA = [
    (0, 48, 768, 31), (0, 72, 768, 53), (0, 98, 768, 78),
    (0, 133, 768, 103), (0, 170, 768, 134), (0, 216, 768, 172),
    (0, 262, 768, 232), (0, 345, 768, 290), (0, 429, 768, 370),
]
LINEAS_MARIPOSA1 = [     
    (0, 82, 768, 53), (0, 110, 768, 77), (0, 138, 768, 104),
    (0, 176, 768, 136), (0, 218, 768, 175), (0, 271, 768, 225),
    (0, 339, 768, 286), (0, 431, 768, 369),
]
LINEAS_MARIPOSA2 = [
    (0, 39, 768, 13), (0, 65, 768, 33), (0, 90, 768, 53),
    (0, 120, 768, 86), (0, 159, 768, 118), (0, 202, 768, 161),
    (0, 261, 768, 211), (0, 335, 768, 279), (0, 425, 768, 368),
]

# ============================================================================
# 3. IMPORTACIONES DEL PROYECTO
# ============================================================================
from proyecto_manual import (
    seleccionar_linea_meta_manual,
    crear_mascara_piscina,
    CarrilesInclinadosDinamicos,
    LineaMetaTracker,
    SwimmerTracker
)

# ============================================================================
# 4. FUNCIÓN SELECCIÓN INTELIGENTE
# ============================================================================
def seleccionar_carriles_smart(video_path):
    print(f"\n🔍 Detectando estilo para: {video_path}")
    if 'mariposa1' in video_path: return LINEAS_MARIPOSA1
    elif 'mariposa2' in video_path: return LINEAS_MARIPOSA2
    elif 'braza' in video_path: return LINEAS_BRAZA
    elif 'espalda' in video_path: return LINEAS_ESPALDA
    elif 'libre' in video_path: return LINEAS_LIBRE
    return LINEAS_LIBRE

# ============================================================================
# 5. FUNCIÓN PRINCIPAL DE GRABACIÓN
# ============================================================================
def grabar_video_pro():
    cap = cv.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print("❌ Error abriendo video")
        return

    # Configuración video salida (Ancho original + Barra Lateral)
    fps = cap.get(cv.CAP_PROP_FPS)
    w_video = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    w_total = w_video + ANCHO_INTERFAZ
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (w_total, h_video))
    
    ret, first_frame = cap.read()
    if not ret: return

    # --- CONFIGURACIÓN INICIAL ---
    lineas = seleccionar_carriles_smart(VIDEO_INPUT)
    carriles = CarrilesInclinadosDinamicos(lineas, NUM_CARRILES)
    
    print("⚠️ Selecciona la META y la ROI en la ventana emergente...")
    meta = seleccionar_linea_meta_manual(first_frame)
    meta_tracker = LineaMetaTracker(first_frame, meta)
    roi = crear_mascara_piscina(first_frame)
    
    tracker = SwimmerTracker(NUM_CARRILES, DIRECCION, roi, carriles)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    
    print(f"\n🔴 GRABANDO VIDEO: {VIDEO_OUTPUT}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        
        # --- LÓGICA ---
        carriles.actualizar(frame)
        meta_now = meta_tracker.actualizar(frame)
        tracker.actualizar_zona_valida(frame.shape, meta_now)
        fg = tracker.bg_subtractor.apply(frame)
        nadadores = tracker.segmentar_nadadores(frame, fg)
        tracker.detectar_llegadas(nadadores, meta_now, frame_num)
        
        # --- VISUALIZACIÓN ---
        # 1. Crear lienzo grande (Negro)
        lienzo = np.zeros((h_video, w_total, 3), dtype=np.uint8)
        
        # 2. Dibujar sobre el frame de la piscina
        vis = frame.copy()
        
        # Zona excluida
        if tracker.mascara_zona_valida is not None:
            ov = vis.copy()
            excl = cv.bitwise_not(tracker.mascara_zona_valida)
            ov[excl > 0] = [0, 0, 255]
            cv.addWeighted(ov, 0.15, vis, 0.85, 0, vis)
            
        vis = carriles.dibujar_carriles(vis)
        
        # Meta
        cv.line(vis, (meta_now[0], meta_now[1]), (meta_now[2], meta_now[3]), (0, 255, 0), 3)
        for i in range(11):
            t = i / 10
            px = int(meta_now[0] + t * (meta_now[2] - meta_now[0]))
            py = int(meta_now[1] + t * (meta_now[3] - meta_now[1]))
            cv.circle(vis, (px, py), 5, (0, 255, 0), -1)
        cv.putText(vis, "META", (meta_now[0] + 10, meta_now[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Nadadores
        for c, r in tracker.rectangulos.items():
            if c in nadadores:
                cx, cy = nadadores[c]
                x1, y1, x2, y2 = r.obtener_rect_frontal()
                col = (0, 255, 0) if r.puede_detectar_llegada() else (0, 165, 255)
                cv.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                cv.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
                cv.line(vis, (x1, y1), (x1, y2), col, 3) # Línea frontal
                cv.putText(vis, f"C{c+1}", (cx, cy-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Pegar piscina en el lienzo
        lienzo[0:h_video, 0:w_video] = vis
        
        # --- 3. INTERFAZ LATERAL ---
        x_ui = w_video + 15
        y_ui = 40
        
        # Cabecera
        cv.putText(lienzo, "PHOTO-FINISH", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_ui += 30
        cv.putText(lienzo, "SYSTEM v1.0", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        y_ui += 40
        
        # Info Estado
        cv.putText(lienzo, f"Frame: {frame_num}/{total_frames}", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_ui += 25
        tiempo_actual = frame_num / fps
        cv.putText(lienzo, f"Tiempo: {tiempo_actual:.2f}s", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_ui += 40
        
        # Línea separadora
        cv.line(lienzo, (w_video, y_ui), (w_total, y_ui), (50, 50, 50), 1)
        y_ui += 30
        
        # Tabla Clasificación
        cv.putText(lienzo, "CLASIFICACION", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_ui += 30
        
        if len(tracker.llegadas) > 0:
            ranking = sorted(tracker.llegadas.items(), key=lambda x: x[1])
            for i, (c, f_llegada) in enumerate(ranking):
                t_final = f_llegada / fps
                
                # Colores Podio
                if i == 0: color = (0, 215, 255)      # Oro
                elif i == 1: color = (192, 192, 192)  # Plata
                elif i == 2: color = (42, 42, 165)    # Bronce
                else: color = (200, 200, 200)
                
                row_text = f"{i+1}. Carril {c+1}"
                time_text = f"{t_final:.2f}s"
                
                cv.putText(lienzo, row_text, (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                cv.putText(lienzo, time_text, (w_total - 75, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                y_ui += 25
        else:
            cv.putText(lienzo, "Esperando llegadas...", (x_ui, y_ui), cv.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Guardar y Mostrar
        out.write(lienzo)
        cv.imshow('Grabando Sistema', lienzo)
        
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("\n✅ GRABACIÓN COMPLETADA.")

if __name__ == "__main__":
    grabar_video_pro()

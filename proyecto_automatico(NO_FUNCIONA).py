import cv2 as cv
import numpy as np
from collections import defaultdict


# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

VIDEO_PATH = 'Proyecto/libre.mp4'
NUM_CARRILES = 8
DIRECCION_CARRERA = 'izquierda'
SELECCIONAR_ROI_MANUAL = True

DELAY_MS = 150
SELECCIONAR_LINEA_META_MANUAL = True

# PARÁMETROS DE DETECCIÓN DE NADADORES
AREA_MIN_NADADOR = 300
AREA_MAX_NADADOR = 20000
ASPECT_RATIO_MIN = 0.2
ASPECT_RATIO_MAX = 8.0
SOLIDITY_MIN = 0.3

MARGEN_EXCLUSION = 100
MARGEN_LLEGADA = 80

# VALIDACIÓN DE TRACKING
FRAMES_MINIMOS_VISIBLE = 10
FRAMES_MAX_PERDIDO = 3


# ============================================================================
# DETECCIÓN SIMPLE - EXTREMOS DE CADA SECCIÓN
# ============================================================================

def detectar_y_extender_lineas_simple(frame, mascara_piscina, num_carriles=8):
    """
    Detecta fragmentos con RANGO 2 [150-180] y traza una recta desde 
    el extremo izquierdo hasta el extremo derecho de cada sección.
    Con tolerancia adaptativa según profundidad (perspectiva).
    """
    print("\n" + "="*60)
    print("DETECCIÓN SIMPLE - EXTREMOS CON TOLERANCIA ADAPTATIVA")
    print("="*60)
    
    height, width = frame.shape[:2]
    
    # Convertir a HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Aplicar máscara de piscina
    if mascara_piscina is not None:
        hsv = cv.bitwise_and(hsv, hsv, mask=mascara_piscina)
    
    # SOLO RANGO 2: [150-180]
    lower2 = np.array([150, 50, 50])
    upper2 = np.array([180, 255, 255])
    mask2 = cv.inRange(hsv, lower2, upper2)
    
    cv.imshow('1. Mascara Original', mask2)
    
    print("✓ Detectando fragmentos...")
    
    # Encontrar contornos directamente (sin morfología)
    contornos, _ = cv.findContours(mask2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    print(f"✓ Fragmentos detectados: {len(contornos)}")
    
    # Filtrar contornos por área mínima
    fragmentos_validos = []
    
    for contorno in contornos:
        area = cv.contourArea(contorno)
        
        if area < 30:  # Reducido de 50 a 30 para capturar más fragmentos del fondo
            continue
        
        fragmentos_validos.append(contorno)
    
    print(f"✓ Fragmentos válidos: {len(fragmentos_validos)}")
    
    if len(fragmentos_validos) < 2:
        print("⚠️ Pocos fragmentos, usando distribución uniforme")
        return crear_lineas_uniformes(height, num_carriles)
    
    # Agrupar fragmentos por fila (posición Y similar)
    print("✓ Agrupando fragmentos por fila con tolerancia adaptativa...")
    
    # Calcular centroide Y de cada fragmento
    fragmentos_con_y = []
    
    for fragmento in fragmentos_validos:
        M = cv.moments(fragmento)
        if M['m00'] == 0:
            continue
        
        cy = int(M['m01'] / M['m00'])
        fragmentos_con_y.append((fragmento, cy))
    
    # Ordenar por Y
    fragmentos_con_y.sort(key=lambda x: x[1])
    
    # Agrupar fragmentos con TOLERANCIA ADAPTATIVA según profundidad
    grupos_filas = []
    grupo_actual = [fragmentos_con_y[0]]
    
    for i in range(1, len(fragmentos_con_y)):
        fragmento_prev = fragmentos_con_y[i-1]
        fragmento_curr = fragmentos_con_y[i]
        
        y_prev = fragmento_prev[1]
        y_curr = fragmento_curr[1]
        
        # TOLERANCIA ADAPTATIVA: más pequeña arriba (cerca), más grande abajo (lejos)
        # En el fondo (y grande), la perspectiva hace que las líneas estén más juntas
        # Normalizar Y entre 0 y 1
        y_norm = y_curr / height
        
        # Tolerancia inversamente proporcional a Y (más pequeña en el fondo)
        # Base: height / (num_carriles * 2)
        # Factor: empieza en 3.0 arriba, termina en 1.0 abajo
        factor_tolerancia = 3.0 - (2.0 * y_norm)  # 3.0 → 1.0
        tolerancia_y = (height / (num_carriles * 2)) * factor_tolerancia
        
        # Si están cerca en Y, pertenecen a la misma fila
        if abs(y_curr - y_prev) < tolerancia_y:
            grupo_actual.append(fragmento_curr)
            print(f"    Agrupando: y_prev={y_prev:.0f}, y_curr={y_curr:.0f}, diff={abs(y_curr-y_prev):.0f}, tolerancia={tolerancia_y:.0f}")
        else:
            grupos_filas.append(grupo_actual)
            grupo_actual = [fragmento_curr]
            print(f"    Nueva fila: y_prev={y_prev:.0f}, y_curr={y_curr:.0f}, diff={abs(y_curr-y_prev):.0f}, tolerancia={tolerancia_y:.0f}")
    
    # Agregar último grupo
    if grupo_actual:
        grupos_filas.append(grupo_actual)
    
    print(f"✓ Filas de fragmentos detectadas: {len(grupos_filas)}")
    
    # Mostrar detalle de cada fila
    for idx, fila in enumerate(grupos_filas):
        y_promedio_fila = np.mean([f[1] for f in fila])
        print(f"  Fila {idx+1}: {len(fila)} fragmentos, Y promedio={y_promedio_fila:.0f}")
    
    if len(grupos_filas) < 2:
        print("⚠️ Pocas filas detectadas, usando distribución uniforme")
        return crear_lineas_uniformes(height, num_carriles)
    
    # Para cada fila, encontrar extremos izquierdo y derecho
    lineas_detectadas = []
    
    for idx, fila in enumerate(grupos_filas):
        # Combinar todos los puntos de todos los fragmentos de esta fila
        todos_puntos = []
        
        for fragmento, _ in fila:
            puntos = fragmento.squeeze()
            
            if len(puntos.shape) == 1:
                puntos = puntos.reshape(1, -1)
            
            todos_puntos.extend(puntos)
        
        todos_puntos = np.array(todos_puntos)
        
        if len(todos_puntos) < 2:
            continue
        
        # Encontrar punto MÁS A LA IZQUIERDA (min X)
        punto_izq = todos_puntos[np.argmin(todos_puntos[:, 0])]
        x_izq, y_izq = punto_izq
        
        # Encontrar punto MÁS A LA DERECHA (max X)
        punto_der = todos_puntos[np.argmax(todos_puntos[:, 0])]
        x_der, y_der = punto_der
        
        # Calcular Y promedio de todos los puntos
        y_promedio = np.mean(todos_puntos[:, 1])
        
        # Ajustar línea recta entre estos dos puntos
        if x_der - x_izq > 0:
            m = (y_der - y_izq) / (x_der - x_izq)
            b = y_izq - m * x_izq
        else:
            # Línea vertical o punto único
            m = 0
            b = y_promedio
        
        # Extender la línea a todo el ancho del frame
        x_min_extend = 0
        x_max_extend = width - 1
        
        y_min = m * x_min_extend + b
        y_max = m * x_max_extend + b
        
        lineas_detectadas.append({
            'coefs': np.array([m, b]),  # [m, b] para y = m*x + b
            'y_promedio': y_promedio,
            'x_min': x_min_extend,
            'x_max': x_max_extend,
            'y_min': y_min,
            'y_max': y_max,
            'num_puntos': len(todos_puntos),
            'x_izq_orig': x_izq,
            'y_izq_orig': y_izq,
            'x_der_orig': x_der,
            'y_der_orig': y_der,
            'num_fragmentos': len(fila)
        })
        
        print(f"  Línea {idx+1}: y_prom={y_promedio:.1f}, extremos=[({x_izq:.0f},{y_izq:.0f}) → ({x_der:.0f},{y_der:.0f})], pendiente={m:.4f}, {len(fila)} fragmentos")
    
    if len(lineas_detectadas) == 0:
        print("⚠️ No se pudieron detectar líneas, usando distribución uniforme")
        return crear_lineas_uniformes(height, num_carriles)
    
    # Ordenar por Y promedio
    lineas_detectadas.sort(key=lambda l: l['y_promedio'])
    
    print(f"✓ Líneas detectadas: {len(lineas_detectadas)}")
    
    # Interpolar si faltan
    if len(lineas_detectadas) < num_carriles + 1:
        print(f"⚠️ Interpolando {num_carriles + 1 - len(lineas_detectadas)} líneas...")
        lineas_finales = interpolar_lineas_horizontal(lineas_detectadas, num_carriles, height, width)
    elif len(lineas_detectadas) > num_carriles + 1:
        print(f"⚠️ Seleccionando {num_carriles + 1} líneas uniformemente...")
        indices = np.linspace(0, len(lineas_detectadas) - 1, num_carriles + 1, dtype=int)
        lineas_finales = [lineas_detectadas[i] for i in indices]
    else:
        lineas_finales = lineas_detectadas
    
    print(f"✓ Líneas FINALES: {len(lineas_finales)}")
    print("="*60 + "\n")
    
    # Visualizar
    visualizar_lineas_extremos(frame, mask2, fragmentos_validos, lineas_finales)
    
    print("Presiona cualquier tecla para continuar...")
    cv.waitKey(0)
    
    # Cerrar ventanas
    cv.destroyWindow('1. Mascara Original')
    cv.destroyWindow('2. Fragmentos Detectados')
    cv.destroyWindow('3. Lineas desde Extremos')
    
    return lineas_finales

def visualizar_lineas_extremos(frame, mask, fragmentos, lineas_finales):
    """Visualiza fragmentos y líneas trazadas desde extremos"""
    
    height, width = frame.shape[:2]
    
    # Frame 1: Fragmentos detectados
    frame_fragmentos = frame.copy()
    
    colores = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), 
               (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
               (0, 128, 128), (128, 128, 0), (128, 0, 128), (64, 128, 255)]
    
    # Dibujar fragmentos
    for idx, fragmento in enumerate(fragmentos):
        color = colores[idx % len(colores)]
        cv.drawContours(frame_fragmentos, [fragmento], -1, color, 2)
    
    cv.putText(frame_fragmentos, f"Fragmentos: {len(fragmentos)}", (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    cv.imshow('2. Fragmentos Detectados', frame_fragmentos)
    
    # Frame 2: Líneas desde extremos
    frame_lineas = frame.copy()
    
    # Overlay de máscara
    mask_bgr = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    frame_lineas = cv.addWeighted(frame_lineas, 0.7, mask_bgr, 0.3, 0)
    
    # Dibujar líneas extendidas y puntos extremos originales
    for idx, linea in enumerate(lineas_finales):
        color = colores[idx % len(colores)]
        
        # Línea extendida
        x_min = int(linea['x_min'])
        x_max = int(linea['x_max'])
        y_min = int(np.clip(linea['y_min'], 0, height - 1))
        y_max = int(np.clip(linea['y_max'], 0, height - 1))
        
        cv.line(frame_lineas, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Marcar extremos ORIGINALES (antes de extender)
        if 'x_izq_orig' in linea:
            x_izq = int(linea['x_izq_orig'])
            y_izq = int(linea['y_izq_orig'])
            x_der = int(linea['x_der_orig'])
            y_der = int(linea['y_der_orig'])
            
            # Círculos en extremos originales
            cv.circle(frame_lineas, (x_izq, y_izq), 10, (0, 255, 0), -1)  # Verde = izquierda
            cv.circle(frame_lineas, (x_der, y_der), 10, (0, 0, 255), -1)  # Rojo = derecha
            
            # Línea gruesa entre extremos originales
            cv.line(frame_lineas, (x_izq, y_izq), (x_der, y_der), (255, 255, 255), 2)
        
        # Etiqueta
        x_centro = width // 2
        y_centro = int(linea['coefs'][0] * x_centro + linea['coefs'][1])
        cv.putText(frame_lineas, f"L{idx+1}", (x_centro + 10, y_centro),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    cv.putText(frame_lineas, f"Lineas: {len(lineas_finales)}", (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv.putText(frame_lineas, "Verde=Izq  Rojo=Der", (10, 60),
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv.imshow('3. Lineas desde Extremos', frame_lineas)


def crear_lineas_uniformes(height, num_carriles):
    """Crea líneas horizontales uniformes como fallback"""
    lineas = []
    
    for i in range(num_carriles + 1):
        y_pos = int(height * i / num_carriles)
        
        lineas.append({
            'coefs': np.array([0.0, y_pos]),
            'y_promedio': y_pos,
            'x_min': 0,
            'x_max': 0,
            'y_min': y_pos,
            'y_max': y_pos,
            'num_puntos': 0
        })
    
    return lineas


def interpolar_lineas_horizontal(lineas_existentes, num_carriles, height, width):
    """Interpola líneas faltantes manteniendo la perspectiva"""
    
    if len(lineas_existentes) < 2:
        return crear_lineas_uniformes(height, num_carriles)
    
    # Extraer posiciones Y promedio y pendientes
    y_promedios = [l['y_promedio'] for l in lineas_existentes]
    coefs_m = [l['coefs'][0] for l in lineas_existentes]
    coefs_b = [l['coefs'][1] for l in lineas_existentes]
    
    # Crear índices para interpolación
    indices_existentes = np.linspace(0, num_carriles, len(lineas_existentes))
    indices_necesarios = np.arange(num_carriles + 1)
    
    # Interpolar
    y_interpolados = np.interp(indices_necesarios, indices_existentes, y_promedios)
    m_interpolados = np.interp(indices_necesarios, indices_existentes, coefs_m)
    b_interpolados = np.interp(indices_necesarios, indices_existentes, coefs_b)
    
    # Crear líneas interpoladas
    lineas_finales = []
    
    for i, y_prom in enumerate(y_interpolados):
        m = m_interpolados[i]
        b = b_interpolados[i]
        
        # Extender a todo el ancho
        x_min = 0
        x_max = width - 1
        y_min = m * x_min + b
        y_max = m * x_max + b
        
        lineas_finales.append({
            'coefs': np.array([m, b]),
            'y_promedio': y_prom,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'num_puntos': 0
        })
    
    return lineas_finales


# ============================================================================
# TRACKING DE LÍNEAS CON CAMSHIFT
# ============================================================================

class LineasCorcherasTracker:
    """Tracker de líneas usando CAMShift"""
    
    def __init__(self, frame_inicial, lineas_iniciales, mascara_piscina):
        self.lineas_actuales = lineas_iniciales
        self.mascara_piscina = mascara_piscina
        
        height, width = frame_inicial.shape[:2]
        self.height = height
        self.width = width
        
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        
        self.track_windows = []
        self.roi_hists = []
        
        hsv_inicial = cv.cvtColor(frame_inicial, cv.COLOR_BGR2HSV)
        
        for linea in lineas_iniciales:
            y_centro = int(linea['y_promedio'])
            
            x = 0
            y = max(0, y_centro - 15)
            w = width
            h = 30
            
            self.track_windows.append((x, y, w, h))
            
            roi = hsv_inicial[y:y+h, x:x+w]
            
            lower2 = np.array([150, 50, 50])
            upper2 = np.array([180, 255, 255])
            mask_roi = cv.inRange(roi, lower2, upper2)
            
            roi_hist = cv.calcHist([roi], [0], mask_roi, [180], [0, 180])
            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)
            
            self.roi_hists.append(roi_hist)
    
    def actualizar(self, frame_actual):
        """Actualiza las líneas usando CAMShift"""
        hsv = cv.cvtColor(frame_actual, cv.COLOR_BGR2HSV)
        
        lower2 = np.array([150, 50, 50])
        upper2 = np.array([180, 255, 255])
        mask_hue = cv.inRange(hsv, lower2, upper2)
        
        if self.mascara_piscina is not None:
            mask_hue = cv.bitwise_and(mask_hue, mask_hue, mask=self.mascara_piscina)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        mask_hue = cv.morphologyEx(mask_hue, cv.MORPH_CLOSE, kernel, iterations=2)
        
        nuevas_lineas = []
        
        for i, (track_window, roi_hist, linea_anterior) in enumerate(zip(self.track_windows, 
                                                                          self.roi_hists, 
                                                                          self.lineas_actuales)):
            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            
            try:
                ret, new_track_window = cv.CamShift(dst, track_window, self.term_crit)
                
                self.track_windows[i] = new_track_window
                
                x, y, w, h = new_track_window
                y_centro_nuevo = y + h // 2
                
                y_min_region = max(0, y - 20)
                y_max_region = min(self.height, y + h + 20)
                
                region_mask = np.zeros_like(mask_hue)
                region_mask[y_min_region:y_max_region, :] = mask_hue[y_min_region:y_max_region, :]
                
                puntos = np.column_stack(np.where(region_mask > 0))
                
                if len(puntos) >= 10:
                    y_vals = puntos[:, 0].astype(np.float32)
                    x_vals = puntos[:, 1].astype(np.float32)
                    
                    try:
                        # y = m*x + b
                        coefs = np.polyfit(x_vals, y_vals, deg=1)
                        m, b = coefs
                        
                        # Extender horizontalmente
                        x_min_ext = 0
                        x_max_ext = self.width - 1
                        y_min_ext = m * x_min_ext + b
                        y_max_ext = m * x_max_ext + b
                        
                        nuevas_lineas.append({
                            'coefs': coefs,
                            'y_promedio': y_centro_nuevo,
                            'x_min': x_min_ext,
                            'x_max': x_max_ext,
                            'y_min': y_min_ext,
                            'y_max': y_max_ext,
                            'num_puntos': len(puntos)
                        })
                    except:
                        linea_anterior['y_promedio'] = y_centro_nuevo
                        nuevas_lineas.append(linea_anterior)
                else:
                    linea_anterior['y_promedio'] = y_centro_nuevo
                    nuevas_lineas.append(linea_anterior)
                    
            except:
                nuevas_lineas.append(linea_anterior)
        
        nuevas_lineas.sort(key=lambda l: l['y_promedio'])
        self.lineas_actuales = nuevas_lineas
        
        return self.lineas_actuales
    
    def obtener_limites_y_en_x(self, x_coord):
        """Dado una coordenada X, retorna las posiciones Y de cada carril"""
        limites_y = []
        
        for linea in self.lineas_actuales:
            coefs = linea['coefs']
            
            if len(coefs) == 2:
                m, b = coefs
                # y = m*x + b
                y_en_x = m * x_coord + b
                y_en_x = np.clip(int(y_en_x), 0, self.height - 1)
                limites_y.append(y_en_x)
            else:
                limites_y.append(int(linea['y_promedio']))
        
        return sorted(limites_y)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def crear_mascara_zona_valida_rapida(frame_shape, linea_meta, direccion='izquierda', margen_exclusion=50):
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = linea_meta
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    lado = A * x_coords + B * y_coords + C
    mascara_zona = np.ones((height, width), dtype=np.uint8) * 255
    if direccion == 'izquierda':
        mascara_zona[lado < margen_exclusion * np.sqrt(A**2 + B**2) / 10] = 0
    else:
        mascara_zona[lado > -margen_exclusion * np.sqrt(A**2 + B**2) / 10] = 0
    return mascara_zona


def seleccionar_linea_meta_manual(frame):
    print("\n" + "="*60)
    print("SELECCIÓN MANUAL DE LA LÍNEA DE META")
    print("="*60)
    print("Instrucciones:")
    print("1. Haz clic en el PUNTO SUPERIOR de la línea de meta")
    print("2. Haz clic en el PUNTO INFERIOR de la línea de meta")
    print("3. Presiona 'r' para reiniciar si te equivocas")
    print("4. Presiona ENTER para confirmar")
    print("="*60 + "\n")
    
    puntos = []
    
    def click_evento(event, x, y, flags, param):
        nonlocal puntos
        if event == cv.EVENT_LBUTTONDOWN:
            if len(puntos) < 2:
                puntos.append((x, y))
                cv.circle(frame_copy, (x, y), 8, (0, 255, 255), -1)
                
                if len(puntos) == 1:
                    cv.putText(frame_copy, "Punto 1 (Superior)", (x + 15, y - 10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif len(puntos) == 2:
                    cv.putText(frame_copy, "Punto 2 (Inferior)", (x + 15, y + 20),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv.line(frame_copy, puntos[0], puntos[1], (0, 255, 0), 3)
                    cv.putText(frame_copy, "LINEA DE META", 
                              (puntos[0][0] + 15, (puntos[0][1] + puntos[1][1]) // 2),
                              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                cv.imshow('Seleccionar Linea de Meta', frame_copy)
    
    frame_copy = frame.copy()
    cv.namedWindow('Seleccionar Linea de Meta')
    cv.setMouseCallback('Seleccionar Linea de Meta', click_evento)
    cv.imshow('Seleccionar Linea de Meta', frame_copy)
    
    while True:
        key = cv.waitKey(1) & 0xFF
        
        if key == 13:
            if len(puntos) == 2:
                break
            else:
                print("Necesitas seleccionar 2 puntos.")
        elif key == ord('r'):
            puntos = []
            frame_copy = frame.copy()
            cv.imshow('Seleccionar Linea de Meta', frame_copy)
            print("Selección reiniciada.")
    
    cv.destroyWindow('Seleccionar Linea de Meta')
    
    x1, y1 = puntos[0]
    x2, y2 = puntos[1]
    
    height = frame.shape[0]
    
    if x2 - x1 == 0:
        return (x1, 0, x1, height)
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    y_top = 0
    y_bottom = height
    x_top = (y_top - b) / m if m != 0 else x1
    x_bottom = (y_bottom - b) / m if m != 0 else x1
    
    linea_extendida = (int(x_top), int(y_top), int(x_bottom), int(y_bottom))
    
    print(f"Línea de meta seleccionada: {linea_extendida}")
    
    return linea_extendida


class LineaMetaTracker:
    def __init__(self, frame_inicial, linea_inicial):
        self.linea_actual = linea_inicial
        
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        x1, y1, x2, y2 = linea_inicial
        num_puntos = 10
        
        self.puntos_linea = []
        for i in range(num_puntos):
            t = i / (num_puntos - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self.puntos_linea.append([x, y])
        
        self.puntos_linea = np.array(self.puntos_linea, dtype=np.float32).reshape(-1, 1, 2)
        self.frame_anterior = cv.cvtColor(frame_inicial, cv.COLOR_BGR2GRAY)
        self.frames_sin_actualizar = 0
        self.max_frames_sin_actualizar = 30
        
    def actualizar(self, frame_actual):
        frame_gray = cv.cvtColor(frame_actual, cv.COLOR_BGR2GRAY)
        
        nuevos_puntos, status, err = cv.calcOpticalFlowPyrLK(
            self.frame_anterior,
            frame_gray,
            self.puntos_linea,
            None,
            **self.lk_params
        )
        
        if nuevos_puntos is not None and status is not None:
            buenos_nuevos = nuevos_puntos[status == 1]
            
            if len(buenos_nuevos) >= 4:
                self.puntos_linea = nuevos_puntos
                
                puntos_2d = nuevos_puntos.reshape(-1, 2)
                
                if len(puntos_2d) >= 2:
                    coefs = np.polyfit(puntos_2d[:, 1], puntos_2d[:, 0], 1)
                    m = coefs[0]
                    b = coefs[1]
                    
                    height = frame_actual.shape[0]
                    y1 = 0
                    y2 = height
                    x1 = int(m * y1 + b)
                    x2 = int(m * y2 + b)
                    
                    self.linea_actual = (x1, y1, x2, y2)
                    self.frames_sin_actualizar = 0
                else:
                    self.frames_sin_actualizar += 1
            else:
                self.frames_sin_actualizar += 1
        else:
            self.frames_sin_actualizar += 1
        
        self.frame_anterior = frame_gray.copy()
        
        if self.frames_sin_actualizar > self.max_frames_sin_actualizar:
            self._reinicializar_puntos(frame_actual)
            self.frames_sin_actualizar = 0
        
        return self.linea_actual
    
    def _reinicializar_puntos(self, frame):
        x1, y1, x2, y2 = self.linea_actual
        num_puntos = 10
        
        self.puntos_linea = []
        for i in range(num_puntos):
            t = i / (num_puntos - 1)
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self.puntos_linea.append([x, y])
        
        self.puntos_linea = np.array(self.puntos_linea, dtype=np.float32).reshape(-1, 1, 2)
        self.frame_anterior = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def crear_mascara_piscina(frame, manual=True):
    if manual:
        print("\n" + "="*60)
        print("SELECCIÓN DE ÁREA DE LA PISCINA")
        print("="*60)
        print("Instrucciones:")
        print("1. Haz clic en las ESQUINAS de la piscina en orden")
        print("2. Presiona ENTER cuando hayas terminado")
        print("3. Presiona 'r' para reiniciar la selección")
        print("="*60 + "\n")
        
        puntos = []
        
        def click_evento(event, x, y, flags, param):
            nonlocal puntos
            if event == cv.EVENT_LBUTTONDOWN:
                puntos.append((x, y))
                cv.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
                cv.putText(frame_copy, str(len(puntos)), (x + 10, y - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if len(puntos) > 1:
                    cv.line(frame_copy, puntos[-2], puntos[-1], (0, 255, 0), 2)
                
                cv.imshow('Seleccionar Piscina', frame_copy)
        
        frame_copy = frame.copy()
        cv.namedWindow('Seleccionar Piscina')
        cv.setMouseCallback('Seleccionar Piscina', click_evento)
        cv.imshow('Seleccionar Piscina', frame_copy)
        
        while True:
            key = cv.waitKey(1) & 0xFF
            
            if key == 13:
                if len(puntos) >= 3:
                    break
                else:
                    print("Necesitas al menos 3 puntos.")
            elif key == ord('r'):
                puntos = []
                frame_copy = frame.copy()
                cv.imshow('Seleccionar Piscina', frame_copy)
        
        cv.destroyWindow('Seleccionar Piscina')
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        puntos_array = np.array(puntos, dtype=np.int32)
        cv.fillPoly(mask, [puntos_array], 255)
        
        return mask


# ============================================================================
# TRACKING DE NADADORES
# ============================================================================

class RectanguloNadador:
    def __init__(self, carril, cx, cy, y_top, y_bottom, frame_width):
        self.carril = carril
        self.cx = cx
        self.cy = cy
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.frame_width = frame_width
        
        altura_carril = abs(y_bottom - y_top)
        self.altura_rect = max(10, int(altura_carril * 0.4))
        
        x_norm = cx / frame_width if frame_width > 0 else 0.5
        self.ancho_rect = max(8, int(40 * (1 - x_norm * 0.3)))
        
        self.frames_visible = 1
        self.frames_sin_ver = 0
        self.ha_llegado = False
        self.frames_en_llegada = 0
    
    def actualizar(self, cx, cy, y_top, y_bottom):
        self.cx = cx
        self.cy = cy
        self.y_top = y_top
        self.y_bottom = y_bottom
        
        altura_carril = abs(y_bottom - y_top)
        self.altura_rect = max(10, int(altura_carril * 0.4))
        
        x_norm = cx / self.frame_width if self.frame_width > 0 else 0.5
        self.ancho_rect = max(8, int(40 * (1 - x_norm * 0.3)))
        
        self.frames_visible += 1
        self.frames_sin_ver = 0
    
    def no_detectado(self):
        self.frames_sin_ver += 1
        if self.frames_sin_ver > FRAMES_MAX_PERDIDO:
            self.frames_visible = 0
    
    def obtener_rect_frontal(self):
        x_izq = int(self.cx - self.ancho_rect // 2)
        x_der = int(self.cx + self.ancho_rect // 2)
        y_sup = int(self.cy - self.altura_rect // 2)
        y_inf = int(self.cy + self.altura_rect // 2)
        
        return x_izq, y_sup, x_der, y_inf
    
    def obtener_punto_frontal(self):
        x_izq, _, _, _ = self.obtener_rect_frontal()
        return x_izq
    
    def rect_cruza_linea(self, linea_meta):
        x1, y1, x2, y2 = linea_meta
        
        x_frontal = self.obtener_punto_frontal()
        y_centro = int(self.cy)
        
        if x2 - x1 == 0:
            linea_x = x1
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            linea_x = (y_centro - b) / m if m != 0 else x1
        
        ha_cruzado = x_frontal <= linea_x + MARGEN_LLEGADA
        
        return ha_cruzado, x_frontal
    
    def validar_llegada(self, ha_cruzado):
        if ha_cruzado:
            self.frames_en_llegada += 1
            if self.frames_en_llegada >= 5:
                return True
        else:
            self.frames_en_llegada = 0
        
        return False
    
    def puede_detectar_llegada(self):
        return self.frames_visible >= FRAMES_MINIMOS_VISIBLE


class SwimmerTracker:
    def __init__(self, num_carriles, direccion='izquierda', mascara_piscina=None, lineas_tracker=None):
        self.num_carriles = num_carriles
        self.direccion = direccion
        self.mascara_piscina = mascara_piscina
        self.lineas_tracker = lineas_tracker
        self.llegadas = {}
        
        self.rectangulos = {}
        
        self.mascara_zona_valida = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=60,
            detectShadows=False
        )
    
    def actualizar_zona_valida(self, frame_shape, linea_meta):
        self.mascara_zona_valida = crear_mascara_zona_valida_rapida(
            frame_shape, 
            linea_meta, 
            self.direccion, 
            margen_exclusion=MARGEN_EXCLUSION
        )
    
    def segmentar_nadadores(self, frame, fg_mask):
        
        if self.mascara_piscina is not None:
            fg_mask = cv.bitwise_and(fg_mask, fg_mask, mask=self.mascara_piscina)
        
        fg_mask_busqueda = fg_mask.copy()
        if self.mascara_zona_valida is not None:
            fg_mask_busqueda = cv.bitwise_and(fg_mask_busqueda, fg_mask_busqueda, mask=self.mascara_zona_valida)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        
        fg_mask_busqueda = cv.morphologyEx(fg_mask_busqueda, cv.MORPH_OPEN, kernel, iterations=2)
        fg_mask_busqueda = cv.morphologyEx(fg_mask_busqueda, cv.MORPH_CLOSE, kernel, iterations=2)
        
        contours_busqueda, _ = cv.findContours(fg_mask_busqueda, cv.RETR_EXTERNAL, 
                                               cv.CHAIN_APPROX_SIMPLE)
        
        # Obtener límites de carriles
        x_promedio = frame.shape[1] // 2
        limites_y = self.lineas_tracker.obtener_limites_y_en_x(x_promedio)
        
        nadadores_por_carril = {}
        
        for i in range(self.num_carriles):
            y_top = limites_y[i]
            y_bottom = limites_y[i + 1]
            
            mejor_contorno = None
            mejor_area = 0
            mejor_centroid = None
            
            for contour in contours_busqueda:
                area = cv.contourArea(contour)
                
                if area < AREA_MIN_NADADOR or area > AREA_MAX_NADADOR:
                    continue
                
                x, y, w, h = cv.boundingRect(contour)
                
                if h == 0:
                    continue
                    
                aspect_ratio = w / h
                
                if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
                    continue
                
                # Filtro anti-brillos
                mascara_contorno = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv.drawContours(mascara_contorno, [contour], -1, 255, -1)
                intensidad_media = cv.mean(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), mask=mascara_contorno)[0]
                
                if intensidad_media > 210:
                    continue
                
                M = cv.moments(contour)
                if M['m00'] == 0:
                    continue
                
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                if not (y_top < cy < y_bottom):
                    continue
                
                hull = cv.convexHull(contour)
                hull_area = cv.contourArea(hull)
                
                if hull_area > 0:
                    solidity = area / hull_area
                    if solidity < SOLIDITY_MIN:
                        continue
                
                if area > mejor_area:
                    mejor_area = area
                    mejor_centroid = (cx, cy)
            
            if mejor_centroid is not None:
                cx, cy = mejor_centroid
                
                if i not in self.rectangulos:
                    self.rectangulos[i] = RectanguloNadador(i, cx, cy, y_top, y_bottom, frame.shape[1])
                else:
                    self.rectangulos[i].actualizar(cx, cy, y_top, y_bottom)
                
                nadadores_por_carril[i] = (cx, cy)
        
        return nadadores_por_carril, limites_y
    
    def detectar_llegadas(self, nadadores, linea_meta, frame_num):
        
        for carril in range(self.num_carriles):
            if carril not in nadadores and carril in self.rectangulos:
                self.rectangulos[carril].no_detectado()
        
        for carril, (cx, cy) in nadadores.items():
            if carril in self.llegadas:
                continue
            
            if carril not in self.rectangulos:
                continue
            
            rect = self.rectangulos[carril]
            
            if not rect.puede_detectar_llegada():
                continue
            
            ha_cruzado, x_frontal = rect.rect_cruza_linea(linea_meta)
            
            if rect.validar_llegada(ha_cruzado):
                self.llegadas[carril] = frame_num
                print(f"\n🏁 ¡LLEGADA! Nadador del carril {carril + 1} en frame {frame_num}!\n")
    
    def obtener_ranking(self):
        if not self.llegadas:
            return []
        
        ranking = sorted(self.llegadas.items(), key=lambda x: x[1])
        return [(carril + 1, frame) for carril, frame in ranking]


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def procesar_video_natacion(video_path, direccion='izquierda', seleccionar_roi=True, 
                           delay_ms=50, seleccionar_meta_manual=True):
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video")
        return
    
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    print(f"\n" + "="*60)
    print("SISTEMA DE FOTOFINISH PARA NATACIÓN")
    print("="*60)
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Método: DETECCIÓN POR EXTREMOS + EXTENSIÓN HORIZONTAL")
    print("="*60 + "\n")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return
    
    # PASO 1: Seleccionar máscara de piscina
    mascara_piscina = crear_mascara_piscina(first_frame, manual=seleccionar_roi)
    
    # PASO 2: Detectar líneas por extremos
    lineas_iniciales = detectar_y_extender_lineas_simple(first_frame, mascara_piscina, NUM_CARRILES)
    
    # PASO 3: Inicializar tracker
    lineas_tracker = LineasCorcherasTracker(first_frame, lineas_iniciales, mascara_piscina)
    
    # PASO 4: Seleccionar línea de meta
    linea_meta_inicial = seleccionar_linea_meta_manual(first_frame)
    linea_meta_tracker = LineaMetaTracker(first_frame, linea_meta_inicial)
    
    # PASO 5: Inicializar tracker de nadadores
    tracker = SwimmerTracker(NUM_CARRILES, direccion, mascara_piscina, lineas_tracker)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    
    print("\n🏊 Procesando video... (Presiona 'q' para salir)\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        linea_meta = linea_meta_tracker.actualizar(frame)
        lineas_actualizadas = lineas_tracker.actualizar(frame)
        
        tracker.actualizar_zona_valida(frame.shape, linea_meta)
        
        fg_mask = tracker.bg_subtractor.apply(frame)
        nadadores, limites_y = tracker.segmentar_nadadores(frame, fg_mask)
        tracker.detectar_llegadas(nadadores, linea_meta, frame_num)
        
        # Visualización
        frame_vis = frame.copy()
        
        if tracker.mascara_zona_valida is not None:
            overlay = frame_vis.copy()
            zona_excluida = cv.bitwise_not(tracker.mascara_zona_valida)
            overlay[zona_excluida > 0] = [0, 0, 255]
            cv.addWeighted(overlay, 0.15, frame_vis, 0.85, 0, frame_vis)
        
        x1, y1, x2, y2 = linea_meta
        cv.line(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        num_puntos = 10
        for i in range(num_puntos + 1):
            t = i / num_puntos
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            cv.circle(frame_vis, (px, py), 5, (0, 255, 0), -1)
        
        cv.putText(frame_vis, "META", (x1 + 10, y1 + 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Dibujar líneas HORIZONTALES
        for idx, linea in enumerate(lineas_actualizadas):
            x_min = int(linea['x_min'])
            x_max = int(linea['x_max'])
            y_min = int(np.clip(linea['y_min'], 0, height - 1))
            y_max = int(np.clip(linea['y_max'], 0, height - 1))
            
            cv.line(frame_vis, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            if idx < NUM_CARRILES:
                x_centro = width // 2
                y_centro = int(linea['coefs'][0] * x_centro + linea['coefs'][1])
                cv.putText(frame_vis, f"C{idx + 1}", (width - 50, y_centro),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Nadadores
        for carril, (cx, cy) in nadadores.items():
            rect = tracker.rectangulos[carril]
            x_izq, y_sup, x_der, y_inf = rect.obtener_rect_frontal()
            
            if rect.puede_detectar_llegada():
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)
            
            cv.rectangle(frame_vis, (x_izq, y_sup), (x_der, y_inf), color, 2)
            cv.circle(frame_vis, (cx, cy), 5, (0, 0, 255), -1)
            cv.line(frame_vis, (x_izq, y_sup), (x_izq, y_inf), color, 3)
            
            x_frontal = x_izq
            if x2 - x1 == 0:
                linea_x = x1
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                linea_x = (cy - b) / m if m != 0 else x1
            
            distancia = abs(x_frontal - linea_x)
            cv.line(frame_vis, (int(x_frontal), int(cy)), (int(linea_x), int(cy)), 
                   (255, 0, 0), 2)
            
            texto_distancia = f"Dist: {int(distancia)}px"
            color_texto = (0, 255, 0) if distancia < MARGEN_LLEGADA else (0, 165, 255)
            cv.putText(frame_vis, texto_distancia, (cx - 40, cy + 30),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 2)
            
            texto = f"C{carril + 1} [{rect.frames_visible}]"
            cv.putText(frame_vis, texto, (cx + 15, cy),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        llegadas_text = f"Llegadas: {len(tracker.llegadas)}"
        cv.putText(frame_vis, f"Frame: {frame_num}/{total_frames} | Nadadores: {len(nadadores)} | {llegadas_text}", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv.imshow('Fotofinish Natacion', frame_vis)
        cv.imshow('Mascara Foreground', fg_mask)
        
        if cv.waitKey(delay_ms) & 0xFF == ord('q'):
            break
    
    ranking = tracker.obtener_ranking()
    
    print("\n" + "="*60)
    print("🏆 RANKING FINAL - FOTOFINISH")
    print("="*60)
    
    if ranking:
        for posicion, (carril, frame) in enumerate(ranking, 1):
            tiempo = frame / fps
            print(f"{posicion}º lugar: Carril {carril} - Frame {frame} - Tiempo: {tiempo:.3f}s")
    else:
        print("⚠️ No se detectaron llegadas")
    
    print("="*60 + "\n")
    
    cap.release()
    cv.destroyAllWindows()
    
    return ranking


if __name__ == "__main__":
    ranking = procesar_video_natacion(
        VIDEO_PATH, 
        direccion=DIRECCION_CARRERA,
        seleccionar_roi=SELECCIONAR_ROI_MANUAL,
        delay_ms=DELAY_MS,
        seleccionar_meta_manual=SELECCIONAR_LINEA_META_MANUAL
    )

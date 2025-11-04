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
# DETECCIÓN CON HOUGH LINES + AGRUPACIÓN
# ============================================================================

def detectar_carriles_hough_standard(frame, mascara_piscina=None, num_carriles=8):
    """Detecta carriles usando Hough Lines"""
    print("\n" + "="*60)
    print("DETECCIÓN DE CARRILES - HOUGH LINES")
    print("="*60)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    if mascara_piscina is not None:
        gray = cv.bitwise_and(gray, gray, mask=mascara_piscina)
    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    blurred = cv.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv.Canny(blurred, 20, 80)
    
    cv.imshow('1. Canny Original', edges)
    
    height, width = edges.shape
    
    mascara_busqueda = np.ones(edges.shape, dtype=np.uint8) * 255
    mascara_busqueda[:, :int(width*0.15)] = 0
    edges = cv.bitwise_and(edges, edges, mask=mascara_busqueda)
    
    print("\n📐 Detectando líneas con Hough Lines...")
    
    lines = cv.HoughLines(edges, rho=1, theta=np.pi/180, threshold=55)
    
    if lines is None or len(lines) < 5:
        print("⚠️ Pocas líneas detectadas, usando valores por defecto")
        return [int(height * i / num_carriles) for i in range(num_carriles + 1)]
    
    print(f"✓ Líneas detectadas: {len(lines)}")
    
    lineas_xy = []
    
    for line in lines:
        rho, theta = line[0]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            b_line = y1 - m * x1
            x_ref = width // 4
            y_ref = m * x_ref + b_line
        else:
            y_ref = (y1 + y2) // 2
        
        y_ref = np.clip(int(y_ref), 0, height - 1)
        angulo = theta * 180 / np.pi
        
        if abs(angulo - 90) < 20:
            lineas_xy.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'y_ref': y_ref,
                'rho': rho,
                'theta': theta,
                'angulo': angulo
            })
    
    lineas_xy.sort(key=lambda l: l['y_ref'])
    
    print(f"✓ Líneas casi-verticales: {len(lineas_xy)}")
    
    print("\n🔗 Agrupando líneas similares...")
    
    grupos = []
    usado = set()
    
    for i, linea_i in enumerate(lineas_xy):
        if i in usado:
            continue
        
        grupo = [linea_i]
        usado.add(i)
        
        distancia_max = height // 10
        
        for j, linea_j in enumerate(lineas_xy[i+1:], start=i+1):
            if j in usado:
                continue
            
            distancia_y = abs(linea_j['y_ref'] - linea_i['y_ref'])
            
            if distancia_y < distancia_max:
                grupo.append(linea_j)
                usado.add(j)
        
        grupos.append(grupo)
    
    grupos = [g for g in grupos if len(g) >= 1]
    print(f"✓ Grupos encontrados: {len(grupos)}")
    
    lineas_finales = []
    
    for grupo in grupos:
        y_valores = [l['y_ref'] for l in grupo]
        y_promedio = int(np.mean(y_valores))
        lineas_finales.append(y_promedio)
    
    lineas_finales.sort()
    
    print(f"✓ Líneas antes de interpolación: {len(lineas_finales)}")
    
    if len(lineas_finales) < num_carriles + 1:
        print(f"⚠️ Interpolando líneas faltantes...")
        
        if len(lineas_finales) < 2:
            lineas_finales = [int(height * i / num_carriles) for i in range(num_carriles + 1)]
        else:
            x_actual = np.arange(len(lineas_finales))
            x_nuevo = np.linspace(0, len(lineas_finales) - 1, num_carriles + 1)
            lineas_interpoladas = np.interp(x_nuevo, x_actual, lineas_finales)
            lineas_finales = [int(y) for y in lineas_interpoladas]
    
    elif len(lineas_finales) > num_carriles + 1:
        print(f"⚠️ Seleccionando líneas uniformemente...")
        
        indices = np.linspace(0, len(lineas_finales) - 1, num_carriles + 1, dtype=int)
        lineas_finales = [lineas_finales[i] for i in indices]
    
    for i in range(1, len(lineas_finales)):
        if lineas_finales[i] <= lineas_finales[i-1]:
            lineas_finales[i] = lineas_finales[i-1] + 1
    
    lineas_finales = [np.clip(int(y), 0, height - 1) for y in lineas_finales]
    
    print(f"\n✓ Líneas FINALES: {len(lineas_finales)}")
    print("="*60 + "\n")
    
    visualizar_hough_standard(frame, lineas_xy, grupos, lineas_finales)
    
    print("Presiona cualquier tecla para continuar...")
    cv.waitKey(0)
    cv.destroyWindow('1. Canny Original')
    try:
        cv.destroyWindow('2. Hough Lines Grupos')
    except:
        pass
    
    return lineas_finales

def visualizar_hough_standard(frame, lineas_xy, grupos, lineas_finales):
    """Visualiza las líneas detectadas"""
    
    frame_viz = frame.copy()
    
    colores = [
        (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 0, 255), (255, 128, 0),
    ]
    
    for grupo_idx, grupo in enumerate(grupos):
        color = colores[grupo_idx % len(colores)]
        
        for linea in grupo:
            x1, y1, x2, y2 = linea['x1'], linea['y1'], linea['x2'], linea['y2']
            
            y1 = np.clip(y1, 0, frame.shape[0] - 1)
            y2 = np.clip(y2, 0, frame.shape[0] - 1)
            x1 = np.clip(x1, 0, frame.shape[1] - 1)
            x2 = np.clip(x2, 0, frame.shape[1] - 1)
            
            cv.line(frame_viz, (x1, y1), (x2, y2), color, 2)
    
    for linea_y in lineas_finales:
        cv.line(frame_viz, (0, linea_y), (frame.shape[1], linea_y), (0, 0, 255), 3)
    
    cv.putText(frame_viz, f"Grupos: {len(grupos)}", (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv.putText(frame_viz, f"Lineas finales: {len(lineas_finales)}", (10, 60),
              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv.imshow('2. Hough Lines Grupos', frame_viz)

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
# TRACKING POR RECTÁNGULO ADAPTATIVO
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
    def __init__(self, num_carriles, direccion='izquierda', mascara_piscina=None, limites_carriles=None):
        self.num_carriles = num_carriles
        self.direccion = direccion
        self.mascara_piscina = mascara_piscina
        self.limites_carriles = limites_carriles
        self.llegadas = {}
        
        self.rectangulos = {}
        
        self.mascara_zona_valida = None
        self.term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
        
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=25,
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
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv.morphologyEx(fg_mask, cv.MORPH_CLOSE, kernel, iterations=2)
        
        fg_mask_busqueda = cv.morphologyEx(fg_mask_busqueda, cv.MORPH_OPEN, kernel, iterations=1)
        fg_mask_busqueda = cv.morphologyEx(fg_mask_busqueda, cv.MORPH_CLOSE, kernel, iterations=2)
        
        contours_busqueda, _ = cv.findContours(fg_mask_busqueda, cv.RETR_EXTERNAL, 
                                               cv.CHAIN_APPROX_SIMPLE)
        
        nadadores_por_carril = {}
        
        for i in range(self.num_carriles):
            y_top = self.limites_carriles[i]
            y_bottom = self.limites_carriles[i + 1]
            
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
        
        return nadadores_por_carril
    
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
    print(f"Método: Hough Lines Automático + Rectángulos Adaptativos")
    print("="*60 + "\n")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return
    
    mascara_piscina = crear_mascara_piscina(first_frame, manual=seleccionar_roi)
    limites_carriles = detectar_carriles_hough_standard(first_frame, mascara_piscina, NUM_CARRILES)
    linea_meta_inicial = seleccionar_linea_meta_manual(first_frame)
    linea_meta_tracker = LineaMetaTracker(first_frame, linea_meta_inicial)
    tracker = SwimmerTracker(NUM_CARRILES, direccion, mascara_piscina, limites_carriles)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    
    print("\n🏊 Procesando video... (Presiona 'q' para salir)\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        linea_meta = linea_meta_tracker.actualizar(frame)
        tracker.actualizar_zona_valida(frame.shape, linea_meta)
        
        fg_mask = tracker.bg_subtractor.apply(frame)
        nadadores = tracker.segmentar_nadadores(frame, fg_mask)
        tracker.detectar_llegadas(nadadores, linea_meta, frame_num)
        
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
        
        for idx, y in enumerate(limites_carriles):
            cv.line(frame_vis, (0, y), (width, y), (0, 255, 255), 2)
            if idx < NUM_CARRILES:
                y_centro = (limites_carriles[idx] + limites_carriles[idx + 1]) // 2
                cv.putText(frame_vis, f"C{idx + 1}", (width - 50, y_centro),
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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
            
            texto_tamaño = f"H:{rect.altura_rect} W:{rect.ancho_rect}"
            cv.putText(frame_vis, texto_tamaño, (x_izq, y_sup - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
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

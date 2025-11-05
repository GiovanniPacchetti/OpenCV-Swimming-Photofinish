import cv2 as cv
import numpy as np

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

VIDEO_PATH = 'Proyecto/libre.mp4'
NUM_CARRILES = 7
DIRECCION_CARRERA = 'izquierda'

DELAY_MS = 150

# PARÁMETROS DE DETECCIÓN
AREA_MIN_NADADOR_LEJANO = 80
AREA_MAX_NADADOR = 20000
ASPECT_RATIO_MIN = 0.2
ASPECT_RATIO_MAX = 8.0
SOLIDITY_MIN_LEJANO = 0.10
AREA_MIN_NADADOR = 300
SOLIDITY_MIN = 0.3

# VALIDACIÓN
MARGEN_LLEGADA = 50
FRAMES_MINIMOS_VISIBLE = 10  # 35   libre/esplada =10 // mariposa1/mariposa2 = 35/30 (cambiar valor segun la distancia del final de la carrera)
FRAMES_MAX_PERDIDO = 3

MARGEN_EXCLUSION = 100

# ============================================================================
# COORDENADAS HARDCODEADAS DE CARRILES
# ============================================================================

LINEAS_CARRILES_HARDCODED = [
    (0, 52, 768, 34),      # Línea 1: ARRIBA
    (0, 69, 768, 57),      # Línea 2
    (0, 100, 768, 79),     # Línea 3
    (0, 130, 768, 116),    # Línea 4
    (0, 168, 768, 149),    # Línea 5
    (0, 216, 768, 191),    # Línea 6
    (0, 271, 768, 239),    # Línea 7
    (0, 330, 768, 321),    # Línea 8
    (0, 431, 768, 405),    # Línea 9: ABAJO
]

# ============================================================================
# CLASE PARA MANEJAR CARRILES CON TRACKING DINÁMICO
# ============================================================================

class CarrilesInclinadosDinamicos:
    """Carriles que se adaptan al movimiento de cámara"""
    
    def __init__(self, lineas, num_carriles=8):
        self.lineas = lineas
        self.lineas_actuales = [list(l) for l in lineas]  # Copia para tracking
        self.num_carriles = num_carriles
        
        # Optical flow para tracking de puntos
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Puntos de las líneas para optical flow
        self.puntos_lineas = self._crear_puntos_tracking()
        self.frame_anterior = None
    
    def _crear_puntos_tracking(self):
        """Crear puntos en las líneas para tracking"""
        puntos = []
        num_puntos = 8  # Puntos por línea
        
        for linea in self.lineas:
            x1, y1, x2, y2 = linea
            puntos_linea = []
            for i in range(num_puntos):
                t = i / (num_puntos - 1)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                puntos_linea.append([x, y])
            
            puntos.append(np.array(puntos_linea, dtype=np.float32).reshape(-1, 1, 2))
        
        return puntos
    
    def actualizar(self, frame):
        """Actualizar posición de líneas usando optical flow"""
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        if self.frame_anterior is None:
            self.frame_anterior = frame_gray.copy()
            return
        
        # Optical flow para cada línea
        for idx, puntos_linea in enumerate(self.puntos_lineas):
            try:
                nuevos_puntos, status, err = cv.calcOpticalFlowPyrLK(
                    self.frame_anterior,
                    frame_gray,
                    puntos_linea,
                    None,
                    **self.lk_params
                )
                
                if nuevos_puntos is not None and status is not None:
                    buenos_nuevos = nuevos_puntos[status == 1]
                    
                    if len(buenos_nuevos) >= 4:
                        # Actualizar puntos
                        self.puntos_lineas[idx] = nuevos_puntos
                        
                        # Recalcular línea con puntos tracked
                        puntos_2d = nuevos_puntos.reshape(-1, 2)
                        
                        # Ajustar línea con polinomio de grado 1
                        if len(puntos_2d) >= 2:
                            coefs = np.polyfit(puntos_2d[:, 0], puntos_2d[:, 1], 1)
                            m = coefs[0]
                            b = coefs[1]
                            
                            width = frame.shape[1]
                            height = frame.shape[0]
                            
                            x1, x2 = 0, width
                            y1 = int(m * x1 + b)
                            y2 = int(m * x2 + b)
                            
                            y1 = np.clip(y1, 0, height - 1)
                            y2 = np.clip(y2, 0, height - 1)
                            
                            self.lineas_actuales[idx] = [x1, y1, x2, y2]
            except:
                pass
        
        self.frame_anterior = frame_gray.copy()
    
    def obtener_y_en_x(self, x, indice_carril):
        """Obtiene Y en X usando línea actual"""
        if indice_carril < 0 or indice_carril >= len(self.lineas_actuales):
            return 0
        
        x1, y1, x2, y2 = self.lineas_actuales[indice_carril]
        
        if x2 - x1 == 0:
            return y1
        
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        y = int(m * x + b)
        
        return np.clip(y, 0, 1000)
    
    def obtener_rango_y_del_carril(self, x, indice_carril):
        """Obtiene rango Y del carril"""
        y_top = self.obtener_y_en_x(x, indice_carril)
        y_bottom = self.obtener_y_en_x(x, indice_carril + 1)
        return y_top, y_bottom
    
    def obtener_altura_carril(self, x, indice_carril):
        """Obtiene altura del carril"""
        y_top, y_bottom = self.obtener_rango_y_del_carril(x, indice_carril)
        return abs(y_bottom - y_top)
    
    def dibujar_carriles(self, frame):
        """Dibujar carriles actuales (con tracking)"""
        frame_viz = frame.copy()
        width = frame.shape[1]
        height = frame.shape[0]
        
        for idx, linea in enumerate(self.lineas_actuales):
            x1, y1, x2, y2 = linea
            
            y1 = np.clip(y1, 0, height - 1)
            y2 = np.clip(y2, 0, height - 1)
            x1 = np.clip(x1, 0, width - 1)
            x2 = np.clip(x2, 0, width - 1)
            
            cv.line(frame_viz, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            x_medio = width // 2
            y_medio = self.obtener_y_en_x(x_medio, idx)
            
            if idx == 0:
                texto = "ARRIBA"
            elif idx == len(self.lineas_actuales) - 1:
                texto = "ABAJO"
            else:
                texto = f"C{idx}"
            
            cv.putText(frame_viz, texto, (x_medio + 10, y_medio),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame_viz

# ============================================================================
# SELECCIÓN DE CARRILES
# ============================================================================

def seleccionar_carriles(frame, num_carriles=8, usar_hardcoded=None):
    if usar_hardcoded is None:
        print("\n" + "="*60)
        print("SELECCIÓN DE CARRILES")
        print("="*60)
        print("¿Qué deseas hacer?")
        print("1. Usar coordenadas HARDCODEADAS (rápido)")
        print("2. Seleccionar carriles MANUALMENTE (personalizado)")
        print("="*60)
        
        while True:
            opcion = input("Elige opción (1 o 2): ").strip()
            if opcion in ['1', '2']:
                usar_hardcoded = (opcion == '1')
                break
            else:
                print("Opción inválida. Intenta de nuevo.")
    
    if usar_hardcoded:
        print("\n✓ Usando coordenadas HARDCODEADAS")
        return LINEAS_CARRILES_HARDCODED
    else:
        print("\n" + "="*60)
        print("SELECCIÓN MANUAL DE CARRILES")
        print("="*60)
        print(f"Necesitas hacer {num_carriles + 1} líneas:")
        print("- Para cada carril: haz 2 clics (arriba y abajo)")
        print("\nPresiona 'r' para reiniciar")
        print("Presiona ENTER cuando termines")
        print("="*60 + "\n")
        
        frame_copy = frame.copy()
        lineas = []
        puntos_actuales = []
        
        def click_evento(event, x, y, flags, param):
            nonlocal frame_copy, lineas, puntos_actuales
            
            if event == cv.EVENT_LBUTTONDOWN:
                puntos_actuales.append((x, y))
                cv.circle(frame_copy, (x, y), 6, (0, 255, 0), -1)
                
                if len(puntos_actuales) == 2:
                    x1, y1 = puntos_actuales[0]
                    x2, y2 = puntos_actuales[1]
                    
                    if x2 != x1:
                        m = (y2 - y1) / (x2 - x1)
                        b = y1 - m * x1
                        x_izq, x_der = 0, frame.shape[1]
                        y_izq = int(m * x_izq + b)
                        y_der = int(m * x_der + b)
                    else:
                        x_izq, x_der = x1, x1
                        y_izq, y_der = 0, frame.shape[0]
                    
                    y_izq = np.clip(y_izq, 0, frame.shape[0] - 1)
                    y_der = np.clip(y_der, 0, frame.shape[0] - 1)
                    
                    cv.line(frame_copy, (x_izq, y_izq), (x_der, y_der), (0, 255, 255), 2)
                    lineas.append((x_izq, y_izq, x_der, y_der))
                    
                    num_linea = len(lineas)
                    print(f"✓ Línea {num_linea}: ({x_izq},{y_izq}) - ({x_der},{y_der})")
                    
                    puntos_actuales = []
                
                cv.putText(frame_copy, f"Líneas: {len(lineas)}/{num_carriles + 1}", 
                          (10, frame.shape[0] - 20),
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                cv.imshow('Seleccionar Carriles', frame_copy)
        
        cv.namedWindow('Seleccionar Carriles')
        cv.setMouseCallback('Seleccionar Carriles', click_evento)
        cv.imshow('Seleccionar Carriles', frame_copy)
        
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == 13:
                if len(lineas) == num_carriles + 1:
                    print(f"\n✓ Carriles confirmados!")
                    break
                else:
                    print(f"⚠️ Necesitas {num_carriles + 1} líneas")
            elif key == ord('r'):
                lineas = []
                puntos_actuales = []
                frame_copy = frame.copy()
                cv.imshow('Seleccionar Carriles', frame_copy)
                print("Selección reiniciada")
        
        cv.destroyWindow('Seleccionar Carriles')
        return lineas

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
    print("3. Presiona 'r' para reiniciar")
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
    def __init__(self, carril, cx, cy, y_top, y_bottom, carriles_inclinados):
        self.carril = carril
        self.cx = cx
        self.cy = cy
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.carriles_inclinados = carriles_inclinados
        
        altura_carril = abs(y_bottom - y_top)
        self.altura_rect = max(10, int(altura_carril * 0.4))
        
        x_norm = cx / carriles_inclinados.lineas[0][2] if carriles_inclinados.lineas[0][2] > 0 else 0.5
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
        
        x_norm = cx / self.carriles_inclinados.lineas[0][2] if self.carriles_inclinados.lineas[0][2] > 0 else 0.5
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
    def __init__(self, num_carriles, direccion='izquierda', mascara_piscina=None, 
                 carriles_inclinados=None):
        self.num_carriles = num_carriles
        self.direccion = direccion
        self.mascara_piscina = mascara_piscina
        self.carriles_inclinados = carriles_inclinados
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
    
    def obtener_parametros_carril(self, carril):
        if carril < 2:
            return {
                'area_min': AREA_MIN_NADADOR_LEJANO,
                'solidity_min': SOLIDITY_MIN_LEJANO,
            }
        else:
            return {
                'area_min': AREA_MIN_NADADOR,
                'solidity_min': SOLIDITY_MIN,
            }
    
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
            params = self.obtener_parametros_carril(i)
            area_min = params['area_min']
            solidity_min = params['solidity_min']
            
            x_ref = frame.shape[1] // 4
            y_top, y_bottom = self.carriles_inclinados.obtener_rango_y_del_carril(x_ref, i)
            
            mejor_contorno = None
            mejor_area = 0
            mejor_centroid = None
            
            for contour in contours_busqueda:
                area = cv.contourArea(contour)
                
                if area < area_min or area > AREA_MAX_NADADOR:
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
                    if solidity < solidity_min:
                        continue
                
                if area > mejor_area:
                    mejor_area = area
                    mejor_centroid = (cx, cy)
            
            if mejor_centroid is not None:
                cx, cy = mejor_centroid
                
                if i not in self.rectangulos:
                    self.rectangulos[i] = RectanguloNadador(i, cx, cy, y_top, y_bottom, 
                                                            self.carriles_inclinados)
                else:
                    self.rectangulos[i].actualizar(cx, cy, y_top, y_bottom)
                
                nadadores_por_carril[i] = (cx, cy)
        
        return nadadores_por_carril
    
    def detectar_llegadas(self, nadadores, linea_meta, frame_num):
        
        for carril in range(self.num_carriles):
            if carril not in nadadores and carril in self.rectangulos:
                self.rectangulos[carril].no_detectado()
        
        for carril, (cx, cy) in nadadores.items():
            if self.llegadas.get(carril) is not None:
                continue
            
            if carril not in self.rectangulos:
                continue
            
            rect = self.rectangulos[carril]
            
            if not rect.puede_detectar_llegada():
                continue
            
            ha_cruzado, x_frontal = rect.rect_cruza_linea(linea_meta)
            
            if rect.validar_llegada(ha_cruzado):
                self.llegadas[carril] = frame_num
                print(f"🏁 ¡LLEGADA! Carril {carril + 1} - Frame {frame_num}")
    
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
    print(f"Método: RECTÁNGULOS ADAPTATIVOS + TRACKING DINÁMICO DE CARRILES")
    print("="*60 + "\n")
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: No se pudo leer el primer frame")
        return
    
    lineas_carriles = seleccionar_carriles(first_frame, NUM_CARRILES)
    # ✅ Usar clase dinámica
    carriles_inclinados = CarrilesInclinadosDinamicos(lineas_carriles, NUM_CARRILES)
    
    linea_meta_inicial = seleccionar_linea_meta_manual(first_frame)
    linea_meta_tracker = LineaMetaTracker(first_frame, linea_meta_inicial)
    
    mascara_piscina = crear_mascara_piscina(first_frame, manual=seleccionar_roi)
    
    tracker = SwimmerTracker(NUM_CARRILES, direccion, mascara_piscina, carriles_inclinados)
    
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    frame_num = 0
    
    print("\n🏊 Procesando video... (Presiona 'q' para salir)\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # ✅ ACTUALIZAR CARRILES CON TRACKING DINÁMICO
        carriles_inclinados.actualizar(frame)
        
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
        
        # ✅ DIBUJAR CARRILES DINÁMICOS
        frame_vis = carriles_inclinados.dibujar_carriles(frame_vis)
        
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
        
        for carril, rect in tracker.rectangulos.items():
            if carril not in nadadores:
                continue
            
            cx, cy = nadadores[carril]
            x_izq, y_sup, x_der, y_inf = rect.obtener_rect_frontal()
            
            if rect.puede_detectar_llegada():
                color = (0, 255, 0)
            else:
                color = (0, 165, 255)
            
            cv.rectangle(frame_vis, (x_izq, y_sup), (x_der, y_inf), color, 2)
            cv.circle(frame_vis, (cx, cy), 5, (0, 0, 255), -1)
            cv.line(frame_vis, (x_izq, y_sup), (x_izq, y_inf), color, 3)
            
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
        seleccionar_roi=True,
        delay_ms=DELAY_MS,
        seleccionar_meta_manual=True
    )

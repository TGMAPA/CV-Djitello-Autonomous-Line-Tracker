import cv2
import numpy as np

class Detector:
    def __init__(self):
        self.annotated_frame = None

        self.roi_y = None
        self.frame_height = None
        self.frame_width = None

        # Canny limits
        self.canny_low = 60
        self.canny_high = 150

        # Blur kernel
        self.blur_kernel = (11,11)

        # ROI Y Proportiuon
        self.roi_y_proportion = 0.5

        # Dilate kernel
        self.dilate_kernel = (5,5)

    # Crear ROI (Region of interest)
    def createROI(self, frame):
        # Obtener el tamaño y de la region de interes
        self.roi_y = int(self.frame_height * self.roi_y_proportion)

        # Recortar la zona de interes del frame original
        roi = frame[self.roi_y:, :]

        # Dibujar ROI
        cv2.rectangle(
            self.annotated_frame,
            (0, self.roi_y),
            (self.frame_width, self.frame_height),
            (0, 255, 255),
            2
        )

        return roi

    def preprocess(self, frame):
        # Incrementar contraste de la frame

        # Alpha > 1 aumenta el contraste y 0 < Alpha < 1 lo decrementa
        # Beta añade brillo (-127 a 127)
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=0)

        # Transformar a escala de grises
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Agregar blur al frame para difuminar el ruido
        frame = cv2.GaussianBlur(frame, self.blur_kernel, 0)

        return frame
    
    def extractEdges(self, frame):
        # Aplicar CANNY: Detección de bordes
        edges = cv2.Canny(
            frame,
            self.canny_low,
            self.canny_high,
            L2gradient = True
        )

        # Engrosar bordes
        kernel = np.ones(self.dilate_kernel, dtype=np.uint8)

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=1
        )

        return edges

    def slidingWindows(
            self, 
            frame_edges,
            nwindows = 8, 
            margin = 60, 
            minpix = 50
        ):
        
        # Sliding windows
        nonzero = frame_edges.nonzero()

        # Coordenadas Y
        nonzero_y = np.array(nonzero[0])

        # Coordenadas X
        nonzero_x = np.array(nonzero[1])

        for i in range(len(nonzero_x)):
            px = nonzero_x[i]
            py = nonzero_y[i] + self.roi_y

            cv2.circle(
                self.annotated_frame,
                (px, py),
                1,
                (0,255,0),
                -1
            )

        # Crear histograma de puntos
        histogram = np.sum(
            frame_edges[
                frame_edges.shape[0] // 2:,
                :
            ],
            axis=0
        )

        base_x = np.argmax(histogram)

        cv2.circle(
            self.annotated_frame,
            (base_x, self.frame_height - 20),
            10,
            (0,255,255),
            -1
        )

        # Altura de cada ventana
        window_height = int(
            frame_edges.shape[0] / nwindows
        )

        # Posición inicial
        current_x = base_x

        # Lista de índices encontrados
        lane_inds = []
        
        # Iterar ventanas
        for window in range(nwindows):
            # Límites verticales
            win_y_low = (
                frame_edges.shape[0]
                - (window + 1) * window_height
            )
            win_y_high = (
                frame_edges.shape[0]
                - window * window_height
            )

            # Límites horizontales
            win_x_low = current_x - margin
            win_x_high = current_x + margin

            # Dibujar ventana
            cv2.rectangle(
                self.annotated_frame,
                (win_x_low, win_y_low + self.roi_y),
                (win_x_high, win_y_high + self.roi_y),
                (255, 0, 0),
                2
            )

            # Buscar píxeles dentro de ventana
            good_inds = (
                (nonzero_y >= win_y_low)
                & (nonzero_y < win_y_high)
                & (nonzero_x >= win_x_low)
                & (nonzero_x < win_x_high)
            ).nonzero()[0]

            # Guardar índices
            lane_inds.append(good_inds)

            # Recentrar ventana
            if len(good_inds) > minpix:
                current_x = int(
                    np.mean(
                        nonzero_x[good_inds]
                    )
                )

        # Unir todos los puntos 
        if len(lane_inds) == 0:
            return None, None

        lane_inds = np.concatenate(lane_inds)

        # Obtener píxeles finales
        lane_x = nonzero_x[lane_inds]
        lane_y = nonzero_y[lane_inds]

        # Dibujar pixeles finales
        for i in range(len(lane_x)):
            px = lane_x[i]
            py = lane_y[i] + self.roi_y

            cv2.circle(
                self.annotated_frame,
                (px, py),
                2,
                (0,0,255),
                -1
            )

        return lane_y, lane_x
    
    def polynomialFitting(self, lane_y, lane_x, frame_edges):
        try:
            # Ajustar curva
            curve = np.polyfit(
                lane_y,
                lane_x,
                2
            )

            # Generar puntos de la curva
            plot_y = np.linspace(
                0,
                frame_edges.shape[0] - 1,
                frame_edges.shape[0]
            )

            plot_x = (
                curve[0] * plot_y**2
                + curve[1] * plot_y
                + curve[2]
            )

            for i in range(len(plot_y) - 1):
                x1 = int(plot_x[i])
                y1 = int(plot_y[i]) + self.roi_y

                x2 = int(plot_x[i + 1])
                y2 = int(plot_y[i + 1]) + self.roi_y

                cv2.line(
                    self.annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    (0,255,255),
                    3
                )

            # Obtener dirección del camino 
            y_eval = frame_edges.shape[0]

            slope = (
                2 * curve[0] * y_eval
                + curve[1]
            )

            # Curvatura real
            curvature = abs(
                2 * curve[0]
            )

            if lane_x is None or len(lane_x) < 50:
                return
            
        except Exception as e:
            print("Warning: ", e)
            return 0,0,0
        
        return curve, slope, curvature

    def analyze(self, frame):
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]

        self.annotated_frame = frame.copy()

        # Preprocesar frame
        frame_preprocessed = self.preprocess(frame)

        # Recortar ROI del frame original
        frame_roi = self.createROI(frame_preprocessed)

        # Extraer Contornos
        frame_edges = self.extractEdges(frame_roi)

        # Crear sliding windows
        lane_y, lane_x = self.slidingWindows(frame_edges)

        # Ajustar curva
        curve, slope, curvature = self.polynomialFitting(lane_y, lane_x, frame_edges)

        # Obtener Centro de Cámara
        center_x = self.frame_width // 2
        cv2.line(
            self.annotated_frame,
            (center_x, 0),
            (center_x, self.frame_height),
            (255, 255, 0),
            2
        )

        return {
            "Edges": frame_edges,
            "Line Detection": self.annotated_frame
        }


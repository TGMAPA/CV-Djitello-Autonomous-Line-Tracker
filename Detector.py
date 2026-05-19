import cv2
import numpy as np


class Detector:
    def __init__(self):
        self.annotated_frame = None

        self.roi_y = None
        self.frame_height = None
        self.frame_width = None

        # Canny limits
        self.canny_low = 20
        self.canny_high = 75  

        # Blur kernel
        self.blur_kernel = (7,7)

        # ROI Y Proportiuon
        self.roi_y_proportion = 0.4

        # Dilate kernel
        self.dilate_kernel = (3,3)

        self.smoothed_curve = None

        self.previous_centers = []

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
            nwindows = 6, 
            margin = 100, 
            minpix = 50
        ):
        
        # Sliding windows
        nonzero = frame_edges.nonzero()

        # Coordenadas Y
        nonzero_y = np.array(nonzero[0])

        # Coordenadas X
        nonzero_x = np.array(nonzero[1])


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

                new_center = int(
                    np.mean(
                        nonzero_x[good_inds]
                    )
                )

                # MOMENTUM / PREDICCIÓN
                self.previous_centers.append(new_center)

                if len(self.previous_centers) > 5:
                    self.previous_centers.pop(0)

                # if len(self.previous_centers) > 2:

                #     dx = (
                #         self.previous_centers[-1]
                #         - self.previous_centers[-2]
                #     )

                #     # Limitar cambios bruscos
                #     dx = np.clip(dx, -50, 50)

                #     current_x = new_center + dx
                #     current_x = new_center + dx

                # else:
                #     current_x = new_center
                
                current_x = new_center

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
  
    def drawError(self, lateral_error, plot_x, plot_y, camera_center):
        # VISUALIZAR ERROR LATERAL

        # Punto inferior de la trayectoria
        target_x = int(plot_x[-1])
        target_y = int(plot_y[-1] + self.roi_y)

        # Centro de cámara
        camera_x = int(camera_center)
        camera_y = target_y

        # Dibujar centro cámara
        cv2.circle(
            self.annotated_frame,
            (camera_x, camera_y),
            8,
            (255, 255, 0),
            -1
        )

        # Dibujar punto objetivo
        cv2.circle(
            self.annotated_frame,
            (target_x, target_y),
            8,
            (0, 0, 255),
            -1
        )

        # Dibujar línea de error
        cv2.line(
            self.annotated_frame,
            (camera_x, camera_y),
            (target_x, target_y),
            (0, 0, 255),
            3
        )

        # TEXTO ERROR
        cv2.putText(
            self.annotated_frame,
            f"Error: {lateral_error:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50,50,255),
            2
        )

    def polynomialFitting(
            self,
            lane_y,
            lane_x
        ):

        if lane_x is None or len(lane_x) < 80:
            return None, None, None, None

        try:

            # ELIMINAR OUTLIERS SIMPLES
            median_x = np.median(lane_x)

            mask = np.abs(
                lane_x - median_x
            ) < 250

            lane_x = lane_x[mask]
            lane_y = lane_y[mask]

            if len(lane_x) < 50:
                return None, None, None, None

            # POLYFIT SIMPLE
            curve = np.polyfit(
                lane_y,
                lane_x,
                2
            )

            # SUAVIZADO TEMPORAL
            alpha = 0.2

            if self.smoothed_curve is None:
                self.smoothed_curve = curve

            else:
                self.smoothed_curve = (
                    alpha * curve
                    + (1 - alpha)
                    * self.smoothed_curve
                )

            curve = self.smoothed_curve

            # GENERAR CURVA
            plot_y = np.linspace(
                lane_y.min(),
                lane_y.max(),
                80
            )

            plot_x = (
                curve[0] * plot_y**2
                + curve[1] * plot_y
                + curve[2]
            )

            # DIBUJAR
            points = np.array([
                np.transpose(
                    np.vstack([
                        plot_x,
                        plot_y + self.roi_y
                    ])
                )
            ]).astype(np.int32)

            cv2.polylines(
                self.annotated_frame,
                points,
                False,
                (0,255,255),
                3
            )

            # DIRECCIÓN
            y_eval = plot_y[-1]

            slope = (
                2 * curve[0] * y_eval
                + curve[1]
            )

            curvature = abs(
                2 * curve[0]
            )

            # Calcular error lateral
            line_x = plot_x[-1]

            camera_center = self.frame_width // 2

            lateral_error = (
                line_x - camera_center
            )

            # Dibujar error lateral con respeceto al centro de la camara
            self.drawError( lateral_error, plot_x, plot_y, camera_center)
            
            return (
                curve,
                slope,
                curvature,
                lateral_error
            )

        except Exception as e:
            print("Warning:", e)
            return None, None, None, None

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

        # Ajustar curva y obtener error lateral
        curve, slope, curvature, lateral_error = self.polynomialFitting(lane_y, lane_x)
        
        # Obtener Centro de Cámara
        center_x = self.frame_width // 2
        cv2.line(
            self.annotated_frame,
            (center_x, 0),
            (center_x, self.frame_height),
            (255, 255, 0),
            2
        )

        return (
            {
                "Preprocessed Frame": frame_preprocessed,
                "Edges": frame_edges,
                "Line Detection": self.annotated_frame
            },
            lateral_error,
            slope
        )


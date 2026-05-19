from Detector import Detector
from Drone import Drone
from PID import PID

import cv2, math
import numpy as np

# === Constantes de Configuración ===
FRAME_SIZE = (640, 360)

# Porcentaje de la altura original a analizar
#ROI_Y_PROPORTION = 0.5

# Parametros de color de frame 
FRAME_COLOR_PITCH = 1
FRAME_CONTRAST_ALPHA = 1


def increasePitch(img, pitch = 1.5):
    # Convertr a hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Incrementar saturación
    h, s, v = cv2.split(hsv)
    s = s * pitch
    s = np.clip(s, 0, 255).astype(np.uint8) # Clipear entre 0-255

    # 4. Merge the channels back and convert to RGB
    hsv_enhanced = cv2.merge([h, s, v])
    enhanced_img = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return enhanced_img

def increaseContrast(img, alpha = 1.5):
    # Parameters: src1, alpha (contrast), src2, beta (brightness), gamma
    enhanced_img = cv2.addWeighted(img, alpha, img, 0, 30)

    return enhanced_img

def create_debug_dashboard(frames_dict):

    # ======================================================
    # CONFIGURACIÓN GENERAL
    # ======================================================

    dashboard_width = 1600
    dashboard_height = 900

    padding = 20

    bg_color = (20, 20, 20)

    title_color = (0, 255, 255)

    # ======================================================
    # CREAR CANVAS
    # ======================================================

    dashboard = np.zeros(
        (dashboard_height, dashboard_width, 3),
        dtype=np.uint8
    )

    dashboard[:] = bg_color

    # ======================================================
    # GRID DINÁMICO
    # ======================================================

    n_frames = len(frames_dict)

    cols = math.ceil(math.sqrt(n_frames))
    rows = math.ceil(n_frames / cols)

    # ======================================================
    # TAMAÑO DE CELDAS
    # ======================================================

    cell_w = dashboard_width // cols
    cell_h = dashboard_height // rows

    # ======================================================
    # ITERAR FRAMES
    # ======================================================

    for idx, (name, frame) in enumerate(
        frames_dict.items()
    ):

        # --------------------------------------------------
        # Convertir grayscale -> BGR
        # --------------------------------------------------
        if len(frame.shape) == 2:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR
            )

        # --------------------------------------------------
        # POSICIÓN EN GRID
        # --------------------------------------------------
        row = idx // cols
        col = idx % cols

        x = col * cell_w
        y = row * cell_h

        # --------------------------------------------------
        # ÁREA DISPONIBLE
        # --------------------------------------------------
        available_w = cell_w - 2 * padding
        available_h = cell_h - 60

        # --------------------------------------------------
        # RESIZE MANTENIENDO ASPECT RATIO
        # --------------------------------------------------
        h, w = frame.shape[:2]

        scale = min(
            available_w / w,
            available_h / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            frame,
            (new_w, new_h)
        )

        # --------------------------------------------------
        # CENTRAR IMAGEN
        # --------------------------------------------------
        offset_x = x + (cell_w - new_w) // 2
        offset_y = y + 40

        # --------------------------------------------------
        # PEGAR IMAGEN
        # --------------------------------------------------
        dashboard[
            offset_y:offset_y + new_h,
            offset_x:offset_x + new_w
        ] = resized

        # --------------------------------------------------
        # TÍTULO
        # --------------------------------------------------
        cv2.putText(
            dashboard,
            name,
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            title_color,
            2
        )

        # --------------------------------------------------
        # BORDE VISUAL
        # --------------------------------------------------
        cv2.rectangle(
            dashboard,
            (x + 5, y + 5),
            (x + cell_w - 5, y + cell_h - 5),
            (80, 80, 80),
            2
        )

    return dashboard

def main():

    video_path = "./resources/video/drone_zigzag_greenBG.mp4"
    # video_path = "./resources/video/video_white_greenBackground.mp4"
    # video_path = "./resources/video/video_orange_zigzag_notguided.MOV"
    
    # Video
    cap = cv2.VideoCapture(video_path)

    # Verificar apertura
    if not cap.isOpened():
        print("No se pudo abrir video/cámara")
        exit()

    # Iniciar Detector
    detector = Detector()

    # Iniciar Dron
    drone = None
    #drone = Drone(streamon=True)

    # PID controlador
    yaw_pid = PID(
        kp=0.12,
        ki=0.0,
        kd=0.04
    )

    # Error filtrado
    filtered_error = 0

    # Velocidad de avance
    forward_speed = 0

    # Loop Principal
    while True:
        
        # Extraer fuente de video
        if drone is None:
            # Leer Frame de video
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_180)
        else:
            # Leer frame de drone
            frame = drone.getFrame()
            
        
        # --- Test pitch and contrast
        #frame = increasePitch(frame, pitch=FRAME_COLOR_PITCH)
        #frame = increaseContrast(frame, alpha=FRAME_CONTRAST_ALPHA)

        # Resize
        frame = cv2.resize(frame, FRAME_SIZE)        

        # Analyze image
        frames, error = detector.analyze(frame)

        # Atualizar comando con el error detectado
        if error is not None:
            forward_speed = 20

            # Filtrar error
            alpha = 0.2

            filtered_error = (
                alpha * error
                + (1 - alpha) * filtered_error
            )

            yaw_command = yaw_pid.update(
                filtered_error
            )

            # Aplicar un clip al comando para evitar movimientos agresivos
            yaw_command = int(
                np.clip(
                    yaw_command,
                    -30,
                    30
                )
            )

        else:
            forward_speed = 0
            yaw_command = 0

        # Enviar comando de control al dron
        if drone is not None:
            drone.send_control(
                yaw_command,
                0,      # left-right
                forward_speed,     # forward
                0      # up-down     
            )

        # Mostrar ventanas
        dashboard = create_debug_dashboard(
            frames
        )

        cv2.imshow(
            "Debug Dashboard",
            dashboard
        )


        # Tecla especial
        if cv2.waitKey(15) == 27:
            break
    

    # Liberar recursos
    if drone is None:
        cap.release()
    else:
        drone.kill()

        del drone

    cv2.destroyAllWindows()


# Main
if __name__ == "__main__":
    main()
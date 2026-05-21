from Detector import Detector
from Drone import Drone

import cv2, math
import numpy as np
import time

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
    #drone = None
    drone = Drone(streamon=True)

    # Despegar el dron
    print("[Drone] taking off...")
    drone.takeoff()

    print("[Drone] IMU STAB...")
    time.sleep(1)  # dejar estabilizar IMU

    move_up = False
    if move_up:
        print("[Drone] Moving up...")
        # subir usando RC control
        drone.tello.send_rc_control(0,0,65,0)

        time.sleep(2)

        print("[Drone] Stopping 'move_up'...")
        # detener movimiento vertical
        drone.send_control(0, 0, 0, 0)

    # Error filtrado
    filtered_error = 0

    # Velocidad de avance
    forward_speed = 0

    previous_yaw = 0

    filtered_slope = 0

    print("[Drone] Starting analysis...")

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
        frames, error, slope = detector.analyze(frame)

        # Atualizar comando con el error detectado
        if error is not None:
            # Filtrar error
            alpha = 0.4

            filtered_error = (
                alpha * error
                + (1 - alpha) * filtered_error
            )

            if abs(filtered_error) < 2:
                filtered_error = 0
            

            yaw_position = filtered_error * 0.28

            angle_error = np.degrees(np.arctan(slope))

            angle_error = np.clip(
                angle_error,
                -25,
                25
            )

            yaw_angle = angle_error * 0.5
            # SOlo rotar
            if abs(angle_error) > 15:
                forward_speed = 0

            # Error combinado
            combined_error = (
                abs(filtered_error)
                + abs(angle_error) * 1.5
            )

            # COntrolar velocidad forward 
            if combined_error < 10:
                forward_speed = 10

            elif combined_error < 20:
                forward_speed = 6

            elif combined_error < 30:
                forward_speed = 3

            else:
                forward_speed = 0

            
            # YAW
            yaw_position = filtered_error * 0.12
            yaw_angle = angle_error * 0.35

            yaw_command = yaw_position + yaw_angle

            # Aplicar un clip al comando para evitar movimientos agresivos
            yaw_command = int(
                np.clip(
                    yaw_command,
                    -25,
                    25
                )
            )
            if abs(yaw_command) < 4:
                yaw_command = 0

            # SUAVIZADO DE YAW
            yaw_command = (
                0.15 * previous_yaw
                + 0.85 * yaw_command
            )
            previous_yaw = yaw_command

            yaw_command = int(yaw_command)


        else:
            forward_speed = 0
            yaw_command = 0
            left_right_command = 0

        # Enviar comando de control al dron
        if drone is not None:
            drone.send_control(
                left_right_command,      # left-right
                forward_speed,     # forward
                0,      # up-down,
                yaw_command, #yaw   
            )

        drone.display_parameters(frames["Line Detection"])

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
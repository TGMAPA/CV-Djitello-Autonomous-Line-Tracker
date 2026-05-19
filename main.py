from Detector import Detector
from Drone import Drone

import cv2

# === Constantes de Configuración ===
FRAME_SIZE = (1200, 1000)

# Porcentaje de la altura original a analizar
ROI_Y_PROPORTION = 0.5


def main():

    video_path = "./resources/video/video_white_greenBackground.mp4"
    
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
    drone = Drone(streamon=True)

    # Loop Principal
    while True:
        
        # Extraer fuente de video
        if drone is None:
            # Leer Frame de video
            ret, frame = cap.read()

            if not ret:
                break
        else:
            # Leer frame de drone
            frame = drone.getFrame()

        # Resize
        frame = cv2.resize(frame, FRAME_SIZE)        

        # Analyze image
        frames = detector.analyze(frame)

        # Mostrar Ventanas con Resultados
        for window, img in frames.items():
            cv2.imshow(window, img)

        # Tecla especial
        if cv2.waitKey(10) == 27:
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
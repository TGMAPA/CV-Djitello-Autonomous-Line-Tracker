from Drone import Drone
import cv2

drone = Drone(streamon=True)

# Loop Principal
while True:

    # Leer frame de drone
    frame = drone.getFrame()
          

    cv2.imshow(
        "Dron Calibration",
        frame
    )

    # Tecla especial
    if cv2.waitKey(15) == 27:
        break


# Liberar recursos
drone.kill()
del drone
cv2.destroyAllWindows()

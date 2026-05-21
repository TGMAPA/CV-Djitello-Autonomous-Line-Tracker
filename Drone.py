from djitellopy import Tello
import cv2

class Drone:
    def __init__(self, streamon = False):
        try:
            self.tello = Tello()

            self.tello.connect()

            print("Drone Battery: ", self.tello.get_battery())

            if streamon:
                self.tello.streamon()

        except Exception as e:
            print("Hubo un error al hacer la conexión con el dron: ", e)

        self.forward_speed = None
        self.yaw = None

    def getFrame(self):
        return self.tello.get_frame_read().frame
    
    def kill(self):
        self.tello.streamoff()

    def takeoff(self):
        self.tello.takeoff()

    def send_control(
            self, 
            left_right, forward, up_down, yaw_command
        ):
        self.tello.send_rc_control(left_right, forward, up_down, yaw_command)

        # Actualizar parametros
        self.forward_speed = forward
        self.yaw = yaw_command

    def display_parameters(self, frame):
        # TEXTO ERROR

        cv2.putText(
            frame,
            f"FwdSpeed: {self.forward_speed:.3f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50,255,50),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {self.yaw:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (50,255,50),
            2
        )

from djitellopy import Tello

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

    def getFrame(self):
        return self.tello.get_frame_read().frame
    
    def kill(self):
        self.tello.streamoff()
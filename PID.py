import time
import numpy as np

class PID:

    def __init__(
            self,
            kp,
            ki,
            kd
        ):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0
        self.integral = 0

        self.previous_time = time.time()

    def update(self, error):

        current_time = time.time()

        dt = current_time - self.previous_time

        self.previous_time = current_time

        if dt <= 0:
            dt = 1e-6

        # ==================================
        # COMPONENTES PID
        # ==================================

        proportional = error

        self.integral += error * dt

        derivative = (
            error - self.previous_error
        ) / dt

        derivative = np.clip(
            derivative,
            -300,
            300
        )

        self.previous_error = error

        # ==================================
        # OUTPUT
        # ==================================

        output = (
            self.kp * proportional
            + self.ki * self.integral
            + self.kd * derivative
        )

        return output
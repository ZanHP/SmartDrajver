import math

import arcade

from smartdriver.constants import *


class Player(arcade.Sprite):
    """ Player class """

    def __init__(self, image, scale):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)

        # Create a variable to hold our speed. 'angle' is created by the parent
        #self.center_x;
        #self.center_y;
        #self.angle;

        self.speed_angle = 0
        self.speed = 0

        self.accelerating = False
        self.braking = False

    def update(self):
        if self.accelerating:
            speed_temp = (self.speed + ACCELERATION_UNIT)
            self.speed = min(speed_temp, MAX_SPEED)
        elif self.braking:
            speed_temp = (self.speed - ACCELERATION_UNIT * BRAKING_POWER) * FRICTION
            self.speed = max(speed_temp, 0)
        else:
            speed_temp = self.speed * FRICTION
            self.speed = speed_temp if speed_temp > TOL else 0

        #print('center_x: {}, center_y: {}, v: {}, a: {}, sa: {}'.format(*list(map(lambda x : round(x,2),[self.center_x, self.center_y, self.speed, self.angle, self.speed_angle]))))
        # Convert angle in degrees to radians.
        angle_rad = math.radians(self.angle)
        change_angle_rad = math.radians(self.change_angle)

        # Rotate the ship
        self.angle += self.change_angle        

        # Use math to find our change based on our speed and angle
        speed_angle_temp = (1 - (1-GRIP) * (self.speed / MAX_SPEED)**0.3) * (self.angle - self.speed_angle)
        if abs(speed_angle_temp) > TOL_ANGLE:
            self.speed_angle += speed_angle_temp    
        #elif abs(speed_angle_temp) > TOL_ANGLE and self.change_angle == 0:
            #print(0)
            #self.speed_angle = self.angle
        #else:

        #self.speed_angle = self.speed_angle + speed_angle_temp if abs(speed_angle_temp) > TOL_ANGLE and self.change_angle else self.angle
        speed_angle_rad = math.radians(self.speed_angle)

        #speed_temp = (self.speed[0] - norm(self.speed) * math.sin(angle_rad + 5*change_angle_rad))

        x_temp = self.center_x + self.speed * (-math.sin(speed_angle_rad))
        y_temp = self.center_y + self.speed * math.cos(speed_angle_rad)
        if x_temp < 0:
            self.center_x = 0
        elif x_temp > SCREEN_WIDTH:
            self.center_x = SCREEN_WIDTH
        else:
            self.center_x = x_temp
        if y_temp < 0:
            self.center_y = 0
        elif y_temp > SCREEN_HEIGHT:
            self.center_y = SCREEN_HEIGHT
        else:
            self.center_y = y_temp

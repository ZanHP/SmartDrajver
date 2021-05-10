import math

import arcade

from smartdriver.constants import *
from smartdriver.track import Track
import random


class Player(arcade.Sprite):
    """ Player class """
    
    def __init__(self, image, scale, start, track, smart=False, show=True, verbose=False):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)

        self.track = track
        self.show = show
        self.smart = smart

        self.center_x = start[0]
        self.center_y = start[1]
        self.center_x_noShow = self.center_x
        self.center_y_noShow = self.center_y
        self.verbose = verbose

        self.speed_angle = 0
        self.speed = 0

        self.accelerating = False
        self.braking = False

        # indeks checkpointov v track
        self.next_checkpoint = 1

    def __repr__(self):
        return "smart: {}, up: {}, down: {}, right: {}, left: {}".format(self.smart, self.accelerating, self.braking, self.change_angle, self.change_angle)

    def __str__(self):
        #property_names=[p for p in dir(Player) if isinstance(getattr(Player,p),property)]
        #return str(property_names)
        return "smart: {}, up: {}, down: {}, right: {}, left: {}".format(self.smart, self.accelerating, self.braking, self.change_angle, self.change_angle)    

    def print_self(self):
        if self.show:            
            print('x: {}, y: {}, v: {}, a: {}, sa: {}'.format(*list(map(lambda x : round(x,2),[self.center_x, self.center_y, self.speed, self.angle, self.speed_angle]))))
        else:
            print('x: {}, y: {}, v: {}, a: {}, sa: {}'.format(*list(map(lambda x : round(x,2),[self.center_x_noShow, self.center_y_noShow, self.speed, self.angle, self.speed_angle]))))

    def update(self):
        if self.verbose:
            #self.print_self()
            print(self)

        if self.accelerating:
            speed_temp = (self.speed + ACCELERATION_UNIT)
            self.speed = min(speed_temp, MAX_SPEED)
        elif self.braking:
            speed_temp = (self.speed - ACCELERATION_UNIT * BRAKING_POWER) * FRICTION
            self.speed = max(speed_temp, 0)
        else:
            speed_temp = self.speed * FRICTION
            self.speed = speed_temp if speed_temp > TOL else 0

        # Rotate the ship
        self.angle += self.change_angle

        # Use math to find our change based on our speed and angle
        speed_angle_temp = (1 - (1-GRIP) * (self.speed / MAX_SPEED)**0.3) * (self.angle - self.speed_angle)
        if abs(speed_angle_temp) > TOL_ANGLE:
            self.speed_angle += speed_angle_temp   

        speed_angle_rad = math.radians(self.speed_angle)

        x_temp = self.center_x_noShow + self.speed * (-math.sin(speed_angle_rad))
        y_temp = self.center_y_noShow + self.speed * math.cos(speed_angle_rad)
        if x_temp < 0:
            self.center_x_noShow = 0
        elif x_temp > SCREEN_WIDTH:
            self.center_x_noShow = SCREEN_WIDTH
        else:
            self.center_x_noShow = x_temp
        if y_temp < 0:
            self.center_y_noShow = 0
        elif y_temp > SCREEN_HEIGHT:
            self.center_y_noShow = SCREEN_HEIGHT
        else:
            self.center_y_noShow = y_temp

        if self.show:
            self.center_x = self.center_x_noShow
            self.center_y = self.center_y_noShow

        #if random.random() < 0.01:
        #    self.checkpoint_reached()
        if self.distance_to_next_checkpoint() < TOL_CHECKPOINT:
            return self.checkpoint_reached()
        else:
            return True
        #if ((self.center_x_noShow - self.next_checkpoint[0]) ** 2 + (self.center_y_noShow - self.next_checkpoint[1]) ** 2) ** 0.5 < TOL_CHECKPOINT:
        #    print(((self.center_x_noShow - self.next_checkpoint[0]) ** 2 + (self.center_y_noShow - self.next_checkpoint[1]) ** 2) ** 0.5)
        #    self.checkpoint_reached()

        #print(self.angle_of_checkpoint())
        #print('angle:',self.angle)

    def distance_to_next_checkpoint(self):
        x, y = self.center_x_noShow, self.center_y_noShow
        nc = self.track.checkpoints[self.next_checkpoint]
        res = ((x-nc[0])**2 + (y-nc[1])**2)**0.5
        #print(res)
        return res

    def checkpoint_reached(self):
        self.next_checkpoint += 1
        if self.next_checkpoint < len(self.track.checkpoints):
            self.track.next_checkpoint = self.next_checkpoint
            print(True)
            #print(self.next_checkpoint, "True", len(self.track.checkpoints))
            return True
        else:
            #print(self.next_checkpoint, "False", len(self.track.checkpoints))
            print(False)
            return False

    def angle_of_checkpoint(self):
        nc = self.track.checkpoints[self.next_checkpoint]
        nc_direction = [nc[0]-self.center_x_noShow, nc[1]-self.center_y_noShow]
        temp_angle = angle_of_vectors([0,1],nc_direction)
        if self.center_x_noShow < nc[0]:
            return -temp_angle
        else:
            return temp_angle

    def next_move_and_update(self, rand=False):
        if rand:
            if not self.accelerating:
                if random.random() < 0.5:
                    self.on_press_key_up()
            elif random.random() < 0.1:
                self.on_release_key_up()
            elif random.random() < 0.5:
                if random.random() < 0.5:
                    self.on_press_key_right()
                else:
                    self.on_press_key_left()
        else:
            d = self.distance_to_next_checkpoint()
            if d > 2*TOL_CHECKPOINT:
                self.on_press_key_down()
                self.on_press_key_up()
            elif d > TOL_CHECKPOINT :
                self.on_release_key_up()
                self.on_press_key_down()
            angle_dif = (self.angle_of_checkpoint() - self.angle) % 360
            #print(angle_dif)
            if abs(angle_dif) > ANGLE_SPEED:
                if angle_dif < 180:
                    self.on_press_key_left()
                else:
                    self.on_press_key_right()
            else:
                self.on_release_key_left()
                self.on_release_key_right()
        return self.update()

    def on_press_key_up(self):
        self.accelerating = True

    def on_press_key_down(self):
        self.braking = True

    def on_press_key_left(self):
        self.change_angle = ANGLE_SPEED

    def on_press_key_right(self):
        self.change_angle = -ANGLE_SPEED

    def on_release_key_up(self):
        self.accelerating = False

    def on_release_key_down(self):
        self.braking = False

    def on_release_key_left(self):
        self.change_angle = 0

    def on_release_key_right(self):
        self.change_angle = 0

def angle_of_vectors(A,B):
    
    dot = A[0]*B[0] + A[1]*B[1]
    norms = math.sqrt(A[0]**2 + A[1]**2)*math.sqrt(B[0]**2 + B[1]**2)
    angle = dot/norms
    angleInDegree = math.degrees(math.acos(angle))
    return angleInDegree
    #print("θ =",round(angleInDegree),"°")

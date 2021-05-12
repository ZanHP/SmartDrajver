import math
import numpy as np

import arcade

from smartdriver.constants import *

from collections import defaultdict
import random

class Player(arcade.Sprite):
    """ Player class """
    
    def __init__(self, image, scale, start, track, smart=False, best_run=np.Inf, state_actions_dict=defaultdict(), verbose=False):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)

        self.track = track
        #self.show = show
        self.smart = smart

        self.center_x = start[0]
        self.center_y = start[1]
        #self.center_x_noShow = self.center_x
        #self.center_y_noShow = self.center_y
        self.verbose = verbose

        self.speed_angle = 0
        self.speed = 0

        self.has_done_random = False
        self.recorded_actions = []
        self.actions = []
        self.action_index = 0
        self.remember_random = 0
        self.remember_random_action = ""

        self.best_run = best_run
        self.state_actions_dict = state_actions_dict

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
        print('x: {}, y: {}, v: {}, a: {}, sa: {}'.format(*list(map(lambda x : round(x,2),[self.center_x, self.center_y, self.speed, self.angle, self.speed_angle]))))

    def update(self):
        # Naredi en časovni korak. Vrne True, če nismo na koncu proge, False, če smo in None, če je bilo narejenih že preveč korakov.
        if self.verbose:
            #self.print_self()
            print(self)

        if len(self.actions) > self.best_run:
            print(self.best_run, len(self.recorded_actions))
            return None

        self.actions.append(self.get_action())
        #self.state_actions_dict[self.get_state()]
        
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

        x_temp = self.center_x + self.speed * (-math.sin(speed_angle_rad))
        y_temp = self.center_y + self.speed * math.cos(speed_angle_rad)
        if x_temp < 0:
            self.change_x = 0 - self.center_x
        elif x_temp > SCREEN_WIDTH:
            self.change_x = SCREEN_WIDTH - self.center_x
        else:
            self.change_x = x_temp - self.center_x
        if y_temp < 0:
            self.change_y = 0 - self.center_y
        elif y_temp > SCREEN_HEIGHT:
            self.change_y = SCREEN_HEIGHT - self.center_y
        else:
            self.change_y = y_temp - self.center_y
        
        self.center_x += self.change_x
        self.center_y += self.change_y

        if self.verbose:
            self.print_self()

        #if self.show:
        #    self.center_x = self.center_x_noShow
        #    self.center_y = self.center_y_noShow

        #if random.random() < 0.01:
        #    self.checkpoint_reached()
        if self.distance_to_next_checkpoint() < TOL_CHECKPOINT:
            res =  self.checkpoint_reached()
            if not res:
                self.has_recorded_run = True
            return res
        else:
            return True
        #if ((self.center_x_noShow - self.next_checkpoint[0]) ** 2 + (self.center_y_noShow - self.next_checkpoint[1]) ** 2) ** 0.5 < TOL_CHECKPOINT:
        #    print(((self.center_x_noShow - self.next_checkpoint[0]) ** 2 + (self.center_y_noShow - self.next_checkpoint[1]) ** 2) ** 0.5)
        #    self.checkpoint_reached()

        #print(self.angle_of_checkpoint())
        #print('angle:',self.angle)
        

    def distance_to_next_checkpoint(self):
        x, y = self.center_x, self.center_y
        nc = self.track.checkpoints[self.next_checkpoint]
        res = ((x-nc[0])**2 + (y-nc[1])**2)**0.5
        return res

    def checkpoint_reached(self):
        # True, če še nismo na koncu, False, če smo končali.
        self.next_checkpoint += 1
        self.track.next_checkpoint =  self.next_checkpoint
        if self.next_checkpoint < len(self.track.checkpoints):
            return True
        else:
            return False

    def angle_of_checkpoint(self, plus_one=0):
        if len(self.track.checkpoints) > self.next_checkpoint+plus_one:
            nc = self.track.checkpoints[self.next_checkpoint+plus_one]
        else:
            nc = self.track.checkpoints[self.next_checkpoint]
        nc_direction = [nc[0]-self.center_x, nc[1]-self.center_y]
        temp_angle = angle_of_vectors([0,1],nc_direction)
        if self.center_x < nc[0]:
            return -temp_angle
        else:
            return temp_angle

    def next_move_and_update(self):
        rand_val = random.random()
        d = self.distance_to_next_checkpoint()
        if self.recorded_actions and not self.has_done_random and rand_val >= ALPHA:
            # gre po stari najboljši poti
            self.do_action(self.recorded_actions[self.action_index])
            self.action_index += 1
        
        if self.remember_random != 0:
            self.remember_random -= 1 
            self.do_action(self.remember_random_action)

        elif rand_val < ALPHA: #or self.remember_random:
            choices = ["D","A","S",""]
            choice = random.choice(choices) #if not self.remember_random else self.remember_random_action
            #if not self.has_done_random: 
            #    print(self.action_index)
            self.has_done_random = True
            self.do_action(choice)

            if d < 7 * TOL_CHECKPOINT:
                self.remember_random = CONSECUTIVE_STEPS
            else:
                self.remember_random = CONSECUTIVE_STEPS // 2
            self.remember_random_action = choice

        else:            
            if not self.accelerating:
                self.on_press_key_up()

            if d > 2*TOL_CHECKPOINT:
                self.on_release_key_down()
                self.on_press_key_up()

                # če ne dela naključne poteze, upošteva hevristiko
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

            elif d > TOL_CHECKPOINT:
                if random.random() > ALPHA_BRAKE:
                    self.on_release_key_up()
                    self.on_press_key_down()

                # če ne dela naključne poteze, upošteva hevristiko
                angle_dif = (self.angle_of_checkpoint(plus_one=1) - self.angle) % 360
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

    def get_action(self):
        up_down = ""
        if self.accelerating:
            up_down =  "W"
        elif self.braking:
            up_down = "S"
        
        left_right = ""
        if self.change_angle == -ANGLE_SPEED:
            left_right = "D"
        elif self.change_angle == ANGLE_SPEED:
            left_right = "A"
        
        return left_right
    def do_action(self, action):
        if action == "D":
            self.on_press_key_right()
        elif action == "A":
            self.on_press_key_left()
        elif action == "S":
            self.on_press_key_down()
        else:
            self.on_release_key_down()
            self.on_release_key_left()
            self.on_release_key_right()

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

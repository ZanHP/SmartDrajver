from constants import *
import arcade
import math

class Player(arcade.Sprite):
    """ Player class """
    
    def __init__(self, image, scale, smart=False, show=True):
        """ Set up the player """

        # Call the parent init
        super().__init__(image, scale)
        self.show = show
        self.smart = smart

        self.center_x = SCREEN_WIDTH / 2
        self.center_y = SCREEN_HEIGHT / 2
        self.center_x_noShow = self.center_x
        self.center_y_noShow = self.center_y

        self.speed_angle = 0
        self.speed = 0

        self.accelerating = False
        self.braking = False

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
        self.print_self()

        if self.accelerating:
            speed_temp = (self.speed + ACCELERATION_UNIT)
            self.speed = min(speed_temp, MAX_SPEED)
        elif self.braking:
            speed_temp = (self.speed - ACCELERATION_UNIT * BRAKING_POWER) * FRICTION
            self.speed = max(speed_temp, 0)
        else:
            speed_temp = self.speed * FRICTION
            self.speed = speed_temp if speed_temp > TOL else 0

        # Convert angle in degrees to radians.
        #angle_rad = math.radians(self.angle)
        #change_angle_rad = math.radians(self.change_angle)

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

        x_temp = (self.center_x if self.show else self.center_x_noShow) + self.speed * (-math.sin(speed_angle_rad))
        y_temp = (self.center_y if self.show else self.center_y_noShow) + self.speed * math.cos(speed_angle_rad)
        if x_temp < 0:
            if self.show: 
                self.center_x = 0
            else:
                self.center_x_noShow = 0
        elif x_temp > SCREEN_WIDTH:
            if self.show: 
                self.center_x = SCREEN_WIDTH
            else:
                self.center_x_noShow = SCREEN_WIDTH
        else:
            if self.show:
                self.center_x = x_temp
            else:
                self.center_x_noShow = x_temp
        if y_temp < 0:
            if self.show:
                self.center_y = 0
            else:
                self.center_y_noShow = 0
        elif y_temp > SCREEN_HEIGHT:
            if self.show:
                self.center_y = SCREEN_HEIGHT
            else:
                self.center_y_noShow = SCREEN_HEIGHT
        else:
            if self.show:
                self.center_y = y_temp
            else:
                self.center_y_noShow = y_temp

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

    def next_move(self):
        self.on_press_key_up()
        self.update()
    

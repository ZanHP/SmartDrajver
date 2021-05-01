import pygame
pygame.init() #initializes the Pygame
from pygame.locals import* #import all modules from Pygame
import random

width = 800
height = 1000

tol = 0.01

screen = pygame.display.set_mode((width,height))


#changing title of the game window
pygame.display.set_caption('Racing Beast')
#changing the logo
logo = pygame.image.load('pygametest/cargame/logo.jpeg')
pygame.display.set_icon(logo)


#defining our gameloop function

def gameloop():

    #setting background image
    #bg = pygame.image.load('pygametest/cargame/bg.png')

    friction_coef = 0.99

    # setting our player
    maincar = pygame.image.load('pygametest/cargame/car.png')
    x_location = width//2
    y_location = 3*height//4
    x_speed = 0
    y_speed = 0
    x_acceleration = 0
    y_acceleration = 0

    acceleration_unit = 0.04

    #x_location_change = 0
    #y_location_change = 0

    #other cars
    '''
    car1 = pygame.image.load('pygametest/cargame/car1.jpeg')
    car1X = random.randint(178,490)
    car1Y = 100

    car2 = pygame.image.load('pygametest/cargame/car2.png')
    car2X = random.randint(178,490)
    car2Y = 100

    car3 = pygame.image.load('pygametest/cargame/car3.png')
    car3X = random.randint(178,490)
    car3Y = 100
    '''
   
    run = True
    pressed_RIGHT = False
    pressed_LEFT = False
    pressed_UP = False
    pressed_DOWN = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

                #checking if any key has been pressed
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_RIGHT:
                    pressed_RIGHT = True
                    #x_speed = x_speed*friction_coef + 1
            
                if event.key == pygame.K_LEFT:
                    pressed_LEFT = True
                    #x_speed = x_speed*friction_coef - 1
                
                if event.key == pygame.K_UP:
                    pressed_UP = True
                    #y_speed = y_speed*friction_coef - 1
                    
                if event.key == pygame.K_DOWN:
                    pressed_DOWN = True
                    #y_speed = y_speed*friction_coef + 1
                
            if event.type == pygame.KEYUP: 
                if event.key == pygame.K_RIGHT:
                    pressed_RIGHT = False
                    #x_location_change = 0
            
                if event.key == pygame.K_LEFT:
                    pressed_LEFT = False
                    #x_location_change = 0
                
                if event.key == pygame.K_UP:
                    pressed_UP = False
                    #y_location_change = 0
                    
                if event.key == pygame.K_DOWN:
                    pressed_DOWN = False
                    #y_location_change = 0     

        x_acceleration = 0
        y_acceleration = 0

        if pressed_RIGHT:
            x_acceleration = acceleration_unit
        if pressed_LEFT:
            x_acceleration = -acceleration_unit
        if pressed_UP:
            y_acceleration = -acceleration_unit
        if pressed_DOWN:
            y_acceleration = acceleration_unit

        x_temp = (x_speed + x_acceleration) * friction_coef
        y_temp = (y_speed + y_acceleration) * friction_coef

        x_speed = x_temp if abs(x_temp)>tol else 0
        y_speed = y_temp if abs(y_temp)>tol else 0
        
        #print('X: {}, Y: {}, R: {}, L: {}, U: {}, D: {}'.format(x_speed, y_speed, pressed_RIGHT, pressed_LEFT, pressed_UP, pressed_DOWN))
            
        #setting boundary for our main car
        if x_location < 0:
            x_location = 0
        if x_location > width-50:
            x_location = width-50
        
        if y_location < 0:
            y_location = 0
        if y_location > height-100:
            y_location = height-100


        #CHANGING COLOR WITH RGB VALUE, RGB = RED, GREEN, BLUE 
        screen.fill((0,0,0))

        #displaying the background image
        #screen.blit(bg,(0,0))

        #displaying our main car
        screen.blit(maincar,(x_location,y_location))

        #displaing other cars
        #screen.blit(car1,(car1X,car1Y))
        #screen.blit(car2,(car2X,car2Y))
        #screen.blit(car3,(car3X,car3Y))
        
        #updating the values
        x_location += x_speed
        y_location += y_speed

        #movement of the enemies
        #car1Y += 10
        #car2Y += 10
        #car3Y += 10


        #moving enemies infinitely
        #if car1Y > 670:
        #    car1Y = -100
        #if car2Y > 670:
        #    car2Y = -150
        #if car3Y > 670:
        #    car3Y = -200



        pygame.display.update()

gameloop()
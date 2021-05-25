UPDATE_RATE = 1/16000
SPRITE_SCALING = 0.4
RANDOM_SEED = 2

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1000
SCREEN_TITLE = "Pametnjakovič"

#MOVEMENT_SPEED = 5
ANGLE_SPEED = 5
ACCELERATION_UNIT = 0.1
FRICTION = 0.995 # z intervala [0,1]
GRIP = 0.04 # z intervala [0,1]
BRAKING_POWER = 0.6

MAX_SPEED = 3
TOL = 0.01
TOL_ANGLE = 0.01
TOL_CHECKPOINT = 20

TRACK_COLOR_PASSED = (0,255,00)
TRACK_COLOR_CURRENT = (255,0,0)
TRACK_COLOR_FUTURE = (255,255,255)

COLOR_WHITE = (255,255,255)

sh = SCREEN_HEIGHT
sw = SCREEN_WIDTH
TRACK1 = list(map(lambda x : [sw*x[0], sh*x[1]], [[0.1,0.1], [0.1, 0.3], [0.3, 0.3], [0.3, 0.5], [0.6, 0.3], [0.1,0.1]]))
#TRACK1 = list(map(lambda x : [sw*x[0], sh*x[1]], [[0.1,0.1], [0.1, 0.5], [0.5, 0.5],[0.1,0.1]]))
TRACK2 = list(map(lambda x : [sw*x[0], sh*x[1]], [[0.1,0.1], [0.1, 0.5], [0.3, 0.9], [0.4,0.3], [0.9,0.5],[0.1,0.1]]))
#TRACK1 = list(map(lambda x : [sw*x[0], sh*x[1]], [[0.1,0.1], [0.2, 0.4], [0.1, 0.8], [0.3, 0.9], [0.5, 0.6], [0.6,0.7], [0.7,0.9], [0.8,0.4], [0.5,0.2], [0.2,0.2],[0.2,0.1], [0.1,0.1]])) # rekord 1269


ALPHA = 0.5
MAX_DISTANCE = 10*SCREEN_WIDTH
#ALPHA_BRAKE = 0.8
#CONSECUTIVE_STEPS = 2
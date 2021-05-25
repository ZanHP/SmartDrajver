import arcade
from smartdriver.constants import *
from smartdriver.gui import MyGame, GeneticGame
import numpy as np
import tensorflow as tf
import neat
import os
from importlib import reload  
import pyglet


import sys

def main(start_gen=None):
    """ Main method """
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 
    window = GeneticGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=True, show=True, verbose=False, train=True, start_gen=start_gen)
    window.setup()
    arcade.run()

    
    # 20.63

if __name__ == "__main__":
    args = sys.argv[1:]
    # accept argument from which gen start
    if args:
        arg = args[0]
        start_gen = int(arg)
    else:
        start_gen = None
    main(start_gen=start_gen)

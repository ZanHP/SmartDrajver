import arcade
from smartdriver.constants import *
from smartdriver.gui import MyGame, GeneticGame
import numpy as np
import tensorflow as tf
import neat
import os
from importlib import reload  
import pyglet




def main():
    """ Main method """
    
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) 
    window = GeneticGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE, smart=True, show=True, verbose=False, train=True)
    window.setup()
    arcade.run()

    
    # 20.63

if __name__ == "__main__":

    main()

"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

from smartdriver.population import Population
from neat.checkpoint import Checkpointer as OriginalChecpointer


class Checkpointer(OriginalChecpointer):

    @staticmethod
    def restore_checkpoint(filename):
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            random.setstate(rndstate)
            #use our population
            return Population(config, (population, species_set, generation))

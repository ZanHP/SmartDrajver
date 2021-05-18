from collections import deque
from smartdriver.constants import *
import tensorflow as tf
import numpy as np
from tensorflow import keras
import random
import neat
import os

epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
max_epsilon = 1 # You can't explore more than 100% of the time
min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time


class GeneticAgent():
    def __init__(self, player_sprites, state_shape, weights=None, genomes=[], config=[]):
        #self.actions = np.array(["L", "R", ""])#, "U", "D"])
        self.actions = np.array(["L","", "R"])#, "U", "D"])
        self.player_sprites = player_sprites
        self.state_shape = state_shape

        self.finished = False
        self.total_training_rewards = 0

        self.max_time_steps = 60

        self.genomes = genomes

        self.player_neat = {}
        self.time_step = 0

        assert len(player_sprites) == len(genomes)
        i = 0
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            g.fitness = 0

            self.player_neat[self.player_sprites[i]] = {
                "genome": g,
                "net": net
            }
            i += 1
       
        #self.update_target_model()
        

    def draw(self):
        #print(len(self.player_sprites))
        for player_sprite in self.player_sprites:
            player_sprite.draw()


    def get_model(self,player_sprite):
        return self.player_neat[player_sprite]["net"]

    

    def do_summary_fitness_scores(self):

        for player_sprite in self.player_sprites:
            self.player_neat[player_sprite]["genome"].fitness = self.get_fitness_score(player_sprite)



    def do_training_step(self):

        #model = self.model
        #replay_memory = self.replay_memory
        
        #total_training_rewards = self.total_training_rewards
        #target_model = self.target_model

        self.time_step += 1
        if self.time_step == self.max_time_steps:
            self.finished =True
            print(self.time_step)

            # player je prišel do konca kroga

            self.do_summary_fitness_scores()
            return True

            # for player_sprite in self.player_sprites:
            #     # naredimo reset 
            #     player_sprite.next_checkpoint = 1
            #     player_sprite.track.next_checkpoint = 1
            #     player_sprite.center_x = self.player_sprite.track.checkpoints[0][0]
            #     player_sprite.center_y = self.player_sprite.track.checkpoints[0][1]
            #     player_sprite.accelerating = False
            #     player_sprite.braking = False
            #     player_sprite.speed = 0
            #     player_sprite.angle = 0
            #     player_sprite.change_angle = 0

            #     self.do_reproduction()


        for player_sprite in self.player_sprites:
            
            if player_sprite.finished:
                continue
                
            state = player_sprite.get_current_state()
            checkpoint = player_sprite.next_checkpoint
            net = self.get_model(player_sprite)

            predicted = net.activate((state))
            action = self.actions[np.argmax(predicted)]
            #print(action)
            #encoded_reshaped = np.array((player_sprite.get_current_state(),))
            #redicted = model.predict(encoded_reshaped).flatten()

            # odločimo se, kaj bo naslednja poteza
            #action = self.actions[np.argmax(predicted)]

            # glede na izbrano potezo se premaknemo
            is_not_finished = player_sprite.next_move_and_update(action)
            is_finished = not is_not_finished
            if is_finished:
                player_sprite.finished = True
                player_sprite.finish_time = self.time_step
            


        #print("Train iteration: ", self.train_iteration)                    
        return False


    def get_action_ind(self, action):
        #print("get_action:", action,np.where(self.actions == action))
        return np.where(self.actions == action)

    def encode_state(self, state, n_dims):
        return state




    def get_fitness_score(self, player_sprite):
        if player_sprite.finished:
            return player_sprite.finish_time
        else:
            return self.max_time_steps +  player_sprite.distance_to_nn_checkpoint()



import sys
import random
import time
import numpy as np
import math
import os

import pandas as pd
sys.path.insert(0, "path_finder")

import Genetic_Algorithm as gn
import pandas_operations
import Sample

class Game_Board():
     
    def __init__(self, board_size, model, sample_speed, obstacles):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size

        # 0 -> Empty, 1 -> Obstacle, 2 -> Start, 3 -> End
        self.board = np.zeros(shape=(self.board_width, self.board_height), dtype=np.int8)
        
        # Set the sample speed and the loop count
        self.sample_speed = sample_speed
        self.loop_count = 0

        # Default choices
        self.current_obstacle = None;self.paused = False

        # Initialization of counters & arrays
        self.frame_count = 0;self.epoch_count = 0
        self.obstacles = obstacles
        self.population = model.get_population()
        
        # Set the End Points
        self.init_end_points()
        
        # Initialize the end points and create the first generation
        self.distance_between_start_and_end = math.sqrt(
            (self.board_width - self.end_point[0])**2 + (self.board_height - self.end_point[1])**2
        )     
        self.refresh_rate = 8 * self.distance_between_start_and_end / self.sample_speed
        
        # Upload board attributes to the model
        self.model = model
        self.model.board = self
        self.model.assign_board_attributes()

        self.model.loop()
        self.init_screen()
        print("Game Board is initialized correctly")
        
        while True:
            self.update_samples()
           
    def update_samples(self):
        self.frame_count += 1

        # If the number of samples is less than the number of samples, create a new generation
        if not self.paused:
            self.loop_count += 1
            # get the count of status= dead samples
            #print(sum([1 for sample in self.model.get_population() if sample.status == "Alive"]))
            if sum([1 for sample in self.model.get_population() if sample.status == "Alive"]) == 0 or \
                self.refresh_rate < self.frame_count:
                self.frame_count = 0
                if self.model.no_change_counter > 10:
                    self.model.change_parameters(
                        self.model.learning_rate, self.model.mutation_rate
                        )

                self.model.loop()
                return
                
            for index,sample in enumerate(self.model.get_population()):
                new_x, new_y = sample.move()   
                new_position_color = self.get_color((new_x, new_y))
                self.model.handle_status(sample, new_position_color, index, self.frame_count)
            # If the frame count is greater than the reset limit, reset the samples
            
            if self.refresh_rate < self.frame_count:
                self.model.reset_samples();self.frame_count = 0
    

    def get_color(self, position):
        x, y = int(position[0]), int(position[1])
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"
        else:
            val = self.board[x,y]
            if val == 1: # Black
                return "#000000"
            elif val == 3: # Green
                return "#00ff00"
            else:
                return None

    # Inıtialize the screen (obstacles etc.)
    def init_screen(self):
        for obstacle in self.obstacles:
            self.board[
                obstacle["x"]:obstacle["x"]+obstacle["width"],
                obstacle["y"]:obstacle["y"]+obstacle["height"]
            ] = 1
            
    def init_end_points(self):
        self.base_size = (self.board_width // 20, self.board_height // 20)
        self.end_point = (self.board_width - self.board_width // 20, self.board_height // 2)

        self.board[self.end_point[0] - self.base_size[0] // 2:self.end_point[0] + self.base_size[0] // 2,
                   self.end_point[1] - self.base_size[1] // 2:self.end_point[1] + self.base_size[1] // 2] = 3

if __name__ == "__main__":
    #Test

    BOARD_SIZE = (700, 700)
    data_path = "log/results.hdf5" 
    modal = gn.Genetic_Algorithm(
        learning_rate=0.1, mutation_rate=0.1, select_per_epoch=20, generation_multiplier=20, sample_speed=20,
        board_size = BOARD_SIZE, dataframe_path=data_path, save_flag= False, load_flag= False
    )


    obj_width = 200
    obj_height = 400

    default_object = {
        "x": (BOARD_SIZE[0])//2-(obj_width//2),
        "y": (BOARD_SIZE[1])//2-(obj_height//2),
        "width": obj_width,
        "height": obj_height
    }

    board = Game_Board(
        board_size=BOARD_SIZE, model=modal, 
        sample_speed=20, obstacles=[default_object],
    )

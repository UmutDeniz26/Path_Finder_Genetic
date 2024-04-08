import sys
import numpy as np

import pandas as pd
sys.path.insert(0, "path_finder")

try:
    from Sample import Sample
except:
    from scritps.Sample import Sample

class Game_Board():
     
    def __init__(self, board_size, model, obstacles):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.board_size = board_size

        # 0 -> Empty, 1 -> Obstacle, 2 -> Start, 3 -> End
        self.board = np.zeros(shape=(self.board_width, self.board_height), dtype=np.int8)
        
        # Initialization of counters & arrays
        self.frame_count = 0; self.epoch_count = 0; self.loop_count = 0;self.move_cnt = 0
        self.obstacles = obstacles; self.population = []
        
        # Draw essential objects
        self.draw_end_point()
        self.draw_obstacles()

        # Initialize the end points and create the first generation
        self.distance_between_start_and_end = np.linalg.norm(
            np.array(board_size) - np.array(self.end_point)
        )
        
        # Upload board attributes to the model
        self.model = model
        self.model.board = self
        self.model.assign_board_attributes()

        self.model.model_loop() if self.model.learning_rate != 0 else None
        self.refresh_rate = 8 * self.distance_between_start_and_end / self.model.sample_speed
        
        print("Game Board is initialized correctly")
        while True:
            self.model.timer.start_new_timer("Board Loop") if self.model.timer is not None else None
            self.update_samples()
            self.model.timer.stop_timer("Board Loop") if self.model.timer is not None else None
        
    def update_samples(self):
            
        # If the frame count is greater than the reset limit, reset the samples
        if self.refresh_rate < self.frame_count:
            self.model.reset_samples();self.frame_count = 0
        
        living_samples = [ elem for elem in self.model.get_population() if elem.status=="Alive" ]
        #print(len(living_samples))
        # If the number of samples is less than the number of samples, create a new generation
        if len(living_samples) == 0:
            self.frame_count = 0
            if self.model.not_learning_flag:
                best_control_history = self.model.evulation_results[0]["sample"].controls
                best_sample = Sample(
                    board_size        = self.board_size, 
                    speed             = self.model.sample_speed,
                    external_controls = best_control_history
                )
                best_sample.set_score(self.model.evulation_results[0]["score"])
                
                print("\n------------------\n, Len of the control history: ",
                    len(best_control_history)," Last move cnt:",self.move_cnt,"\n",
                    best_control_history,"\n\n"
                )
                self.move_cnt=0
                self.model.population = [best_sample]
            else:
                self.model.model_loop()
            return
        # If len of the population is greater than 0, move the samples
        else:
            
            for sample in living_samples:
                new_x, new_y = sample.move()
                new_position_color = self.get_color(new_x, new_y)
                self.model.handle_status(sample, new_position_color)
                print("(",new_x, new_y,"),",end=" ") if self.model.learning_rate == 0 else None
                self.move_cnt += 1
            self.frame_count += 1
        
    def get_color(self, x, y):
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
    def draw_obstacles(self):
        for obstacle in self.obstacles:
            self.board[
                obstacle["x"]:obstacle["x"]+obstacle["width"],
                obstacle["y"]:obstacle["y"]+obstacle["height"]
            ] = 1 # 1 means obstacle
            
    def draw_end_point(self):
        self.base_size = (self.board_width // 20, self.board_height // 20)
        self.end_point = (self.board_width - self.board_width // 20, self.board_height // 2)

        self.board[self.end_point[0] - self.base_size[0] // 2:self.end_point[0] + self.base_size[0] // 2,
                   self.end_point[1] - self.base_size[1] // 2:self.end_point[1] + self.base_size[1] // 2] = 3

if __name__ == "__main__":
    pass
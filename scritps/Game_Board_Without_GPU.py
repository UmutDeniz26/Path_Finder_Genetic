import sys
import numpy as np
import pandas as pd
import logging

try:
    from Sample import Sample
    from Game_Board import Game_Board_GPU
except:
    from scritps.Sample import Sample
    from scritps.Game_Board import Game_Board as Game_Board_GPU

class Game_Board():
     
    def __init__(self, board_size, model, obstacles):
        super().__init__()

        logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(logging.DEBUG)

        # Essential attributes
        self.board_width, self.board_height = board_size
        self.board_size = board_size

        # 0 -> Empty, 1 -> Obstacle, 2 -> Start, 3 -> End
        self.board = np.zeros(shape=(self.board_width, self.board_height), dtype=np.uint8)
        
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

        self.refresh_rate = \
            ( 8 * self.distance_between_start_and_end / self.model.sample_speed )
                
        logging.info("Game Board is initialized correctly")
        
    def update_samples(self):
            
        # If the frame count is greater than the reset limit, reset the samples
        if self.refresh_rate < self.frame_count:
            self.model.reset_samples();self.frame_count = 0
        
        living_samples = [ elem for elem in self.model.get_population() if elem.status=="Alive" ]
        #print(len(living_samples))
        # If the number of samples is less than the number of samples, create a new generation
        if not living_samples:
            self.frame_count = 0
            if self.model.not_learning_flag:
                best_control_history = self.model.evulation_results[0]["sample"].controls
                best_sample = Sample(
                    board_size        = self.board_size, 
                    speed             = self.model.sample_speed,
                    external_controls = best_control_history
                )
                best_sample.set_score(self.model.evulation_results[0]["score"])
                
                logging.debug(f"Model is in not learning mode, best control history: {best_control_history}")
                logging.debug(f"Best sample score: {self.model.evulation_results[0]['score']}")
                logging.debug(f"Best sample move count: {self.move_cnt}")
                
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
                #logging.debug(f"({new_x}, {new_y}),")
                self.move_cnt += 1
            self.frame_count += 1
        
    def get_color(self, x, y):
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"
        else:
            # select mxm square
            m = 4
            val = self.board[
                max(x - m, 0):min(x + m, self.board_width),
                max(y - m, 0):min(y + m, self.board_height)
            ].max() 

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
import sys
import numpy as np
import pandas as pd
import logging


class Game_Board():
     
    def __init__(self, board_size, model, obstacles):
        super().__init__()

        logging.basicConfig(filename='log/app.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(logging.DEBUG)

        # Essential attributes
        self.board_width, self.board_height = board_size
        self.board_size = board_size
        self.pixel_padding = 4

        # 0 -> Empty, 1 -> Obstacle, 2 -> Start, 3 -> End
        self.board = np.zeros(shape=(self.board_width, self.board_height), dtype=np.uint8)
        
        # Initialization of counters & arrays      
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
        self.model.assign_board_attributes( self )

                # Pre-calculate the padding for the board
        self.board_padding = np.copy(self.board)
        for i in range(self.board_padding.shape[0]):
            for j in range(self.board_padding.shape[1]):
                self.board_padding[i, j] = self.board[
                    max(i - self.pixel_padding, 0):min(i + self.pixel_padding, self.board_width),
                    max(j - self.pixel_padding, 0):min(j + self.pixel_padding, self.board_height)
                ].max()

        logging.info("Game Board is initialized correctly")
        
    # To run the model, execute the main loop
    def update_samples(self):      
        self.model.main_loop()
        
    def get_color(self, x, y):
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"
        
        val = self.board_padding[x, y]

        if val == 0: # White
            return None
        elif val == 1: # Black
            return "#000000"
        elif val == 3: # Green
            return "#00ff00"
        
        raise ValueError("Unknown color value")

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
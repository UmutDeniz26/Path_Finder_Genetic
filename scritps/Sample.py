import math
import random
import numpy as np

id_counter = 0


class Sample:
    def __init__(self, board_size, speed=20, external_controls=[]):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.speed = speed
        self.status = "Alive"
        self.size = 7
        self.score = 0
        self.move_counter = 0
        self.final_move_count = 0


        # Precalculate the cosine and sine values for each angle
        self.cosine_values =\
            np.multiply( np.array([math.cos(math.radians(angle)) for angle in range(360)]), self.speed ).astype(int)
        self.sine_values =\
            np.multiply( np.array([math.sin(math.radians(angle)) for angle in range(360)]), self.speed ).astype(int)

        # End point
        self.target_point = (self.board_width, self.board_height // 2)
        
        # Set the initial position of the sample
        self.spawn_point = (0, self.board_height // 2)
        self.positon = self.spawn_point

        # Initialization of counters & arrays
        self.controls = external_controls if len(external_controls) != 0 else []
        self.assign_id()

    def set_controls(self, assign_mode="rand", external_controls=[], control_count=100):
            
        # Reset the position of the sample
        self.positon = self.spawn_point
        self.move_counter = 0
        self.set_score(0)

        if len(external_controls) != 0 or assign_mode == "copy":
            self.controls = external_controls
        else:
            if assign_mode == "rand":
                self.controls = [random.uniform(0, 360) for i in range(control_count)]
            elif assign_mode == "zero":
                self.controls = [0 for i in range(control_count)]
            elif assign_mode == "empty":
                self.controls = []
            else:
                raise ValueError("Unknown assign_mode")
                

    def move(self):
        # Get the angle of the next move
        while self.move_counter >= len(self.controls):
            self.controls = np.append(self.controls, random.uniform(0, 360))
        angle = int( self.controls[self.get_and_increase_move_counter()] )

        x, y = self.get_pos()

        # Update the position
        # old -> self.set_pos((int(x + self.speed * np.cos(np.radians(angle))),int(y + self.speed * np.sin(np.radians(angle)))))
        self.set_pos((
            x + self.cosine_values[angle], # These cosine and sine values are precalculated
            y + self.sine_values[angle]    # While calculating, its values are multiplied by the speed!!!
        ))
        
        return self.get_pos()
    
    def calculate_fitness(self):
        # Calculate the distance between the sample and the end point
        distance = np.linalg.norm(np.array(self.get_pos()) - np.array(self.target_point))
        distance = 1 if distance == 0 else distance
        return 1 / distance

    
    # Return the control history and the final score of the sample
    def kill_sample_get_score(self):
        # Set the final move count and reset the move counter
        #self.final_move_count = self.move_counter + self.final_move_count
        self.final_move_count = self.move_counter
        self.set_score( self.calculate_fitness( ) )

        return {"sample": self, "score": self.get_score(), "final_move_count": self.final_move_count}


###################################################################################

    # Setters and getters 
    def get_and_increase_move_counter(self):
        self.move_counter += 1
        return self.move_counter - 1
    def set_score(self, score):
        self.score = score
    def get_score(self):
        return self.score
    def set_pos(self, position):
        self.positon = position
    def get_pos(self):
        return self.positon
    def get_controls(self):
        return self.controls
    def get_move_counter(self):
        return self.move_counter
    def get_status(self):
        return self.status
    def set_status(self, status):
        self.status = status
    def get_ID(self):
        return self.ID
    # Increase the move counter and return the previous value
    def get_and_increase_move_counter(self):
        self.move_counter += 1
        return self.move_counter - 1
    
    # Print the features of the sample
    def print_features(self):
        for key, value in self.__dict__.items():
            print(key, ":", value)
        print("Current Position: ", self.x(), self.y())
    def __str__(self):
        self.print_features()
        return ""    
        # Return the bounding rectangle of the sample
    def assign_id(self):
        global id_counter
        self.ID = id_counter
        id_counter += 1
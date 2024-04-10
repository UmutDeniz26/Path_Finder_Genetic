import math
from PyQt5.QtCore import Qt, QRectF, QTimer, QEvent
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsEllipseItem
from PyQt5.QtCore import QPointF

from scritps.Sample import Sample
import random
import numpy as np
    
id_counter = 0

cosine_values = np.array([math.cos(math.radians(angle)) for angle in range(360)])
sine_values = np.array([math.sin(math.radians(angle)) for angle in range(360)])

class Sample_qt(QGraphicsItem):
    def __init__( self, board_size, speed=20,
                external_controls=[], sample_obj=None
            ):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.color = Qt.blue
        self.speed = speed
        self.status = "Alive"
        self.size = 7
        self.score = 0
        self.move_counter = 0
        self.final_move_count=0

        # End point
        self.target_point = QPointF(self.board_width, self.board_height // 2)
        
        # Set the initial position of the sample
        self.spawn_point = (0, self.board_height // 2)
        self.setPos(self.spawn_point[0], self.spawn_point[1])

        # Initialization of counters & arrays
        self.controls = external_controls if len(external_controls) != 0 else []


        if sample_obj is not None:
            self.board_width = sample_obj.board_width
            self.board_height = sample_obj.board_height
            self.speed = sample_obj.speed
            self.color = sample_obj.color
            self.size = sample_obj.size
            self.target_point = sample_obj.target_point
            self.spawn_point = sample_obj.spawn_point
            self.controls = sample_obj.controls
            self.move_counter = sample_obj.move_counter
            self.score = sample_obj.score
            self.final_move_count = sample_obj.final_move_count

        self.assign_id()
        
    # Control angles
    def set_controls(self, assign_mode = "rand", external_controls = [], control_count = 100):
        
        self.setPos(self.spawn_point[0], self.spawn_point[1])
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
    
    # Returns new position of the sample and sets the new position
    def move(self):

        # Get the angle of the next move
        while self.move_counter >= len(self.controls):
            self.controls = np.append(self.controls, random.uniform(0, 360))
        angle = self.controls[self.get_and_increase_move_counter()]
        
        # Calculate the new position of the sample
        new_x = int(self.x() + self.speed * math.cos(math.radians(angle)))
                    #cosine_values[int(angle)%360])
        new_y = int(self.y() + self.speed * math.sin(math.radians(angle)))
                    #sine_values[int(angle)%360])

        # This is where we move the sample
        self.setPos(new_x, new_y)

        return new_x, new_y
    
    def calculate_fitness(self):
        # Calculate the distance between the sample and the end point
        distance = np.linalg.norm(
            np.array([self.x(), self.y()]) - np.array([self.target_point.x(), self.target_point.y()])
        )

        # score=1/distance can be another option to calculate fitness
        score_local = 1000 if distance == 0 else (1 / distance)
        
        return score_local        
            
    # Return the control history and the final score of the sample
    def kill_sample_get_score(self):
        # Set the final move count and reset the move counter
        #self.final_move_count = self.move_counter + self.final_move_count
        self.final_move_count = self.move_counter
        
        self.set_score( self.calculate_fitness() )

        return {"sample": self, "score": self.get_score(), "final_move_count": self.final_move_count}


##############################################################################################################

    # Setters and getters 
    def set_score(self, score):
        self.score = score
    def get_score(self):
        return self.score
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
    def boundingRect(self):
        return QRectF(-self.size/2, -self.size/2, self.size, self.size)
    # Paint the sample (Don't change its parameters)
    def paint(self, painter, option, widget):
        painter.setBrush(self.color)
        painter.drawEllipse(-self.size/2, -self.size/2, self.size, self.size)
    # It assings a id for each sample
    def assign_id(self):
        global id_counter
        id_counter += 1
        self.ID = id_counter
import math
from PyQt5.QtCore import Qt, QRectF, QTimer, QEvent
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsEllipseItem
from PyQt5.QtCore import QPointF
import random

id_counter = 0

class Sample(QGraphicsItem):
    def __init__( self, board_size, speed=20,
                external_controls=[], init_control_mode="rand", 
                max_control_length=100
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
        self.position_history = []

        # End point
        self.target_point = QPointF(self.board_width, self.board_height // 2)
        
        # Set the initial position of the sample
        self.spawn_point = (50, self.board_height // 2)
        self.setPos(self.spawn_point[0], self.spawn_point[1])

        # Initialization of counters & arrays
        if len(external_controls) != 0:
            self.controls = external_controls
        else:
            self.set_controls(
                assign_mode=init_control_mode, control_count=max_control_length)

        self.assign_id()

    # It assings a id for each sample
    def assign_id(self):
        global id_counter
        id_counter += 1
        self.ID = id_counter
        
    # Control angles
    def set_controls(self, assign_mode = "rand", external_controls = [], control_count = 100):
        
        self.set_score(0)
        self.move_counter = 0
        self.setPos(self.spawn_point[0], self.spawn_point[1])
        self.position_history.clear()

        if len(external_controls) != 0:
            self.controls = external_controls
        else:
            if assign_mode == "rand":
                self.controls = [random.uniform(0, 360) for i in range(control_count)]
            elif assign_mode == "zero":
                self.controls = [0 for i in range(control_count)]
            elif assign_mode == "empty":
                self.controls = []
            elif assign_mode == "copy":
                raise ValueError("External controls are empty")
            else:
                raise ValueError("Unknown assign_mode")
    
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

    # Increase the move counter and return the previous value
    def get_and_increase_move_counter(self):
        self.move_counter += 1
        return self.move_counter - 1

    # Setters and getters for the score
    def set_score(self, score):
        if self.ID == 0:
            print()
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

    # Returns new position of the sample and sets the new position
    def move(self):
        if self.move_counter == 0:
            self.set_score(0)

        # Get the angle of the next move
        if self.move_counter >= len(self.controls):
            self.controls.append(random.uniform(0, 360))
        angle = self.controls[self.get_and_increase_move_counter()]
        
        # Calculate the new position of the sample
        new_x = int( self.x() + self.speed * math.cos(math.radians(angle)) )
        new_y = int( self.y() + self.speed * math.sin(math.radians(angle)) )

        # This is where we move the sample
        self.setPos(new_x, new_y)
        self.position_history.append((new_x, new_y))

        # After every move, calculate the fitness of the sample for the target point
        # self.calculate_fitness( self.target_point )

        return new_x, new_y
    
    def calculate_fitness(self, end_point):
        # Calculate the distance between the sample and the end point
        distance = math.sqrt(
            (self.x() - end_point.x())**2 +
            (self.y() - end_point.y())**2
        )

        # score=1/distance can be another option to calculate fitness
        score_local = 1000 if distance == 0 else (1 / distance)
        
        # Update score
        # self.set_score( score_local )

        # score_local and self.score !are differenet! if you use cummulative approach
        return score_local        
            
    # Return the control history and the final score of the sample
    def kill_sample_get_score(self):
        # Set the final move count and reset the move counter
        #self.final_move_count = self.move_counter + self.final_move_count
        self.final_move_count = len(self.position_history)
        self.move_counter = 0
        
        self.set_score( self.calculate_fitness( self.target_point ) )

        return {"sample": self, "score": self.get_score()}
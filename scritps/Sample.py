import math
from PyQt5.QtCore import Qt, QRectF, QTimer, QEvent
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsEllipseItem
from PyQt5.QtCore import QPointF
import random

id_counter = 0

class Sample(QGraphicsItem):
    def __init__( self, board_size, speed=20, external_controls=[], init_control_mode="rand", max_control_length=100 ):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.color = Qt.blue;self.speed = speed;self.score = 0;self.size = 1
        self.x_axis_coefficient = 2;self.y_axis_coefficient = 1
        self.status = "Alive"

        # End point
        self.target_point = QPointF(self.board_width, self.board_height // 2)
        
        # Set the initial position of the sample
        self.spawn_point = (0, self.board_height // 2)
        self.setPos(self.spawn_point[0], self.spawn_point[1])

        # Initialization of counters & arrays
        if len(external_controls) != 0:
            self.controls = external_controls
        else:
            self.set_controls(assign_mode=init_control_mode, control_count=max_control_length)

        self.control_history = []
        self.move_counter = 0
        
        self.assign_id()

    # It assings a id for each sample
    def assign_id(self):
        global id_counter
        self.ID = id_counter
        id_counter += 1

    # Control angles
    def set_controls(self, assign_mode = "rand", external_controls = [], control_count = 100):
        if len(external_controls) != 0:
            self.controls = external_controls
        else:
            if assign_mode == "rand":
                self.controls = [random.uniform(0, 360) for i in range(control_count)]
            elif assign_mode == "zero":
                self.controls = [0 for i in range(control_count)]
            else:
                raise ValueError("Unknown assign_mode")
    
    # Print the features of the sample
    def print_features(self):
        print(f"\nID: {self.ID}, Speed: {self.speed}\nScore: {self.score},Position: ({self.x()}, {self.y()})")

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
        return self.move_counter-1

    # Setters and getters for the score
    def set_score(self, score):
        self.score = score
    def get_score(self):
        return self.score

    # Handle the angle of the next move
    def handle_angle(self, move_counter):

        # If in control array there is a angle information use it, otherwise generate random one
        if move_counter < len(self.controls):
            angle = self.controls[move_counter]
        else:
            angle = random.uniform(0, 360)
        self.control_history.append(angle)
        return angle

    # Returns new position of the sample and sets the new position
    def move(self):

        # Get the angle of the next move
        angle = self.handle_angle( self.get_and_increase_move_counter() )
        
        # Calculate the new position of the sample
        dx = self.speed * math.cos(math.radians(angle))
        dy = self.speed * math.sin(math.radians(angle))
        new_x = int( self.x() + dx )
        new_y = int( self.y() + dy )

        # This is where we move the sample
        self.setPos(new_x, new_y)

        # After every move, calculate the fitness of the sample for the target point
        self.calculate_fitness( self.target_point )

        return new_x, new_y
    
    def calculate_fitness(self, end_point):
        # Calculate the distance between the sample and the end point
        distance = math.sqrt(
            self.x_axis_coefficient  * (self.x() - end_point.x())**2 +
            self.y_axis_coefficient  * (self.y() - end_point.y())**2
        )

        # score=1/distance can be another option to calculate fitness
        if distance == 0:
            score_local = 1000
        else:
            score_local = 1 / distance

        # Update score
        self.set_score( score_local )

        # score_local and self.score !are differenet! if you use cummulative approach
        return self.get_score()
        
            
    # Return the control history and the final score of the sample
    def get_control_history_and_final_score(self):
        
        last_score = self.calculate_fitness( self.target_point )
        
        # It would overwrite if fitness calculation is ( score = 1/distance )
        # Only meaningfull when fitness calculation is ( score += 1/distance )
        #final_score_local = cummulative_score + last_score * 1000 
        #self.set_score(final_score_local)
        return {"sample": self, "score": self.get_score()}
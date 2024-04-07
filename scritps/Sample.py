import math
from PyQt5.QtCore import Qt, QRectF, QTimer, QEvent
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsEllipseItem
from PyQt5.QtCore import QPointF
import random

id_counter = 0

class Sample(QGraphicsItem):
    def __init__(self, board_size, speed=20, external_controls=[]):
        super().__init__()
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.color = Qt.blue;self.speed = speed;self.score = 0;self.size = 7
        self.x_axis_coefficient = 2;self.y_axis_coefficient = 1

        # End point
        self.target_point = QPointF(self.board_width, self.board_height // 2)
        
        # Set the initial position of the sample
        self.spawn_point = (0, self.board_height // 2)
        self.setPos(self.spawn_point[0], self.spawn_point[1])

        # Initialization of counters & arrays
        self.controls = external_controls
        self.control_history = []
        self.move_counter = 0
        
        self.assign_id()

    # It assings a id for each sample
    def assign_id(self):
        global id_counter
        self.ID = id_counter
        id_counter += 1
    
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
        new_x = self.x() + dx
        new_y = self.y() + dy

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
        score_local = 1 / distance

        # Add living time to the score
        score_local += (self.move_counter /1000)

        # Update score
        self.set_score( self.get_score() + score_local )

        # score_local and self.score !are differenet! if you use cummulative approach
        return score_local
        
            
    # Return the control history and the final score of the sample
    def get_control_history_and_final_score(self):
        
        last_score = self.calculate_fitness( self.target_point )
        cummulative_score = self.get_score()

        # It would overwrite if fitness calculation is ( score = 1/distance )
        # Only meaningfull when fitness calculation is ( score += 1/distance )
        final_score_local = cummulative_score + last_score * 1000 
        self.set_score(final_score_local)
        return {"control_history": self.control_history, "score": self.get_score(), "ID": self.ID}
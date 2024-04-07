import sys
import random
import numpy as np
import math
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsRectItem
import time

sys.path.insert(0, "path_finder")
from scritps.Genetic_Algorithm import Genetic_Algorithm
from scritps.Sample import Sample

LEARNING_RATE = 0.1
MUTATION_RATE = 0.1
SELECT_PER_EPOCH = 10
MULTIPLIER = 10

class Game_Board(QGraphicsView):
     
    def __init__(self, board_size, population_size=(SELECT_PER_EPOCH*MULTIPLIER)):
        super().__init__()

        # Set the scene and the scene rectangle
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, *board_size)
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.population_size = population_size
        self.sample_speed = 20;self.loop_count = 0

        #Default choices
        self.current_obstacle = None;self.paused = False

        # Initialization of counters & arrays
        self.frame_count = 0;self.epoch_count = 0
        self.population = []
        self.obstacles = []

        # Set the timer and the mouse tracking
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_samples)
        self.timer.start(5)

        # Set the End Points
        self.init_end_points()
        
        # Initialize the end points and create the first generation
        self.distance_between_start_and_end = math.sqrt(
            (self.board_width - self.end_point[0])**2 + (self.board_height - self.end_point[1])**2
        )     

        self.model = Genetic_Algorithm(
            LEARNING_RATE, MUTATION_RATE, SELECT_PER_EPOCH, MULTIPLIER, board_object=self, sample_speed=self.sample_speed
        )
        self.model.prepare_next_generation()
        self.init_screen()
        print("Game Board is initialized correctly")

    # Print the epoch information
    def print_epoch_info(self):
        self.epoch_count += 1
        print(f"""      
        ===================================== Epoch: "{self.epoch_count} " =====================================
        CONSTANTS:
            LEARNING_RATE   : {LEARNING_RATE:7.2f} | MUTATION_RATE: {MUTATION_RATE:7.2f} 
            SELECT_PER_EPOCH: {SELECT_PER_EPOCH:7.1f} | MULTIPLIER   : {MULTIPLIER:7.1f}
            SAMPLE_SPEED    : {self.sample_speed:7.1f} | BOARD_SIZE   : {self.board_width}x{self.board_height}
        """)

            
    def update_samples(self):
        global MULTIPLIER, SELECT_PER_EPOCH, LEARNING_RATE, MUTATION_RATE
        self.frame_count += 1

        # If the frame count is greater than the reset limit, reset the samples
        refresh_rate = 4 * self.distance_between_start_and_end / self.sample_speed
        if refresh_rate < self.frame_count:
            self.frame_count = 0
            self.model.reset_samples()

        # If the number of samples is less than the number of samples, create a new generation
        if not self.paused:
            self.loop_count += 1
            if len(self.model.get_population()) == 0:
                if self.model.no_change_counter > 20:
                    LEARNING_RATE,MUTATION_RATE = self.model.change_parameters(LEARNING_RATE, MUTATION_RATE)

                self.print_epoch_info() if self.loop_count % 10 + 5 == 0 else None
                self.model.prepare_next_generation()
                
                return
                
            for sample in self.model.get_population():
                new_x, new_y = sample.move()   
                new_position_color = self.get_color((new_x, new_y))
                self.model.handle_status(sample, new_position_color)

            self.render_screen(self.model.get_population())
                
    def init_screen(self):
        for obstacle in self.obstacles:
            self.scene().addItem(obstacle)
        for sample in self.model.get_population():
            self.scene().addItem(sample)
            
    
    def render_screen(self, population):
        # Remove the previous samples
        samples_to_remove = [item for item in self.scene().items() if isinstance(item, Sample)]
        for item in samples_to_remove:
            self.scene().removeItem(item)
        
        # Add the new samples
        for sample in population:
            self.scene().addItem(sample)
    

    def get_color(self, position):
        x, y = position
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"
        else:
            for item in self.scene().items(QRectF(x, y, 1, 1)):
                if isinstance(item, QGraphicsRectItem) and item == self.end_point_item:
                    return Qt.green
                elif isinstance(item, QGraphicsRectItem) and item in self.obstacles:
                    return Qt.black
            return None

    def init_end_points(self):
        self.base_size = (self.board_width // 20, self.board_height // 20)
        self.end_point = (self.board_width - self.board_width // 20, self.board_height // 2)
        end_color = Qt.green
        self.end_point_item = QGraphicsRectItem()
        self.end_point_item.setBrush(end_color)
        self.end_point_item.setRect(self.end_point[0] - self.base_size[0] / 2, 
                                     self.end_point[1] - self.base_size[1] / 2, 
                                     *self.base_size)
        self.scene().addItem(self.end_point_item)

    # Key, mouse events
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.paused = not self.paused
            if self.paused:
                self.timer.stop()
            else:
                self.timer.start(75)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.current_obstacle = QGraphicsRectItem()
            self.current_obstacle.setBrush(Qt.black)
            self.current_obstacle.setRect(event.pos().x(), event.pos().y(), 0, 0)
            self.scene().addItem(self.current_obstacle)

    def mouseMoveEvent(self, event):
        if self.current_obstacle is not None:
            rect = self.current_obstacle.rect()
            rect.setWidth(event.pos().x() - rect.x())
            rect.setHeight(event.pos().y() - rect.y())
            self.current_obstacle.setRect(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_obstacle is not None:
            self.obstacles.append(self.current_obstacle)
            self.current_obstacle = None
            print("Obstacle added, ", len(self.obstacles), " obstacles in total")
            print("Sample count: ", len(self.model.get_population()))
            self.results = []
            self.model.reset_model()

            
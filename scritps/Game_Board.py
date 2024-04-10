import sys
import random
import numpy as np
import math
from PyQt5.QtCore import Qt, QRectF, QTimer
from PyQt5.QtGui import QTransform
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsRectItem
import time

try:
    from scritps.Genetic_Algorithm import Genetic_Algorithm
    from scritps.Sample_qt import Sample_qt
    from scritps.Sample import Sample
except:
    from Genetic_Algorithm import *
    from scritps.Sample_qt import Sample_qt
    from scritps.Sample import Sample

class Game_Board(QGraphicsView):
     
    def __init__(self, board_size, model, obstacles,timer=None, hybrid_interval = 0):
        super().__init__()

        # Set the scene and the scene rectangle
        self.setScene(QGraphicsScene(self))
        self.setSceneRect(0, 0, *board_size)
        
        # Essential attributes
        self.board_width, self.board_height = board_size
        self.board_size = board_size
        self.loop_count = 0
        self.hybrid_interval = hybrid_interval

        if timer is not None:
            self.mytimer = timer

        #Default choices
        self.current_obstacle = None;self.paused = False

        # Initialization of counters & arrays
        self.epoch_count = 0;self.population = []
        self.obstacles = obstacles;self.move_cnt = 0

        # Set the End Points
        self.init_end_points()
        
        # Initialize the end points and create the first generation
        self.distance_between_start_and_end = np.linalg.norm(
            np.array(board_size) - np.array(self.end_point)
        )

        # Upload board attributes to the model
        self.model = model
        self.model.board = self
        self.model.assign_board_attributes( self )

        self.model.progress_to_next_epoch()
        self.init_screen()
        print("Game Board is initialized correctly")

        # Set the timer and the mouse tracking
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_samples)
        self.timer.start(50)

    def update_samples(self):
        # If the hybrid interval is reached, quit the application
        if hasattr(self, "mytimer"):
            self.mytimer.update_timers()
            timer_with_GPU_index = self.mytimer.get_timer_index("with-GPU")
            if self.mytimer.timers[timer_with_GPU_index]["current"] > self.hybrid_interval:
                QApplication.quit()
        
        # If the number of samples is less than the number of samples, create a new generation
        if not self.paused: 
            
            self.model.main_loop()
            self.render_screen(self.model.get_population())

    def init_screen(self):
        for obstacle in self.obstacles:
            self.scene().addItem(obstacle)
        for sample in self.model.get_population():
            self.scene().addItem(sample)
            
    def render_screen(self, population):
        # Remove the previous samples
        if len(self.scene().items()) > 0:
            samples_to_remove = [item for item in self.scene().items() if isinstance(item, Sample_qt)]
            for item in samples_to_remove:
                self.scene().removeItem(item)
            
        # Add the new samples
        for sample in population[:self.model.select_per_epoch]:
            self.scene().addItem(sample)
    
    def get_color(self, x, y):        
        if x < 0 or x >= self.board_width or y < 0 or y >= self.board_height:
            return "Out of bounds"

        for item in self.scene().items(QRectF(x, y, 1, 1)):
            if isinstance(item, QGraphicsRectItem):
                # color = item.brush().color().name()
                return item.brush().color().name()

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
            self.results = []
            self.model.reset_model()

            #print all obstacles in form of default object dict:
            print([{"x": int(obstacle.rect().x()), "y": int(obstacle.rect().y()), 
                    "width": int(obstacle.rect().width()), "height": int(obstacle.rect().height())} 
                    for obstacle in self.obstacles])



if __name__ == "__main__":
    pass

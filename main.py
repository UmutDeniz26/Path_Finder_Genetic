import sys
from PyQt5.QtWidgets import QApplication

from scritps.Game_Board import Game_Board
from scritps.Genetic_Algorithm import Genetic_Algorithm

LEARNING_RATE = 0.1
MUTATION_RATE = 0.1
SELECT_PER_EPOCH = 10   
MULTIPLIER = 10


def main():
    app = QApplication(sys.argv)

    sample_speed = 20

    model = Genetic_Algorithm(
            LEARNING_RATE, MUTATION_RATE, SELECT_PER_EPOCH, MULTIPLIER, sample_speed=sample_speed
        )

    game_board = Game_Board( board_size= (700, 700),
                            model=model,
                            population_size=SELECT_PER_EPOCH*MULTIPLIER,
                            sample_speed=sample_speed)
    game_board.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

import sys
from PyQt5.QtWidgets import QApplication

sys.path.insert(0, "path_finder")
from scritps.Game_Board import Game_Board

def main():
    app = QApplication(sys.argv)
    BOARD_SIZE = (700, 700)
    game_board = Game_Board(BOARD_SIZE)
    game_board.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

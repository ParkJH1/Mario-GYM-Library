from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSlot, pyqtSignal, QObject
import retro
import sys
import os
import numpy as np


import retro
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSlot, pyqtSignal, QObject
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QMainWindow, QPushButton, QLineEdit
import numpy as np
import sys
import random
import time
import os

env: retro.RetroEnv = None


class FrameTimer(QThread):
    end_frame_signal = pyqtSignal()
    update_frame_signal = pyqtSignal()

    def __init__(self, game_speed):
        super().__init__()
        self.game_speed = game_speed
        self.end_frame = True
        self.next_frame_ready = False
        self.is_game_running = True

    def end_frame_event(self):
        if self.game_speed == 2:
            self.next_frame_ready = True
            self.update_frame_signal.emit()
        else:
            self.end_frame = True

    def stop_game(self):
        self.is_game_running = False

    def run(self):
        if self.game_speed != 2:
            while self.is_game_running:
                if self.game_speed == 0:
                    self.msleep(1000 // 60)
                elif self.game_speed == 1:
                    self.msleep(1000 // 120)
                else:
                    self.msleep(1000 // 60)
                self.next_frame_ready = True
                while not self.end_frame:
                    self.msleep(1)
                self.end_frame = False
                self.update_frame_signal.emit()
        else:
            self.end_frame_event()


class Mario(QWidget):
    def __init__(self, main, game_speed):
        super().__init__()
        self.setWindowTitle('Mario')
        self.main = main
        self.game_speed = game_speed

        global env
        if env is not None:
            env.close()
        env = retro.make(game='SuperMarioBros-Nes', state=f'Level1-1')
        self.env = env
        screen = self.env.reset()

        self.key_up = False
        self.key_down = False
        self.key_left = False
        self.key_right = False

        self.key_a = False
        self.key_b = False

        self.screen_width = screen.shape[0] * 2
        self.screen_height = screen.shape[1] * 2

        self.setFixedSize(self.screen_width, self.screen_height)
        self.move(100, 100)

        self.screen_label = QLabel(self)
        self.screen_label.setGeometry(0, 0, self.screen_width, self.screen_height)

        self.press_buttons = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.frame_timer = FrameTimer(self.game_speed)
        self.frame_timer.end_frame_signal.connect(self.frame_timer.end_frame_event)
        self.frame_timer.update_frame_signal.connect(self.update_frame)
        self.frame_timer.start()

    def update_frame(self):
        screen = self.env.get_screen()
        qimage = QImage(screen, screen.shape[1], screen.shape[0], QImage.Format.Format_RGB888)
        pixmap = QPixmap(qimage)
        pixmap = pixmap.scaled(self.screen_width, self.screen_height, Qt.AspectRatioMode.IgnoreAspectRatio)
        self.screen_label.setPixmap(pixmap)

        self.update()

    def paintEvent(self, event):
        if not self.frame_timer.next_frame_ready:
            return
        self.frame_timer.next_frame_ready = False

        ram = self.env.get_ram()
        if ram[0x001D] == 3 or ram[0x0E] in (0x0B, 0x06) or ram[0xB5] == 2:
            if ram[0x001D] == 3:
                pass
            self.env.reset()
        else:
            press_buttons = np.array([self.key_b, 0, 0, 0, self.key_up, self.key_down, self.key_left, self.key_right, self.key_a])
            self.env.step(press_buttons)

        self.frame_timer.end_frame_signal.emit()

        self.main.mario_tile_map.update()
        self.main.mario_key_viewer.update()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return

        key = event.key()
        if key == Qt.Key.Key_Up:
            self.key_up = True
        if key == Qt.Key.Key_Down:
            self.key_down = True
        if key == Qt.Key.Key_Left:
            self.key_left = True
        if key == Qt.Key.Key_Right:
            self.key_right = True
        if key == Qt.Key.Key_A:
            self.key_a = True
        if key == Qt.Key.Key_B:
            self.key_b = True

        if key == Qt.Key.Key_Escape:
            self.main.close_mario()

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return

        key = event.key()
        if key == Qt.Key.Key_Up:
            self.key_up = False
        if key == Qt.Key.Key_Down:
            self.key_down = False
        if key == Qt.Key.Key_Left:
            self.key_left = False
        if key == Qt.Key.Key_Right:
            self.key_right = False
        if key == Qt.Key.Key_A:
            self.key_a = False
        if key == Qt.Key.Key_B:
            self.key_b = False

        if key == Qt.Key.Key_Escape:
            self.main.close_mario()

    def closeEvent(self, event):
        self.main.close_mario()


class MarioTileMap(QWidget):
    def __init__(self, main):
        super().__init__()
        self.setWindowTitle('Tile Map')
        self.main = main

        self.setFixedSize(16 * 20, 13 * 20)
        self.move(560, 100)

        self.show()

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)

        ram = self.main.mario.env.get_ram()

        full_screen_tiles = ram[0x0500:0x069F + 1]
        full_screen_tile_count = full_screen_tiles.shape[0]

        full_screen_page1_tiles = full_screen_tiles[:full_screen_tile_count // 2].reshape((-1, 16))
        full_screen_page2_tiles = full_screen_tiles[full_screen_tile_count // 2:].reshape((-1, 16))

        full_screen_tiles = np.concatenate((full_screen_page1_tiles, full_screen_page2_tiles), axis=1).astype(np.int)

        enemy_drawn = ram[0x000F:0x0014]
        enemy_horizontal_position_in_level = ram[0x006E:0x0072 + 1]
        enemy_x_position_on_screen = ram[0x0087:0x008B + 1]
        enemy_y_position_on_screen = ram[0x00CF:0x00D3 + 1]

        for i in range(5):
            if enemy_drawn[i] == 1:
                ex = (((enemy_horizontal_position_in_level[i] * 256) + enemy_x_position_on_screen[i]) % 512 + 8) // 16
                ey = (enemy_y_position_on_screen[i] - 8) // 16 - 1
                if 0 <= ex < full_screen_tiles.shape[1] and 0 <= ey < full_screen_tiles.shape[0]:
                    full_screen_tiles[ey][ex] = -1

        current_screen_in_level = ram[0x071A]
        screen_x_position_in_level = ram[0x071C]
        screen_x_position_offset = (256 * current_screen_in_level + screen_x_position_in_level) % 512
        sx = screen_x_position_offset // 16

        screen_tiles = np.concatenate((full_screen_tiles, full_screen_tiles), axis=1)[:, sx:sx + 16]

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        for i in range(screen_tiles.shape[0]):
            for j in range(screen_tiles.shape[1]):
                if screen_tiles[i][j] > 0:
                    screen_tiles[i][j] = 1
                if screen_tiles[i][j] == -1:
                    screen_tiles[i][j] = 2
                    painter.setBrush(QBrush(Qt.GlobalColor.red))
                else:
                    painter.setBrush(QBrush(QColor.fromHslF(125 / 239, 0 if screen_tiles[i][j] == 0 else 1, 120 / 240)))
                painter.drawRect(20 * j, 20 * i, 20, 20)

        player_x_position_current_screen_offset = ram[0x03AD]
        player_y_position_current_screen_offset = ram[0x03B8]
        px = (player_x_position_current_screen_offset + 8) // 16
        py = (player_y_position_current_screen_offset + 8) // 16 - 1
        painter.setBrush(QBrush(Qt.GlobalColor.blue))
        painter.drawRect(20 * px, 20 * py, 20, 20)

        painter.end()

    def closeEvent(self, event):
        self.main.close_mario()


class MarioKeyViewer(QWidget):
    def __init__(self, main):
        super().__init__()
        self.setWindowTitle('Key Viewer')
        self.main = main

        self.setFixedSize(320, 180)
        self.move(560, 400)

        self.show()

    def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self)

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_a else Qt.GlobalColor.white))
        painter.drawRect(30, 40, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_a else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(30 + 16, 40 + 24, 'A')

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_b else Qt.GlobalColor.white))
        painter.drawRect(80, 90, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_b else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(80 + 16, 90 + 24, 'B')

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_up else Qt.GlobalColor.white))
        painter.drawRect(200, 40, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_up else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(200 + 14, 40 + 24, '↑')

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_down else Qt.GlobalColor.white))
        painter.drawRect(200, 90, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_down else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(200 + 14, 90 + 24, '↓')

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_left else Qt.GlobalColor.white))
        painter.drawRect(150, 90, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_left else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(150 + 14, 90 + 24, '←')

        painter.setPen(QPen(Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(Qt.GlobalColor.red if self.main.mario.key_right else Qt.GlobalColor.white))
        painter.drawRect(250, 90, 40, 40)
        painter.setPen(QPen(Qt.GlobalColor.white if self.main.mario.key_right else Qt.GlobalColor.black, 2, Qt.PenStyle.SolidLine))
        painter.drawText(250 + 14, 90 + 24, '→')

        painter.end()

    def closeEvent(self, event):
        self.main.close_mario()


class MarioGYM(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mario GYM')

        self.mario = None
        self.mario_analyzer_tile_map = None
        self.mario_analyzer_key_viewer = None

        self.setFixedSize(360, 240)

        mario_button = QPushButton('Super Mario Bros.')
        mario_button.clicked.connect(self.run_mario)
        mario_ai_button = QPushButton('Mario GYM')
        mario_replay_button = QPushButton('Replay')

        self.game_speed_combo_box = QComboBox()
        self.game_speed_combo_box.addItem("보통 속도")
        self.game_speed_combo_box.addItem("빠른 속도")
        self.game_speed_combo_box.addItem("최고 속도")

        vbox_layout = QVBoxLayout()
        vbox_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox_layout.addWidget(mario_button)
        vbox_layout.addWidget(mario_ai_button)
        vbox_layout.addWidget(mario_replay_button)
        vbox_layout.addWidget(self.game_speed_combo_box)

        self.setLayout(vbox_layout)

    def run_mario(self):
        self.mario = Mario(self, self.game_speed_combo_box.currentIndex())
        self.mario.show()

        self.mario_tile_map = MarioTileMap(self)
        self.mario_tile_map.show()

        self.mario_key_viewer = MarioKeyViewer(self)
        self.mario_key_viewer.show()

        self.hide()

    def close_mario(self):
        self.mario.frame_timer.stop_game()
        self.mario.close()
        self.mario_tile_map.close()
        self.mario_key_viewer.close()
        self.show()


def exception_hook(except_type, value, traceback):
    print(except_type, value, traceback)
    exit(1)


def run():
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    w = MarioGYM()
    w.show()
    app.exec()


try:
    if not os.path.exists(os.path.join(retro.data.path(), 'stable', 'SuperMarioBros-Nes', 'rom.nes')):
        rom_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'Super Mario Bros. (World).nes')
        rom_file = open(rom_file_path, "rb")
        data, hash = retro.data.groom_rom(rom_file_path, rom_file)

        known_hashes = retro.data.get_known_hashes()

        if hash in known_hashes:
            game, ext, curpath = known_hashes[hash]
            with open(os.path.join(curpath, game, 'rom%s' % ext), 'wb') as f:
                f.write(data)
except:
    print('failed to import ROM file', file=sys.stderr)


if __name__ == '__main__':
    run()

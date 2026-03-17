#!/usr/bin/env python3

import sys
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

import rospy
import rviz
import serial
import yaml
from std_msgs.msg import String
from PyQt5 import uic
from PyQt5.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPalette, QPen, QPixmap, QTextCursor
from PyQt5.QtWidgets import (QApplication, QDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget)


PORT = "/dev/ttyACM0"
BAUD = 9600
OFFLINE_TIMEOUT_S = 5
STARTUP_GRACE_S = 2

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "Logs"
IMAGE_PATH = SCRIPT_DIR / "flower.png"
LOCALIZATION_CONFIG = (
    SCRIPT_DIR.parent.parent.parent / "ridgeback_desktop" / "ridgeback_viz" / "rviz" / "localization.rviz")
RVIZ_DIALOG_UI = SCRIPT_DIR / "PlantCare_RVizDialog.ui"
RVIZ_BINDINGS = rviz.bindings
GUI_COMMAND_TOPIC = "/plantcare/gui_command"
UR5_CAMERA_FEED_URL = "http://192.168.188.12:5000/video_feed"

PLANT_COORDS = {
    1: (190, 85, 50, 50),
    2: (290, 85, 50, 50),
    3: (390, 85, 50, 50),
    4: (490, 85, 50, 50),
    5: (590, 85, 50, 50),
    6: (690, 85, 50, 50),
    7: (790, 85, 50, 50),
    8: (890, 85, 50, 50),
    9: (235, 135, 50, 50),
    10: (335, 135, 50, 50),
    11: (435, 135, 50, 50),
    12: (535, 135, 50, 50),
    13: (635, 135, 50, 50),
    14: (735, 135, 50, 50),
    15: (835, 135, 50, 50),
    16: (935, 135, 50, 50),
    17: (190, 285, 50, 50),
    18: (290, 285, 50, 50),
    19: (390, 285, 50, 50),
    20: (490, 285, 50, 50),
    21: (590, 285, 50, 50),
    22: (690, 285, 50, 50),
    23: (790, 285, 50, 50),
    24: (890, 285, 50, 50),
    25: (240, 335, 50, 50),
    26: (340, 335, 50, 50),
    27: (440, 335, 50, 50),
    28: (540, 335, 50, 50),
    29: (640, 335, 50, 50),
    30: (740, 335, 50, 50),
    31: (840, 335, 50, 50),
    32: (940, 335, 50, 50),
    33: (195, 485, 50, 50),
    34: (295, 485, 50, 50),
    35: (395, 485, 50, 50),
    36: (495, 485, 50, 50),
    37: (595, 485, 50, 50),
    38: (695, 485, 50, 50),
    39: (795, 485, 50, 50),
    40: (895, 485, 50, 50),
    41: (240, 535, 50, 50),
    42: (340, 535, 50, 50),
    43: (440, 535, 50, 50),
    44: (540, 535, 50, 50),
    45: (640, 535, 50, 50),
    46: (740, 535, 50, 50),
    47: (840, 535, 50, 50),
    48: (940, 535, 50, 50),
}

TABLE_RECTS = [
    (175, 125, 800, 25),
    (175, 325, 800, 25),
    (175, 525, 800, 25),
]


class QueueTextEdit(QTextEdit):
    lineClicked = pyqtSignal(str)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.LineUnderCursor)
        self.lineClicked.emit(cursor.selectedText().strip())


class PlantMapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setGeometry(25, 140, 1025, 650)
        self.robot_rect = (50, 510, 95, 95)
        self.plant_status = {plant_id: -1 for plant_id in PLANT_COORDS}

    def set_plant_status(self, plant_id, status):
        if plant_id in self.plant_status:
            self.plant_status[plant_id] = status
            self.update()

    def _status_colour(self, status):
        if status == -1:
            return QColor("grey")
        if status <= 40:
            return QColor("yellow")
        if 40 <= status <= 60:
            return QColor("green")
        if status > 60:
            return QColor("red")
        if status == 1:
            return QColor("yellow")
        if status == 2:
            return QColor("green")
        if status == 3:
            return QColor("red")
        return QColor("grey")

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("lightblue"))

        table_pen = QPen(Qt.black, 3)
        painter.setPen(table_pen)
        painter.setBrush(QColor("white"))
        for x, y, w, h in TABLE_RECTS:
            painter.drawRect(x, y, w, h)

        plant_pen = QPen(Qt.black, 3)
        painter.setPen(plant_pen)
        for plant_id, (x, y, w, h) in PLANT_COORDS.items():
            painter.setBrush(self._status_colour(self.plant_status[plant_id]))
            painter.drawEllipse(x, y, w, h)

        rx, ry, rw, rh = self.robot_rect
        painter.setPen(QPen(Qt.black, 3))
        painter.setBrush(QColor("black"))
        painter.drawRect(rx, ry, rw, rh)
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 16, QFont.Bold))
        painter.drawText(rx, ry, rw, rh, Qt.AlignCenter, "Robot")


class CameraDialog(QDialog):
    class _MjpegStreamThread(QThread):
        frame_received = pyqtSignal(QImage)
        error_received = pyqtSignal(str)

        def __init__(self, url, parent=None):
            super().__init__(parent)
            self.url = url
            self._running = True
            self._response = None

        def run(self):
            buffer = b""
            try:
                self._response = urllib.request.urlopen(self.url, timeout=5)
                while self._running:
                    chunk = self._response.read(4096)
                    if not chunk:
                        break
                    buffer += chunk

                    while True:
                        start = buffer.find(b"\xff\xd8")
                        end = buffer.find(b"\xff\xd9", start + 2)
                        if start == -1 or end == -1:
                            if len(buffer) > 1024 * 1024:
                                buffer = buffer[-65536:]
                            break

                        frame_bytes = buffer[start:end + 2]
                        buffer = buffer[end + 2:]
                        image = QImage.fromData(frame_bytes)
                        if not image.isNull():
                            self.frame_received.emit(image)
            except Exception as exc:
                if self._running:
                    self.error_received.emit(str(exc))
            finally:
                if self._response is not None:
                    try:
                        self._response.close()
                    except Exception:
                        pass

        def stop(self):
            self._running = False
            if self._response is not None:
                try:
                    self._response.close()
                except Exception:
                    pass

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("UR5 Camera")
        self.setFixedSize(600, 600)
        self._latest_frame = None

        title = QLabel("UR5 Camera", self)
        title.setFont(QFont("Arial", 16))
        title.setGeometry(150, 0, 200, 40)
        title.setAlignment(Qt.AlignCenter)

        close_button = QPushButton("Exit", self)
        close_button.setGeometry(375, 0, 100, 35)
        close_button.clicked.connect(self.close)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 500, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid black;")
        self.image_label.setText("Connecting to camera feed...")

        self.status_label = QLabel(self)
        self.status_label.setGeometry(50, 465, 500, 30)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setText(UR5_CAMERA_FEED_URL)

        self.stream_thread = self._MjpegStreamThread(UR5_CAMERA_FEED_URL, self)
        self.stream_thread.frame_received.connect(self._update_frame)
        self.stream_thread.error_received.connect(self._handle_stream_error)
        self.stream_thread.start()

    def _update_frame(self, image):
        self._latest_frame = image
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.status_label.setText("Live feed connected")

    def _handle_stream_error(self, message):
        self.image_label.setText("Unable to load camera feed")
        self.status_label.setText(f"Camera error: {message}")

    def closeEvent(self, event):
        if hasattr(self, "stream_thread") and self.stream_thread is not None:
            self.stream_thread.stop()
            self.stream_thread.wait(1000)
        super().closeEvent(event)


class ActionLogDialog(QDialog):
    def __init__(self, action_history, generate_report, parent=None):
        super().__init__(parent)
        self.action_history = action_history
        self.generate_report = generate_report
        self.setWindowTitle("Action Log")
        self.setFixedSize(800, 800)

        log_label = QLabel("Action Log", self)
        log_label.setFont(QFont("Arial", 16))
        log_label.setGeometry(300, 50, 200, 40)
        log_label.setAlignment(Qt.AlignCenter)

        self.text_area = QTextEdit(self)
        self.text_area.setGeometry(150, 75, 500, 560)
        self.text_area.setReadOnly(True)
        self.text_area.setFont(QFont("Times New Roman", 15))
        self.refresh()

        exit_button = QPushButton("Exit", self)
        exit_button.setGeometry(100, 700, 160, 80)
        exit_button.setFont(QFont("Arial", 16))
        exit_button.setStyleSheet("background-color: lightgray;")
        exit_button.clicked.connect(self.close)

        report_button = QPushButton("Generate Report", self)
        report_button.setGeometry(525, 700, 180, 80)
        report_button.setFont(QFont("Arial", 16))
        report_button.setStyleSheet("background-color: lightgray;")
        report_button.clicked.connect(self.generate_report)

    def refresh(self):
        self.text_area.setPlainText("\n".join(self.action_history))


class RVizEmbedDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi(str(RVIZ_DIALOG_UI), self)
        self.host_layout = QVBoxLayout(self.rviz_host_widget)
        self.host_layout.setContentsMargins(0, 0, 0, 0)
        self.reload_view_button.clicked.connect(self.reopen_view)
        self.rviz_frame = None
        self.manager = None
        self._create_rviz_frame()
        self.reload_view(rebuild_frame=False)

    def _set_display_property(self, display, property_name, value):
        prop = display.subProp(property_name)
        if prop is not None:
            prop.setValue(value)

    def _create_display(self, class_name, display_name, enabled=True):
        display = self.manager.createDisplay(class_name, display_name, enabled)
        if display is None:
            raise RuntimeError(f"Failed to create RViz display {class_name}")
        return display

    def _build_embedded_config_path(self):
        with LOCALIZATION_CONFIG.open("r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file)

        sanitized_config = {
            "Visualization Manager": config_data.get("Visualization Manager", {}),
        }

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="plantcare_localization_",
            suffix=".rviz",
            delete=False,
            encoding="utf-8",
        )
        yaml.safe_dump(sanitized_config, temp_file, sort_keys=False)
        temp_file.close()
        return temp_file.name

    def _apply_rviz_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(232, 232, 232))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Highlight, QColor(76, 132, 255))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.rviz_frame.setPalette(palette)
        self.rviz_frame.setAutoFillBackground(True)
        self.rviz_frame.setStyleSheet(
            """
            QWidget {
                color: #111111;
                background-color: #f0f0f0;
            }
            QTreeView, QListView, QTextEdit, QLineEdit, QComboBox, QAbstractSpinBox {
                color: #111111;
                background-color: #ffffff;
                selection-background-color: #4c84ff;
                selection-color: #ffffff;
            }
            QToolBar, QMenuBar, QStatusBar {
                color: #111111;
                background-color: #e8e8e8;
            }
            QPushButton, QToolButton, QCheckBox, QRadioButton, QLabel {
                color: #111111;
            }
            """
        )

    def _create_rviz_frame(self):
        if self.rviz_frame is not None:
            self.host_layout.removeWidget(self.rviz_frame)
            self.rviz_frame.setParent(None)
            self.rviz_frame.deleteLater()

        self.rviz_frame = RVIZ_BINDINGS.VisualizationFrame()
        self.rviz_frame.setParent(self.rviz_host_widget)
        self.rviz_frame.setWindowFlags(Qt.Widget)
        self.rviz_frame.setSplashPath("")
        self.rviz_frame.initialize()
        self.rviz_frame.setMenuBar(None)
        self.rviz_frame.setStatusBar(None)
        self.rviz_frame.setHideButtonVisibility(False)
        self.host_layout.addWidget(self.rviz_frame)
        self._apply_rviz_theme()
        self.rviz_frame.show()
        self.manager = self.rviz_frame.getManager()

    def reload_view(self, rebuild_frame=True):
        try:
            rospy.get_master().getPid()
        except Exception:
            self.status_label.setText("Status: ROS master unavailable. Start roscore/roslaunch first.")
            return

        if not LOCALIZATION_CONFIG.exists():
            self.status_label.setText("Status: localization.rviz not found")
            return

        self.status_label.setText("Status: Reloading view")
        if rebuild_frame:
            self._create_rviz_frame()

        embedded_config_path = self._build_embedded_config_path()
        self.rviz_frame.loadDisplayConfig(embedded_config_path)
        self.manager = self.rviz_frame.getManager()
        self.manager.setFixedFrame("map")

        view_manager = self.manager.getViewManager()
        view_manager.setCurrentViewControllerType("rviz/TopDownOrtho")
        current_view = view_manager.getCurrent()
        if current_view is not None:
            for property_name, value in [
                ("Target Frame", "map"),
                ("Scale", 30.0),
                ("Angle", 1.5708),
                ("X", 0.0),
                ("Y", 0.0),
            ]:
                try:
                    current_view.subProp(property_name).setValue(value)
                except Exception:
                    pass

        self.manager.startUpdate()
        self.manager.queueRender()
        self._apply_rviz_theme()
        self.status_label.setText(f"Status: Loaded {LOCALIZATION_CONFIG.name}")

    def reopen_view(self):
        self.status_label.setText("Status: Reopening view")
        parent = self.parent()
        self.close()
        if parent is not None and hasattr(parent, "rviz_dialog"):
            parent.rviz_dialog = None
            QTimer.singleShot(0, parent.show_rviz_embed)


class PlantCareGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PlantCare_GUI")
        self.setFixedSize(1600, 1000)

        self.ser = None
        self.selected_plant = None
        self.action_history = []
        self.last_status = {}
        self.last_seen = {}
        self.offline_plants = set()
        self.queue_entries = {}
        self.start_time = datetime.now()
        self.log_dialog = None
        self.rviz_dialog = None
        self.gui_command_publisher = rospy.Publisher(
            GUI_COMMAND_TOPIC,
            String,
            queue_size=10,
        )

        self._build_ui()
        self._connect_serial()
        self._start_timer()

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)

        self.stop_button = self._make_button(
            central,
            "Emergency Stop",
            (25, 0, 220, 120),
            self.stop_operation,
            "lightgray",
        )
        self.stop_button.setStyleSheet(
            "background-color: lightgray; color: black; border: 2px solid black;"
        )

        self.battery_label = self._make_status_label(
            central, "Battery: 50%", "yellow", (300, 0, 220, 120)
        )
        self.fill_level_label = self._make_status_label(
            central, "Tank: Full", "lightblue", (550, 0, 220, 120)
        )
        self.status_label = self._make_status_label(
            central, "Status: Idle", "orange", (800, 0, 220, 120)
        )

        self.ur5_button = self._make_button(
            central, "UR5_Camera", (1050, 0, 190, 120), self.show_camera, "lightgray"
        )
        self.action_log_button = self._make_button(
            central, "Action Log", (1275, 0, 190, 120), self.show_log, "lightgray"
        )
        self.rviz_button = self._make_button(
            central, "View map", (1180, 780, 180, 90), self.show_rviz_embed, "lightgray"
        )

        self.plant_map = PlantMapWidget(central)

        self.queue_text = QueueTextEdit(central)
        self.queue_text.setGeometry(1075, 140, 440, 500)
        self.queue_text.setReadOnly(True)
        self.queue_text.setFont(QFont("Times New Roman", 15))
        self.queue_text.lineClicked.connect(self.on_queue_click)

        self.send_button = QPushButton("Send to Ridgeback", central)
        self.send_button.setGeometry(1080, 660, 180, 90)
        self.send_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.send_button.setStyleSheet("background-color: lightgreen; border: 3px solid black;")
        self.send_button.clicked.connect(self.send_to_ridgeback)

        self.reject_button = QPushButton("Reject Plant", central)
        self.reject_button.setGeometry(1280, 660, 180, 90)
        self.reject_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.reject_button.setStyleSheet("background-color: lightcoral; border: 3px solid black;")
        self.reject_button.clicked.connect(self.reject_plant)

        self.refresh_queue_text()

    def _make_button(self, parent, text, geometry, handler, bg):
        button = QPushButton(text, parent)
        button.setGeometry(*geometry)
        button.setFont(QFont("Arial", 14, QFont.Bold))
        button.setStyleSheet(f"background-color: {bg}; color: black; border: 2px solid grey;")
        button.clicked.connect(handler)
        return button

    def _make_status_label(self, parent, text, bg, geometry):
        label = QLabel(text, parent)
        label.setGeometry(*geometry)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 14, QFont.Bold))
        label.setStyleSheet(f"background-color: {bg}; color: black; border: 3px solid gray;")
        return label

    def _connect_serial(self):
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0.1)
            self.set_status("Status: Connected")
        except Exception:
            self.ser = None
            self.set_status("Status: No Serial")

    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(50)

    def set_status(self, text):
        self.status_label.setText(text)

    def show_camera(self):
        dialog = CameraDialog(self)
        dialog.exec_()

    def show_log(self):
        self.log_dialog = ActionLogDialog(self.action_history, self.generate_report, self)
        self.log_dialog.exec_()

    def show_rviz_embed(self):
        if self.rviz_dialog is None:
            self.rviz_dialog = RVizEmbedDialog(self)
        self.rviz_dialog.show()
        self.rviz_dialog.raise_()
        self.rviz_dialog.activateWindow()

    def generate_report(self):
        LOG_DIR.mkdir(exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        report_path = LOG_DIR / filename

        with report_path.open("w", encoding="utf-8") as report_file:
            report_file.write("Time,Event\n")
            for entry in self.action_history:
                time_part, event_part = entry.split("]", 1)
                report_file.write(f"{time_part.replace('[', '')},{event_part.strip()}\n")

        QMessageBox.information(self, "Report Generated", f"Saved {report_path.name}")

    def log_event(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.action_history.append(f"[{timestamp}] {message}")
        if self.log_dialog is not None:
            self.log_dialog.refresh()

    def refresh_queue_text(self):
        self.queue_text.clear()
        self.queue_text.setAlignment(Qt.AlignLeft)
        cursor = self.queue_text.textCursor()
        cursor.movePosition(QTextCursor.Start)
        block_format = cursor.blockFormat()
        block_format.setAlignment(Qt.AlignCenter)
        cursor.setBlockFormat(block_format)
        char_format = cursor.charFormat()
        char_format.setFontUnderline(True)
        char_format.setFont(QFont("Times New Roman", 15))
        cursor.setCharFormat(char_format)
        cursor.insertText("Job Queue:")
        cursor.insertBlock()

        block_format.setAlignment(Qt.AlignLeft)
        cursor.setBlockFormat(block_format)
        char_format.setFontUnderline(False)
        cursor.setCharFormat(char_format)

        for plant_id in sorted(self.queue_entries):
            cursor.insertText(self.queue_entries[plant_id])
            cursor.insertBlock()

        self.queue_text.setTextCursor(cursor)
        self.queue_text.moveCursor(QTextCursor.Start)

    def on_queue_click(self, line):
        if line.startswith("Plant "):
            try:
                self.selected_plant = int(line.split()[1])
                self.set_status(f"Status: Selected Plant {self.selected_plant}")
            except ValueError:
                self.selected_plant = None

    def delete_plant_line(self, plant_id):
        if plant_id in self.queue_entries:
            del self.queue_entries[plant_id]
            self.refresh_queue_text()

    def set_plant(self, plant_id, status):
        self.plant_map.set_plant_status(plant_id, status)

    def remove_from_queue_if_present(self, plant_id):
        self.delete_plant_line(plant_id)
        if self.selected_plant == plant_id:
            self.selected_plant = None
            self.set_status("Status: Selected plant went offline")

    def stop_operation(self):
        self.log_event("User stopped operation")
        self.gui_command_publisher.publish("emergency_stop")
        if self.ser:
            self.ser.write(b"Stop\n")
            self.ser.flush()
        self.set_status("Status: Emergency stop sent")

    def send_to_ridgeback(self):
        if self.selected_plant is None:
            self.set_status("Status: Click a plant in the job queue first")
            return

        if self.ser:
            self.ser.write(f"RIDGEBACK,{self.selected_plant}\n".encode())
            self.ser.flush()

        sent_plant = self.selected_plant
        self.delete_plant_line(sent_plant)
        self.set_status(f"Status: Sent Plant {sent_plant} to Ridgeback")
        self.log_event(f"User decided to water Plant {sent_plant}")
        self.selected_plant = None

    def reject_plant(self):
        if self.selected_plant is None:
            self.set_status("Status: Click a plant in the job queue first")
            return

        rejected_plant = self.selected_plant
        self.delete_plant_line(rejected_plant)
        self.set_status(f"Status: Rejected Plant {rejected_plant}")
        self.set_plant(rejected_plant, 2)
        self.last_status[rejected_plant] = 2
        self.log_event(f"User decided to reject Plant {rejected_plant}")
        self.selected_plant = None

    def poll_serial(self):
        now = datetime.now()

        for plant_id in PLANT_COORDS:
            seen = self.last_seen.get(plant_id)
            if seen is None:
                if (now - self.start_time).total_seconds() > STARTUP_GRACE_S:
                    if plant_id not in self.offline_plants:
                        self.remove_from_queue_if_present(plant_id)
                        self.offline_plants.add(plant_id)
                        self.set_plant(plant_id, -1)
                        self.log_event(f"Plant Node {plant_id} is offline")
                continue

            if (now - seen).total_seconds() > OFFLINE_TIMEOUT_S:
                if plant_id not in self.offline_plants:
                    self.remove_from_queue_if_present(plant_id)
                    self.offline_plants.add(plant_id)
                    self.set_plant(plant_id, -1)
                    self.log_event(f"Plant {plant_id} offline (no data for {OFFLINE_TIMEOUT_S}s)")

        if self.ser is None or not self.ser.in_waiting:
            return

        try:
            line = self.ser.readline().decode(errors="ignore").strip()
        except Exception:
            return

        if "," not in line:
            return

        a, b = line.split(",", 1)
        try:
            plant_id = int(a)
            status = int(b)
        except ValueError:
            return

        self.last_seen[plant_id] = datetime.now()

        if plant_id in self.offline_plants:
            self.offline_plants.remove(plant_id)
            self.log_event(f"Plant {plant_id} back online")

        self.set_plant(plant_id, status)

        if self.last_status.get(plant_id) == status:
            return

        self.last_status[plant_id] = status
        self.delete_plant_line(plant_id)

        if status == 1:
            self.queue_entries[plant_id] = f"Plant {plant_id} is UNDERWATERED"
            self.log_event(f"Plant {plant_id} reported UNDERWATERED")
        elif status == 3:
            self.queue_entries[plant_id] = f"Plant {plant_id} is OVERWATERED, intervention needed"
            self.log_event(f"Plant {plant_id} reported OVERWATERED")
        else:
            self.log_event(f"Plant {plant_id} reported WATERED")

        self.refresh_queue_text()

    def closeEvent(self, event):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    if not rospy.core.is_initialized():
        rospy.init_node("plantcare_gui", anonymous=True, disable_signals=True)

    app = QApplication(sys.argv)
    window = PlantCareGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

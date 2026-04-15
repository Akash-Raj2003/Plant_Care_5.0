#!/usr/bin/env python3

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import rospy
import serial
from std_msgs.msg import Int32, String
from my_simulations.msg import dispensingActionFeedback
from plant_msgs.msg import PlantMoisture
import yaml
from PyQt5 import uic
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette, QTextCursor
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from PlantCare_GUI import (
    ActionLogDialog,
    CameraDialog,
    LOG_DIR,
    LOCALIZATION_CONFIG,
    OFFLINE_TIMEOUT_S,
    PLANT_COORDS,
    PORT,
    BAUD,
    QueueTextEdit,
    PlantMapWidget,
    RVIZ_BINDINGS,
    STARTUP_GRACE_S,
)


SCRIPT_DIR = Path(__file__).resolve().parent
UI_PATH = SCRIPT_DIR / "PlantCare_Studio.ui"
RIDGEBACK_STATE_TOPIC = "/ridgeback/state"
DISPENSE_FEEDBACK_TOPIC = "/dispense_water/feedback"
PLANT_MOISTURE_TOPIC = "/plant/moisture_alert"
GUI_COMMAND_TOPIC = "/plantcare/gui_command"
GUI_EMERGENCY_STOP = "emergency_stop"
GUI_RESUME_DISPATCH = "resume_dispatch"
RIDGEBACK_STATES = {
    0: "available",
    1: "moving_to_plant",
    2: "waiting_for_dispense",
    3: "dispensing",
    4: "complete",
    5: "interrupted",
}


class EmbeddedMainRVizWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.rviz_frame = None
        self.manager = None
        self._create_rviz_frame()
        self.load_view()

    def _build_embedded_config_path(self):
        with LOCALIZATION_CONFIG.open("r", encoding="utf-8") as config_file:
            config_data = yaml.safe_load(config_file)

        sanitized_config = {
            "Visualization Manager": config_data.get("Visualization Manager", {}),
        }

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="plantcare_main_localization_",
            suffix=".rviz",
            delete=False,
            encoding="utf-8",
        )
        yaml.safe_dump(sanitized_config, temp_file, sort_keys=False)
        temp_file.close()
        return temp_file.name

    def _apply_theme(self):
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
        self.rviz_frame = RVIZ_BINDINGS.VisualizationFrame()
        self.rviz_frame.setParent(self)
        self.rviz_frame.setWindowFlags(Qt.Widget)
        self.rviz_frame.setSplashPath("")
        self.rviz_frame.initialize()
        self.rviz_frame.setMenuBar(None)
        self.rviz_frame.setStatusBar(None)
        self.rviz_frame.setHideButtonVisibility(False)
        self.layout.addWidget(self.rviz_frame)
        self._apply_theme()
        self.rviz_frame.show()
        self.manager = self.rviz_frame.getManager()

    def load_view(self):
        if not LOCALIZATION_CONFIG.exists():
            return

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
        self._apply_theme()


class PlantCareStudio(QMainWindow):
    plant_moisture_received = pyqtSignal(int, bool, float)

    def __init__(self):
        super().__init__()
        uic.loadUi(str(UI_PATH), self)

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
        self.rviz_panel = None
        self.gui_command_publisher = None
        self.dispatch_interrupted = False
        self.plant_moisture_received.connect(self._handle_plant_moisture_update)

        self._replace_placeholders()
        self._wire_actions()
        self._connect_serial()
        self._connect_ros_topics()
        self._start_timer()
        self.refresh_queue_text()
        self._set_ridgeback_status(None)
        self._set_dispense_progress(None, None)
        self.dispatch_warning_label.hide()

    def _replace_widget(self, old_widget, new_widget):
        new_widget.setParent(old_widget.parent())
        new_widget.setGeometry(old_widget.geometry())
        new_widget.setObjectName(old_widget.objectName())
        new_widget.show()
        old_widget.deleteLater()
        return new_widget

    def _replace_placeholders(self):
        stack = QStackedWidget(self.centralwidget)
        self.plant_map_placeholder = self._replace_widget(
            self.plant_map_placeholder,
            stack,
        )
        self.main_stack = self.plant_map_placeholder
        self.plant_map = PlantMapWidget(self.main_stack)
        self.main_stack.addWidget(self.plant_map)

        queue_widget = QueueTextEdit(self.centralwidget)
        queue_widget.setReadOnly(True)
        queue_widget.setFont(QFont("Times New Roman", 15))
        self.queue_text_placeholder = self._replace_widget(
            self.queue_text_placeholder,
            queue_widget,
        )
        self.queue_text = self.queue_text_placeholder

    def _wire_actions(self):
        self.stop_button.clicked.connect(self.stop_operation)
        self.ur5_button.clicked.connect(self.show_camera)
        self.action_log_button.clicked.connect(self.show_log)
        self.send_button.clicked.connect(self.send_to_ridgeback)
        self.reject_button.clicked.connect(self.reject_plant)
        self.view_map_button.clicked.connect(self.toggle_map_view)
        self.resume_dispatch_button.clicked.connect(self.resume_dispatch)
        self.queue_text.lineClicked.connect(self.on_queue_click)

    def _connect_serial(self):
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0.1)
            self.set_status("Status: Connected")
        except Exception:
            self.ser = None
            self.set_status("Status: No Serial")

    def _connect_ros_topics(self):
        self.gui_command_publisher = rospy.Publisher(
            GUI_COMMAND_TOPIC,
            String,
            queue_size=10,
        )
        self.ridgeback_state_subscriber = rospy.Subscriber(
            RIDGEBACK_STATE_TOPIC,
            Int32,
            self._ridgeback_state_callback,
        )
        self.dispense_feedback_subscriber = rospy.Subscriber(
            DISPENSE_FEEDBACK_TOPIC,
            dispensingActionFeedback,
            self._dispense_feedback_callback,
        )
        self.plant_moisture_subscriber = rospy.Subscriber(
            PLANT_MOISTURE_TOPIC,
            PlantMoisture,
            self._plant_moisture_callback,
        )

    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(50)

    def set_status(self, text):
        self.status_label.setText(text)

    def _set_ridgeback_status(self, state_value):
        if state_value is None:
            self.ridgeback_status_label.setText("Ridgeback status:")
            return

        state_name = RIDGEBACK_STATES.get(state_value, f"unknown ({state_value})")
        self.ridgeback_status_label.setText(
            f"Ridgeback status: {state_name.replace('_', ' ')} ({state_value})"
        )
        if state_value == 5:
            self.dispatch_interrupted = True
            self.dispatch_warning_label.show()
        elif state_value in (0, 1, 2, 3, 4) and not self.dispatch_interrupted:
            self.dispatch_warning_label.hide()

    def _ridgeback_state_callback(self, msg):
        self._set_ridgeback_status(msg.data)

    def _set_dispense_progress(self, percent_complete, state_name):
        if percent_complete is None or state_name is None:
            self.dispense_progress_label.setText("Pump inactive")
            return

        self.dispense_progress_label.setText(
            f"Dispensing progress: {percent_complete:.1f}% ({state_name})"
        )

    def _dispense_feedback_callback(self, msg):
        self._set_dispense_progress(
            msg.feedback.percent_complete,
            msg.feedback.state,
        )

    def _plant_moisture_callback(self, msg):
        self.plant_moisture_received.emit(
            msg.plant_id,
            msg.low_moisture,
            msg.moisture_level,
        )

    def _handle_plant_moisture_update(self, plant_id, low_moisture, moisture_level):
        self.last_seen[plant_id] = datetime.now()

        if plant_id in self.offline_plants:
            self.offline_plants.remove(plant_id)
            self.log_event(f"Plant {plant_id} back online")

        self.set_plant(plant_id, moisture_level)

        if self.last_status.get(plant_id) == moisture_level:
            return

        self.last_status[plant_id] = moisture_level
        self.delete_plant_line(plant_id)

        if moisture_level <= 40:
            self.queue_entries[plant_id] = (
                f"Plant {plant_id} is UNDERWATERED (moisture: {moisture_level:.1f})"
            )
            self.log_event(
                f"Plant {plant_id} reported UNDERWATERED from bridge (moisture {moisture_level:.1f})"
            )
        elif moisture_level > 60:
            self.queue_entries[plant_id] = (
                f"Plant {plant_id} is OVERWATERED (moisture: {moisture_level:.1f})"
            )
            self.log_event(
                f"Plant {plant_id} reported OVERWATERED from bridge (moisture {moisture_level:.1f})"
            )
        else:
            self.log_event(
                f"Plant {plant_id} reported WATERED from bridge (moisture {moisture_level:.1f})"
            )

        self.refresh_queue_text()

    def show_camera(self):
        dialog = CameraDialog(self)
        dialog.exec_()

    def show_log(self):
        self.log_dialog = ActionLogDialog(self.action_history, self.generate_report, self)
        self.log_dialog.exec_()

    def toggle_map_view(self):
        if self.rviz_panel is None:
            self.rviz_panel = EmbeddedMainRVizWidget(self.main_stack)
            self.main_stack.addWidget(self.rviz_panel)

        if self.main_stack.currentWidget() is self.plant_map:
            self.main_stack.setCurrentWidget(self.rviz_panel)
            self.view_map_button.setText("View plants")
        else:
            self.main_stack.setCurrentWidget(self.plant_map)
            self.view_map_button.setText("View map")

    def generate_report(self):
        LOG_DIR.mkdir(exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")
        report_path = LOG_DIR / filename
        with report_path.open("w", encoding="utf-8") as report_file:
            report_file.write("Time,Event\n")
            for entry in self.action_history:
                time_part, event_part = entry.split("]", 1)
                report_file.write(f"{time_part.replace('[', '')},{event_part.strip()}\n")

    def log_event(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.action_history.append(f"[{timestamp}] {message}")
        if self.log_dialog is not None:
            self.log_dialog.refresh()

    def refresh_queue_text(self):
        self.queue_text.clear()
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
        if self.gui_command_publisher is not None:
            self.gui_command_publisher.publish(GUI_EMERGENCY_STOP)
        if self.ser:
            self.ser.write(b"Stop\n")
            self.ser.flush()
        self.set_status("Status: Emergency stop sent")
        self.dispatch_interrupted = True
        self.dispatch_warning_label.show()

    def resume_dispatch(self):
        if self.gui_command_publisher is not None:
            self.gui_command_publisher.publish(GUI_RESUME_DISPATCH)
        self.set_status("Status: Resume dispatch sent")
        self.dispatch_interrupted = False
        self.dispatch_warning_label.hide()

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
        rospy.init_node("plantcare_studio", anonymous=True, disable_signals=True)

    app = QApplication(sys.argv)
    window = PlantCareStudio()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

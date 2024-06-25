import sys
import cv2
import time
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.new_gui import Ui_MainWindow
from sqlalchemy import create_engine, text
from datetime import datetime
import logging
import queue
from onnxmodules.yolov8onnx import YOLOv8

# Logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("yeni.log"),
                        logging.StreamHandler()
                    ])

class LatestFrame:
    def __init__(self):
        self.frame = queue.Queue(maxsize=1)
        self.frame_id = queue.Queue(maxsize=1)

    def clear_buffer(self):
        with self.frame.mutex, self.frame_id.mutex:
            self.frame.queue.clear()
            self.frame_id.queue.clear()

    def put(self, frame, frame_id, realtime=False):
        try:
            if self.frame.full() and realtime is True:
                self.clear_buffer()
            self.frame.put(frame, block=True, timeout=None)
            self.frame_id.put(frame_id, block=True, timeout=None)
        except queue.Full:
            logging.error("Error LatestFrame.put is error")

    def get(self):
        try:
            frame_tmp = self.frame.get(block=True, timeout=None)
            id_tmp = self.frame_id.get(block=True, timeout=None)
            return id_tmp, frame_tmp
        except queue.Empty:
            logging.error("Error LatestFrame get function.")


class DatabaseThread(QtCore.QThread):
    def __init__(self, detected_dict, db_manager):
        super().__init__()
        self.detected_dict = detected_dict
        self.db_manager = db_manager
        self.finished.connect(self.log_thread_finished)

    def run(self):
        try:
            self.update_detection_in_database(self.detected_dict)
        except Exception as e:
            logging.error(f"Error in DatabaseThread: {e}")

    def update_detection_in_database(self, detected_dict):
        if not self.db_manager.connected:
            self.db_manager.connect()
        with self.db_manager.engine.connect() as connection:
            transaction = connection.begin()
            try:
                params = {
                    'tespit_zamani_yeni': detected_dict['Tespit Edilme Saati'].strftime("%Y-%m-%d %H:%M:%S"),
                    'tespit_durumu_yeni': detected_dict['Tespit Durumu'],
                    'veri_toplama_id_yeni': detected_dict['ID']
                }

                query = text("""
                    UPDATE [AIDATA].[dbo].[veri_toplama_yeni]
                    SET [tespit_zamani_yeni] = :tespit_zamani_yeni,
                        [tespit_durumu_yeni] = :tespit_durumu_yeni
                    WHERE [veri_toplama_id_yeni] = :veri_toplama_id_yeni
                """)

                connection.execute(query, params)
                transaction.commit()
            except Exception as e:
                transaction.rollback()
                logging.error(f"Error writing to database: {e}")
            finally:
                connection.close()

    def log_thread_finished(self):
        logging.info("DatabaseThread successfully finished.")

class DatabaseManager():
    def __init__(self):
        super().__init__()
        self.engine = None
        self.connected = False

    def connect(self):
        try:
            if not self.connected:
                server = '10.161.112.70'
                database = 'AIDATA'
                username = 'majorskt'
                password = 'gargara'
                driver = 'ODBC Driver 17 for SQL Server'
                connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
                self.engine = create_engine(
                    connection_string,
                    pool_size=20,  # Varsayılan değeri artırın
                    max_overflow=40,  # Varsayılan değeri artırın
                    pool_timeout=30,  # Zaman aşımı süresini gerektiğinde artırın
                    pool_recycle=1800  # Bağlantıların geri dönüşüm süresini ayarlayın
                )
                self.connected = True
                logging.info("Database connected")
                logging.info(f"Active connections: {self.engine.pool.status()}")
        except Exception as e:
            logging.error(f"Error in DatabaseManager.connect: {e}")

    def disconnect(self):
        try:
            if self.connected:
                self.engine.dispose()
                self.connected = False
                logging.info("Database disconnected")
                logging.info(f"Active connections: {self.engine.pool.status()}")
        except Exception as e:
            logging.error(f"Error in DatabaseManager.disconnect: {e}")

class YOLOv8Model:
    def __init__(self, model_path):
        try:
            self.model = YOLOv8(model_path)
            logging.info(f"Model initialized: {model_path}")
        except Exception as e:
            logging.error(f"Error in YOLOv8Model.__init__: {e}")

    def detect(self, image, iou_threshold=None, conf_threshold=None):
        try:
            if iou_threshold is not None:
                self.model.iou_threshold = iou_threshold
            if conf_threshold is not None:
                self.model.conf_threshold = conf_threshold

            boxes, scores, class_ids, inference_time = self.model.detect_objects(image)
            assert len(boxes) == len(scores) == len(class_ids), "Output dimensions mismatch"
            bboxes = [(int(box[0]), int(box[1]), int(box[2]), int(box[3])) for box in boxes]
            tespit_durumu = 1 if len(bboxes) > 0 else 0
            tespit_zamani = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return bboxes, inference_time, tespit_durumu, tespit_zamani, len(bboxes)
        except AssertionError as e:
            logging.error(f"AssertionError in YOLOv8Model.detect: {e}")
        except Exception as e:
            logging.error(f"Error in YOLOv8Model.detect: {e}")

    def draw_boxes(self, image, bboxes):
        try:
            return self.model.draw_detections(image)
        except Exception as e:
            logging.error(f"Error in YOLOv8Model.draw_boxes: {e}")
            return image

    def set_iou_threshold(self, value):
        try:
            self.model.iou_threshold = value
            logging.info(f"IOU threshold set to: {value}")
        except Exception as e:
            logging.error(f"Error in YOLOv8Model.set_iou_threshold: {e}")

    def set_conf_threshold(self, value):
        try:
            self.model.conf_threshold = value
            logging.info(f"Confidence threshold set to: {value}")
        except Exception as e:
            logging.error(f"Error in YOLOv8Model.set_conf_threshold: {e}")

class CameraThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(QtGui.QImage)
    update_fps_signal = QtCore.pyqtSignal(float)
    update_person_count_signal = QtCore.pyqtSignal(int)

    def __init__(self, source, yolo_model, latest_frame, db_manager):
        super().__init__()
        self.source = source
        self.running = False
        self.cap = None
        self.yolo_model = yolo_model
        self.latest_frame = latest_frame
        self.db_manager = db_manager
        self.frame_id = 0
        self.last_db_write_time = time.time()
        self.last_detection_time = time.time()
        self.tespit_durumu = 0
        self.tespit_durumu_yeni = -1  # Initial value to detect the first change
        self.finished.connect(self.log_thread_finished)
        self.last_time = time.time()
        self.processing_frames = True  # New flag to indicate processing frames

    def run(self):
        try:
            logging.info("CameraThread started")
            start_time = time.time()
            self.cap = cv2.VideoCapture(self.source)
            end_time = time.time()
            logging.info(f"Camera access time: {end_time - start_time:.2f} seconds")

            if not self.cap.isOpened():
                try:
                    logging.error("Failed to open camera")
                    self.retry_connection()
                    return
                except Exception as e:
                    logging.error(f"Kamera tekrar açılamadı: {e}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.running = True
            while not self.isInterruptionRequested():
                ret, frame = self.cap.read()
                if ret:
                    self.latest_frame.put(frame, self.frame_id, realtime=True)
                    self.frame_id += 1

                    _, latest_frame = self.latest_frame.get()
                    bboxes, inference_time, tespit_durumu, tespit_zamani, person_count = self.yolo_model.detect(latest_frame)
                    frame = self.yolo_model.draw_boxes(latest_frame, bboxes)

                    current_time = time.time()
                    fps = 1.0 / (current_time - self.last_time)
                    self.last_time = current_time
                    self.update_fps_signal.emit(fps)
                    self.update_person_count_signal.emit(person_count)

                    if tespit_durumu != self.tespit_durumu_yeni:
                        self.last_detection_time = current_time
                        self.tespit_durumu_yeni = tespit_durumu
                        detected_dict = {
                            'Tespit Edilme Saati': datetime.strptime(tespit_zamani, "%Y-%m-%d %H:%M:%S"),
                            'Tespit Durumu': tespit_durumu,
                            'ID': 2  # Güncellenecek kaydın ID'si
                        }
                        db_thread = DatabaseThread(detected_dict, self.db_manager)
                        db_thread.start()
                        if tespit_durumu == 1 and not self.db_manager.connected:
                            self.db_manager.connect()

                    if tespit_durumu == 0 and current_time - self.last_detection_time > 3 and self.db_manager.connected:
                        self.db_manager.disconnect()

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                    p = convert_to_Qt_format.scaled(640, 640, QtCore.Qt.KeepAspectRatio)
                    self.change_pixmap_signal.emit(p)
                else:
                    logging.warning("Cannot read frame from camera")
                    self.stop()
                    self.retry_connection() # Frame alınmazsa tekrar bağlantı kur
            self.cap.release()
            self.processing_frames = False  # Indicate that processing is done
        except Exception as e:
            logging.error(f"Error in CameraThread: {e}")
            if self.cap:
                self.cap.release()

    def stop(self):
        try:
            self.requestInterruption()
            logging.info("Stopping CameraThread")
            self.wait()
        except Exception as e:
            logging.error(f"Error in CameraThread.stop: {e}")

    def retry_connection(self):
        try:
            while not self.running:
                logging.info("Attempting to reconnect to camera...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.source)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
                if self.cap.isOpened():
                    logging.info("Reconnected to camera")
                    self.running = True
                    break
                else:
                    logging.warning("Failed to reconnect to camera")
        except Exception as e:
            logging.error(f"Error in CameraThread.retry_connection: {e}")

    def log_thread_finished(self):
        logging.info("CameraThread successfully finished.")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    stop_signal = QtCore.pyqtSignal()

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.camera_thread = None
        self.latest_frame = LatestFrame()
        self.db_manager = DatabaseManager()
        self.yolo_model = YOLOv8Model('models/yolov8n.onnx')  # Default model
        self.init_ui()
        self.fps_label.setText(self.format_fps("--"))
        self.model_label.setText(self.format_model("--"))
        self.person_label.setText(self.format_person("--"))

        # Connect stop signal to the slot
        self.stop_signal.connect(self.write_last_detection)

        self.update_provider_label(self.yolo_model.model.provider)

    def init_ui(self):
        try:
            self.iou_slider.setMinimum(0)
            self.iou_slider.setMaximum(100)
            self.iou_slider.setValue(50)  # Default value
            self.iou_slider.valueChanged.connect(self.update_iou_value)

            self.conf_slider.setMinimum(0)
            self.conf_slider.setMaximum(100)
            self.conf_slider.setValue(50)  # Default value
            self.conf_slider.valueChanged.connect(self.update_conf_value)

            self.model_box.addItems(['yolov8n', 'yolov8m', 'yolov8l', 'yolov8x'])
            self.model_box.currentTextChanged.connect(self.change_model)

            self.camera_box.addItems(['rtsp://admin:admin1admin1@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'])

            self.open_camera.clicked.connect(self.start_camera)
            self.stop_camera.clicked.connect(self.stop_camera_stream)
        except Exception as e:
            logging.error(f"Error in MainWindow.init_ui: {e}")

    def update_iou_value(self, value):
        try:
            iou_value = value / 100.0
            self.iou_label_value.setText(f"{iou_value:.2f}")
            self.yolo_model.set_iou_threshold(iou_value)
            logging.info(f"IOU slider changed: {iou_value}")
        except Exception as e:
            logging.error(f"Error in MainWindow.update_iou_value: {e}")

    def update_conf_value(self, value):
        try:
            conf_value = value / 100.0
            self.conf_label_value.setText(f"{conf_value:.2f}")
            self.yolo_model.set_conf_threshold(conf_value)
            logging.info(f"Confidence slider changed: {conf_value}")
        except Exception as e:
            logging.error(f"Error in MainWindow.update_conf_value: {e}")

    def change_model(self, model_name):
        try:
            model_path = f'models/{model_name}.onnx'
            self.yolo_model = YOLOv8Model(model_path)
            self.model_label.setText(self.format_model(model_name))
            logging.info(f"Model changed: {model_name}")
        except Exception as e:
            logging.error(f"Error in MainWindow.change_model: {e}")

    def start_camera(self):
        try:
            if self.camera_thread is None:
                camera_source = self.camera_box.currentText()
                camera_url = 'rtsp://admin:admin1admin1@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'
                '''if camera_source == '0':
                    camera_url = 0
                else:
                    camera_url = camera_source'''

                self.camera_thread = CameraThread(camera_url, self.yolo_model, self.latest_frame, self.db_manager)
                time.sleep(0.2)
                self.camera_thread.change_pixmap_signal.connect(self.update_image)
                self.camera_thread.update_fps_signal.connect(self.update_fps)
                self.camera_thread.update_person_count_signal.connect(self.update_person_count)
                self.camera_thread.start()
                logging.info("Camera started")
        except Exception as e:
            logging.error(f"Error in MainWindow.start_camera: {e}")

    def stop_camera_stream(self):
        try:
            if self.camera_thread is not None:
                # Stop the camera thread first
                self.camera_thread.stop()
                self.camera_thread.terminate()

                # Emit the stop signal to write the last detection status
                self.stop_signal.emit()

                # Clear the frame buffer
                self.latest_frame.clear_buffer()

                # Clean up the camera thread and UI
                self.camera_thread = None
                self.clear_camera_label()
                logging.info("Camera stopped")
                self.fps_label.setText(self.format_fps("--"))
                self.person_label.setText(self.format_person("--"))
                self.model_label.setText(self.format_model("--"))
                self.db_manager.disconnect()
        except Exception as e:
            logging.error(f"Error in MainWindow.stop_camera: {e}")

    def write_last_detection(self):
        try:
            # Write the last detection status (0) to the database before stopping
            tespit_zamani = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            detected_dict = {
                'Tespit Edilme Saati': datetime.strptime(tespit_zamani, "%Y-%m-%d %H:%M:%S"),
                'Tespit Durumu': 0,
                'ID': 2  # Güncellenecek kaydın ID'si
            }
            db_thread = DatabaseThread(detected_dict, self.db_manager)
            db_thread.start()
            db_thread.wait()  # Ensure the database thread finishes before disconnecting
        except Exception as e:
            logging.error(f"Error in MainWindow.write_last_detection: {e}")

    def clear_camera_label(self):
        try:
            # Set a blank pixmap to clear the label
            blank_image = QtGui.QPixmap(640, 640)
            blank_image.fill(QtCore.Qt.white)
            self.camera_label.setPixmap(blank_image)

            # Process events to ensure the label is updated
            QtWidgets.QApplication.processEvents()

            # Set the text message
            self.camera_label.setText("Model ve Parametre Belirleyip Kamerayı Başlatınız")
            self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
            self.camera_label.setStyleSheet("QLabel { background-color : white; color : black; }")

            # Process events again to ensure the text and styles are applied
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            logging.error(f"Error in MainWindow.clear_camera_label: {e}")

    def update_image(self, qt_image):
        try:
            self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
            self.camera_label.setText("")  # Clear the text when showing the live feed
            self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
            self.camera_label.setStyleSheet("QLabel { background-color : none; }")
            # Process events to ensure the label is updated
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            logging.error(f"Error in MainWindow.update_image: {e}")

    def update_fps(self, fps):
        try:
            self.fps_label.setText(self.format_fps(f"{fps:.2f}"))
        except Exception as e:
            logging.error(f"Error in MainWindow.update_fps: {e}")

    def update_person_count(self, count):
        try:
            self.person_label.setText(self.format_person(str(count)))
        except Exception as e:
            logging.error(f"Error in MainWindow.update_person_count: {e}")

    def update_provider_label(self, provider):
        try:
            provider_text = self.format_provider(provider)
            self.provider_label.setText(provider_text)
        except Exception as e:
            logging.error(f"Error in MainWindow.update_proivder_label: {e}")

    def format_provider(self, provider_value):
        return f"<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">PROVİDER</span></p><hr/><p align=\"center\"><span style=\" font-size:12pt;\">{provider_value}</span></p></body></html>"
    def format_fps(self, fps_value):
        return f"<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">Fps</span></p><hr/><p align=\"center\"><span style=\" font-size:12pt;\">{fps_value}</span></p></body></html>"

    def format_model(self, model_value):
        return f"<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">Use Model</span></p><hr/><p align=\"center\"><span style=\" font-size:12pt;\">{model_value}</span></p></body></html>"

    def format_person(self, person_value):
        return f"<html><head/><body><p align=\"center\"><span style=\" font-size:12pt;\">Detection Person</span></p><hr/><p align=\"center\"><span style=\" font-size:12pt;\">{person_value}</span></p></body></html>"

try:
    if __name__ == "__main__":
        app = QtWidgets.QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        sys.exit(app.exec_())
except Exception as e:
    logging.exception("An error occurred: %s", e)
    input("Press Enter to exit...")

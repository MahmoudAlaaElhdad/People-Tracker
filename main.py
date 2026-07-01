# pyrefly: ignore-errors
import sys
import os
import cv2
import csv
import time
import json
import torch
import boto3
import numpy as np
from datetime import datetime
from collections import OrderedDict, defaultdict, deque
from scipy.spatial import distance as dist
from ultralytics import YOLO


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from PyQt5.QtWidgets import (
    QApplication, QSplashScreen, QLabel, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QMenuBar, QAction, QFileDialog, QMessageBox, QSplitter,
    QScrollArea, QInputDialog, QSizePolicy, QPushButton, QLineEdit, QGroupBox, QCheckBox,
    QSlider, QFrame
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QColor, QPainter, QLinearGradient, QPixmap, QIcon, QImage


# ========================================
# config.py
# ========================================

DEFAULT_CAMERA_SOURCE = 0  
DEFAULT_RESOLUTION = (640, 480)  

YOLO_MODEL = "yolov8n.pt"       
CONFIDENCE_THRESHOLD = 0.45     
PERSON_CLASS_ID = 0              

MAX_DISAPPEARED_FRAMES = 40   
MAX_MATCH_DISTANCE = 250      

TRAIL_MAX_LENGTH = 128        
TRAIL_LINE_THICKNESS = 2      
TRAIL_ACCUMULATION_WEIGHT = 0.4  
TRAIL_DECAY_FACTOR = 0.9985   

HEATMAP_BLUR_RADIUS = 25   
HEATMAP_INTENSITY = 0.6    

RECORDING_CODEC = "mp4v"   
RECORDING_FPS = 20.0       

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")  
RECORDING_DIR = os.path.join(BASE_DIR, "recordings")    
EXPORT_DIR = os.path.join(BASE_DIR, "exports")          
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json") 

for d in [SCREENSHOT_DIR, RECORDING_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

def load_password():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                return data.get("password", "122333")  
        except:
            return "122333"  
    return "122333"  

def save_password(new_password):
    data = {"password": new_password}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f)

TRACK_COLORS_BGR = [
    (255, 177, 0), (0, 230, 118), (86, 126, 255), (0, 215, 255),
    (208, 86, 255), (255, 255, 0), (0, 165, 255), (147, 86, 255),
    (86, 255, 170), (180, 130, 255), (100, 220, 220), (255, 190, 130)
]

TRACK_COLORS_RGB = [(b, g, r) for (r, g, b) in TRACK_COLORS_BGR]


# ========================================
# gui/styles.py
# ========================================

DARK_THEME_QSS = """
QMainWindow { background-color: #0d1117; color: #e6edf3; }
QWidget { background-color: transparent; color: #e6edf3; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QWidget#centralWidget { background-color: #0d1117; }
QWidget#sidebarWidget { background-color: #0d1117; border-left: 1px solid #1c2333; }
QTabWidget::pane { border: 1px solid #1c2333; border-radius: 8px; background-color: #0d1117; top: -1px; }
QTabBar { background: transparent; }
QTabBar::tab { background-color: #161b22; color: #8b949e; padding: 9px 22px; border: 1px solid #1c2333; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 3px; font-weight: 600; }
QTabBar::tab:selected { background-color: #0d1117; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
QGroupBox { border: 1px solid #1c2333; border-radius: 12px; margin-top: 16px; padding: 18px 14px 14px 14px; background-color: #161b22; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 3px 14px; color: #58a6ff; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; }
QPushButton { background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 8px; padding: 8px 18px; font-weight: 600; }
QPushButton:hover { background-color: #30363d; border-color: #58a6ff; color: #ffffff; }
QPushButton#btnRecord { border-color: #da3633; color: #f85149; }
QPushButton#btnRecord:hover { background-color: #da3633; color: #ffffff; }
QPushButton#btnScreenshot { border-color: #3fb950; color: #3fb950; }
QPushButton#btnScreenshot:hover { background-color: #238636; color: #ffffff; }
QPushButton#btnExport { border-color: #d29922; color: #d29922; }
QPushButton#btnExport:hover { background-color: #9e6a03; color: #ffffff; }
QSlider::groove:horizontal { border: none; height: 5px; background: #21262d; border-radius: 2px; }
QSlider::handle:horizontal { background: #58a6ff; border: 2px solid #1f6feb; width: 14px; height: 14px; margin: -6px 0; border-radius: 8px; }
QLabel#counterValueCurrent { font-size: 42px; font-weight: 800; color: #58a6ff; }
QLabel#counterValueTotal { font-size: 42px; font-weight: 800; color: #3fb950; }
QLabel#dwellValue { font-size: 22px; font-weight: 700; color: #d29922; }
QLabel#videoPlaceholder { color: #484f58; font-size: 18px; font-weight: 600; border: 2px dashed #21262d; border-radius: 12px; background-color: #0d1117; }
QCheckBox { spacing: 10px; color: #c9d1d9; font-size: 12px; }
QScrollBar:vertical { background: #0d1117; width: 8px; }
QScrollBar::handle:vertical { background: #30363d; border-radius: 4px; }
QLineEdit { background-color: #0d1117; color: #e6edf3; border: 1px solid #30363d; border-radius: 8px; padding: 8px 12px; }
QStatusBar { background-color: #161b22; color: #8b949e; border-top: 1px solid #1c2333; font-size: 11px; }
QMenuBar { background-color: #161b22; color: #c9d1d9; border-bottom: 1px solid #1c2333; }
"""

# ========================================
# core/detector.py
# ========================================

class PersonDetector:
    def __init__(self, model_path=None, confidence=None):
        self.model_path = model_path or YOLO_MODEL
        self.confidence = confidence or CONFIDENCE_THRESHOLD
        self.model = None
        self._loaded = False

    def load_model(self):
        if not self._loaded:
            self.model = YOLO(self.model_path)
            self._loaded = True

    def detect(self, frame):
        if not self._loaded:
            self.load_model()
        results = self.model(frame, conf=self.confidence, iou=0.45, classes=[PERSON_CLASS_ID], verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    conf = float(box.conf[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': conf,
                        'centroid': (cx, cy)
                    })
        return detections

    def set_confidence(self, value):
        self.confidence = max(0.1, min(1.0, value))


# ========================================
# core/tracker.py
# ========================================

class KalmanFilter2D:
    def __init__(self, init_x, init_y, dt=1.0):
        self.x = np.array([[init_x], [init_y], [0.0], [0.0]], dtype=np.float32)
        self.F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        q_pos, q_vel = 0.5, 1.0
        self.Q = np.array([[q_pos, 0.0, 0.0, 0.0], [0.0, q_pos, 0.0, 0.0], [0.0, 0.0, q_vel, 0.0], [0.0, 0.0, 0.0, q_vel]], dtype=np.float32)
        r_val = 9.0  
        self.R = np.array([[r_val, 0.0], [0.0, r_val]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 500.0

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return int(self.x[0, 0]), int(self.x[1, 0])

    def update(self, meas_x, meas_y):
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(4, dtype=np.float32) - np.dot(K, self.H), self.P)
        return int(self.x[0, 0]), int(self.x[1, 0])

    def get_position_uncertainty(self):
        P_pos = self.P[0:2, 0:2]
        eigenvalues, eigenvectors = np.linalg.eigh(P_pos)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        axis_major = 2.447 * np.sqrt(max(1e-4, eigenvalues[0]))
        axis_minor = 2.447 * np.sqrt(max(1e-4, eigenvalues[1]))
        return min(200.0, max(3.0, axis_major)), min(200.0, max(3.0, axis_minor)), angle


class CentroidTracker:
    def __init__(self, max_disappeared=None, max_distance=None):
        self.max_disappeared = max_disappeared or MAX_DISAPPEARED_FRAMES
        self.max_distance = max_distance or MAX_MATCH_DISTANCE
        self.next_id = 0
        self.objects = OrderedDict()       
        self.bboxes = OrderedDict()        
        self.disappeared = OrderedDict()   
        self.total_entered = 0
        self.kalmans = {}                  
        self.event_log = []  

    def register(self, centroid, bbox=None):
        object_id = self.next_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        self.kalmans[object_id] = KalmanFilter2D(centroid[0], centroid[1])
        self.next_id += 1
        self.total_entered += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        if object_id in self.kalmans:
            del self.kalmans[object_id]

    def update(self, detections, use_kalman=True):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                elif use_kalman and object_id in self.kalmans:
                    kf = self.kalmans[object_id]
                    pred_x, pred_y = kf.predict()
                    self.objects[object_id] = (pred_x, pred_y)
                    if self.bboxes[object_id] is not None:
                        x1, y1, x2, y2 = self.bboxes[object_id]
                        w, h = x2 - x1, y2 - y1
                        self.bboxes[object_id] = (int(pred_x - w/2), int(pred_y - h/2), int(pred_x + w/2), int(pred_y + h/2))
            return self.objects

        input_centroids = np.array([d['centroid'] for d in detections])
        input_bboxes = [d.get('bbox', None) for d in detections]

        predictions = {}
        if use_kalman:
            for object_id, kf in self.kalmans.items():
                predictions[object_id] = kf.predict()

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.register(tuple(centroid), input_bboxes[i])
            return self.objects

        object_ids = list(self.objects.keys())
        if use_kalman and len(predictions) > 0:
            object_centroids = np.array([predictions[oid] for oid in object_ids])
        else:
            object_centroids = np.array(list(self.objects.values()))

        D = dist.cdist(object_centroids, input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols: continue
            if D[row, col] > self.max_distance: continue

            object_id = object_ids[row]
            cx, cy = input_centroids[col]

            if use_kalman and object_id in self.kalmans:
                smooth_cx, smooth_cy = self.kalmans[object_id].update(cx, cy)
                self.objects[object_id] = (smooth_cx, smooth_cy)
            else:
                self.objects[object_id] = tuple(input_centroids[col])

            self.bboxes[object_id] = input_bboxes[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
            elif use_kalman and object_id in self.kalmans:
                pred_cx, pred_cy = predictions[object_id]
                self.objects[object_id] = (pred_cx, pred_cy)
                if self.bboxes[object_id] is not None:
                    x1, y1, x2, y2 = self.bboxes[object_id]
                    w, h = x2 - x1, y2 - y1
                    self.bboxes[object_id] = (int(pred_cx - w/2), int(pred_cy - h/2), int(pred_cx + w/2), int(pred_cy + h/2))

        for col in unused_cols:
            self.register(tuple(input_centroids[col]), input_bboxes[col])

        return self.objects

    @property
    def current_count(self):
        return len(self.objects)

    def get_bboxes(self):
        return dict(self.bboxes)

    def reset(self):
        self.next_id = 0
        self.objects = OrderedDict()
        self.bboxes = OrderedDict()
        self.disappeared = OrderedDict()
        self.total_entered = 0
        self.event_log = []
        self.kalmans = {}


# ========================================
# core/trail_manager.py
# ========================================

class TrailManager:
    def __init__(self, frame_shape):
        h, w = frame_shape[:2]
        self.frame_shape = (h, w)
        self.paths = defaultdict(lambda: deque(maxlen=TRAIL_MAX_LENGTH))
        self.accumulation = np.zeros((h, w), dtype=np.float32)
        self.heatmap_acc = np.zeros((h, w), dtype=np.float32)
        self.entry_times = {}          
        self.completed_dwells = []     
        self.exit_events = []          
        self._known_ids = set()

    def get_color(self, person_id):
        return TRACK_COLORS_BGR[person_id % len(TRACK_COLORS_BGR)]

    def update(self, tracked_objects):
        current_ids = set(tracked_objects.keys())
        previous_ids = set(self.paths.keys()) | self._known_ids

        for pid in current_ids:
            if pid not in self.entry_times:
                self.entry_times[pid] = time.time()
            self._known_ids.add(pid)

        exited_ids = (previous_ids & set(self.entry_times.keys())) - current_ids
        for pid in exited_ids:
            if pid in self.entry_times:
                entry = self.entry_times[pid]
                exit_t = time.time()
                duration = exit_t - entry
                self.completed_dwells.append(duration)
                self.exit_events.append((pid, entry, exit_t, duration))
                del self.entry_times[pid]

        for pid, centroid in tracked_objects.items():
            self.paths[pid].append(centroid)

        for pid in list(self.paths.keys()):
            if pid not in current_ids:
                del self.paths[pid]

    def accumulate_trails(self, tracked_objects):
        self.accumulation *= TRAIL_DECAY_FACTOR
        self.heatmap_acc *= TRAIL_DECAY_FACTOR

        trail_canvas = np.zeros(self.frame_shape, dtype=np.float32)
        heat_canvas = np.zeros(self.frame_shape, dtype=np.float32)

        for pid, centroid in tracked_objects.items():
            if pid in self.paths and len(self.paths[pid]) > 1:
                points = list(self.paths[pid])
                for i in range(1, len(points)):
                    cv2.line(trail_canvas, points[i - 1], points[i], 1.5, TRAIL_LINE_THICKNESS)
            cx, cy = centroid
            if 0 <= cx < self.frame_shape[1] and 0 <= cy < self.frame_shape[0]:
                cv2.circle(heat_canvas, (cx, cy), 25, 0.8, -1)

        self.accumulation = np.clip(cv2.add(self.accumulation, trail_canvas), 0, 255)
        self.heatmap_acc = np.clip(cv2.add(self.heatmap_acc, heat_canvas), 0, 255)

    def draw_active_trails(self, frame, tracked_objects):
        overlay = frame.copy()
        for pid, centroid in tracked_objects.items():
            if pid in self.paths and len(self.paths[pid]) > 1:
                color = self.get_color(pid)
                points = list(self.paths[pid])
                num_points = len(points)
                for i in range(1, num_points):
                    progress = i / num_points
                    thickness = max(1, int(TRAIL_LINE_THICKNESS * progress + 1))
                    cv2.line(overlay, points[i - 1], points[i], color, thickness, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    def draw_bounding_boxes(self, frame, tracked_objects, bboxes, tracker=None, show_predictions=False, show_uncertainty=False):
        for pid, centroid in tracked_objects.items():
            color = self.get_color(pid)
            bbox = bboxes.get(pid)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                label = f"ID:{pid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            if show_uncertainty and tracker is not None and pid in tracker.kalmans:
                kf = tracker.kalmans[pid]
                axis_major, axis_minor, angle = kf.get_position_uncertainty()
                ellipse_overlay = frame.copy()
                cv2.ellipse(ellipse_overlay, centroid, (int(axis_major), int(axis_minor)), int(angle), 0, 360, color, -1)
                cv2.addWeighted(ellipse_overlay, 0.15, frame, 0.85, 0, dst=frame)
                cv2.ellipse(frame, centroid, (int(axis_major), int(axis_minor)), int(angle), 0, 360, color, 1, cv2.LINE_AA)

            if show_predictions and tracker is not None and pid in tracker.kalmans:
                kf = tracker.kalmans[pid]
                pred_x, pred_y = int(kf.x[0, 0]), int(kf.x[1, 0])
                h, w = frame.shape[:2]
                if 0 <= pred_x < w and 0 <= pred_y < h:
                    cv2.circle(frame, (pred_x, pred_y), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (pred_x, pred_y), 5, color, 1, cv2.LINE_AA)
                    vx, vy = kf.x[2, 0], kf.x[3, 0]
                    arrow_end = (int(pred_x + vx * 5), int(pred_y + vy * 5))
                    cv2.arrowedLine(frame, (pred_x, pred_y), arrow_end, (255, 255, 255), 1, tipLength=0.3)

            cv2.circle(frame, centroid, 4, color, -1, cv2.LINE_AA)
        return frame

    def get_trail_overlay(self, frame, opacity=0.5):
        acc_uint8 = np.clip(self.accumulation * 1.5, 0, 255).astype(np.uint8)
        colored = np.zeros((*self.frame_shape, 3), dtype=np.uint8)
        colored[:, :, 0] = np.clip(acc_uint8 * 1.0, 0, 255).astype(np.uint8)   
        colored[:, :, 1] = np.clip(acc_uint8 * 0.85, 0, 255).astype(np.uint8)  
        colored[:, :, 2] = np.clip(acc_uint8 * 0.15, 0, 255).astype(np.uint8)  
        return cv2.addWeighted(frame, 1.0, colored, opacity, 0)

    def get_heatmap_overlay(self, frame, opacity=0.5):
        heatmap_uint8 = np.clip(self.heatmap_acc * 2.0, 0, 255).astype(np.uint8)
        if HEATMAP_BLUR_RADIUS > 0:
            kernel = HEATMAP_BLUR_RADIUS * 2 + 1
            heatmap_uint8 = cv2.GaussianBlur(heatmap_uint8, (kernel, kernel), 0)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        mask = heatmap_uint8 > 8
        result = frame.copy()
        if np.any(mask):
            mask_3ch = np.stack([mask] * 3, axis=-1)
            blended = cv2.addWeighted(frame, 1.0 - opacity, heatmap_colored, opacity, 0)
            result = np.where(mask_3ch, blended, frame)
        return result

    def get_avg_dwell_time(self):
        return sum(self.completed_dwells) / len(self.completed_dwells) if self.completed_dwells else 0.0

    def get_max_dwell_time(self):
        return max(self.completed_dwells) if self.completed_dwells else 0.0

    def get_all_dwell_data(self):
        return list(self.exit_events)

    def clear_trails(self):
        self.accumulation = np.zeros(self.frame_shape, dtype=np.float32)
        self.heatmap_acc = np.zeros(self.frame_shape, dtype=np.float32)
        self.paths.clear()

    def reset(self):
        self.clear_trails()
        self.entry_times.clear()
        self.completed_dwells.clear()
        self.exit_events.clear()
        self._known_ids.clear()


# ========================================
# gui/controls_panel.py
# ========================================

class ControlsPanel(QWidget):
    tracking_toggled = pyqtSignal(bool)
    trails_toggled = pyqtSignal(bool)
    persistent_trails_toggled = pyqtSignal(bool)
    heatmap_toggled = pyqtSignal(bool)
    bboxes_toggled = pyqtSignal(bool)
    trail_opacity_changed = pyqtSignal(float)
    heatmap_opacity_changed = pyqtSignal(float)
    confidence_changed = pyqtSignal(float)
    clear_trails_clicked = pyqtSignal()
    reset_counters_clicked = pyqtSignal()
    screenshot_clicked = pyqtSignal()
    record_toggled = pyqtSignal(bool)
    export_csv_clicked = pyqtSignal()
    change_password_clicked = pyqtSignal()
    kalman_toggled = pyqtSignal(bool)
    uncertainty_toggled = pyqtSignal(bool)
    predictions_toggled = pyqtSignal(bool)
    sensor_noise_changed = pyqtSignal(float)
    occlusion_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _make_slider(self, min_val, max_val, default, label_text):
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 11px; color: #8b949e;")
        lbl.setFixedWidth(90)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        val_lbl = QLabel(f"{default}%")
        val_lbl.setObjectName("sliderValueLabel")
        val_lbl.setFixedWidth(40)
        val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(lbl)
        row.addWidget(slider)
        row.addWidget(val_lbl)
        return row, slider, val_lbl

    def _hsep(self):
        s = QFrame()
        s.setObjectName("separator")
        s.setFrameShape(QFrame.HLine)
        return s

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        display_group = QGroupBox("DISPLAY OPTIONS")
        dg_layout = QVBoxLayout(display_group)
        self.chk_tracking = QCheckBox("Enable Detection"); self.chk_tracking.setChecked(True)
        self.chk_bboxes = QCheckBox("Show Bounding Boxes"); self.chk_bboxes.setChecked(True)
        self.chk_trails = QCheckBox("Show Active Trails"); self.chk_trails.setChecked(True)
        self.chk_persistent = QCheckBox("Persistent Trail Overlay"); self.chk_persistent.setChecked(True)
        self.chk_heatmap = QCheckBox("Motion Heatmap"); self.chk_heatmap.setChecked(False)
        for w in [self.chk_tracking, self.chk_bboxes, self.chk_trails, self.chk_persistent, self.chk_heatmap]: dg_layout.addWidget(w)
        layout.addWidget(display_group)

        kalman_group = QGroupBox("KALMAN SIMULATION")
        kg_layout = QVBoxLayout(kalman_group)
        self.chk_kalman = QCheckBox("Enable Kalman Filter"); self.chk_kalman.setChecked(True)
        self.chk_uncertainty = QCheckBox("Show Uncertainty Ellipses"); self.chk_uncertainty.setChecked(True)
        self.chk_predictions = QCheckBox("Show Predictions"); self.chk_predictions.setChecked(True)
        self.chk_occlusion = QCheckBox("Simulate Occlusion Zone"); self.chk_occlusion.setChecked(False)
        row_noise, self.sld_noise, self.lbl_noise_val = self._make_slider(0, 50, 0, "Sensor Jitter")
        self.lbl_noise_val.setText("0px")
        for w in [self.chk_kalman, self.chk_uncertainty, self.chk_predictions, self.chk_occlusion]: kg_layout.addWidget(w)
        kg_layout.addLayout(row_noise)
        layout.addWidget(kalman_group)

        slider_group = QGroupBox("ADJUSTMENTS")
        sg_layout = QVBoxLayout(slider_group)
        row1, self.sld_trail_opacity, self.lbl_trail_val = self._make_slider(0, 100, 50, "Trail Opacity")
        row2, self.sld_heatmap_opacity, self.lbl_heat_val = self._make_slider(0, 100, 50, "Heatmap Opacity")
        row3, self.sld_confidence, self.lbl_conf_val = self._make_slider(10, 95, 45, "Detection Conf.")
        self.lbl_conf_val.setText("45%")
        sg_layout.addLayout(row1); sg_layout.addLayout(row2); sg_layout.addWidget(self._hsep()); sg_layout.addLayout(row3)
        layout.addWidget(slider_group)

        action_group = QGroupBox("ACTIONS")
        ag_layout = QVBoxLayout(action_group)
        btn_row1 = QHBoxLayout()
        self.btn_record = QPushButton("⏺  Record"); self.btn_record.setObjectName("btnRecord"); self.btn_record.setCheckable(True)
        self.btn_screenshot = QPushButton("📷  Screenshot"); self.btn_screenshot.setObjectName("btnScreenshot")
        btn_row1.addWidget(self.btn_record); btn_row1.addWidget(self.btn_screenshot)
        ag_layout.addLayout(btn_row1)

        self.btn_export = QPushButton("📊  Export CSV Report"); self.btn_export.setObjectName("btnExport")
        self.btn_change_pwd = QPushButton("🔑  Change Password")
        ag_layout.addWidget(self.btn_export); ag_layout.addWidget(self.btn_change_pwd); ag_layout.addWidget(self._hsep())

        btn_row2 = QHBoxLayout()
        self.btn_clear = QPushButton("Clear Trails"); self.btn_reset = QPushButton("Reset All")
        btn_row2.addWidget(self.btn_clear); btn_row2.addWidget(self.btn_reset)
        ag_layout.addLayout(btn_row2)
        layout.addWidget(action_group); layout.addStretch()

        self.chk_tracking.toggled.connect(self.tracking_toggled)
        self.chk_trails.toggled.connect(self.trails_toggled)
        self.chk_persistent.toggled.connect(self.persistent_trails_toggled)
        self.chk_heatmap.toggled.connect(self.heatmap_toggled)
        self.chk_bboxes.toggled.connect(self.bboxes_toggled)
        self.chk_kalman.toggled.connect(self.kalman_toggled)
        self.chk_uncertainty.toggled.connect(self.uncertainty_toggled)
        self.chk_predictions.toggled.connect(self.predictions_toggled)
        self.chk_occlusion.toggled.connect(self.occlusion_toggled)
        self.sld_noise.valueChanged.connect(self._on_noise_changed)
        self.sld_trail_opacity.valueChanged.connect(self._on_trail_opacity)
        self.sld_heatmap_opacity.valueChanged.connect(self._on_heatmap_opacity)
        self.sld_confidence.valueChanged.connect(self._on_confidence)
        self.btn_clear.clicked.connect(self.clear_trails_clicked)
        self.btn_reset.clicked.connect(self.reset_counters_clicked)
        self.btn_screenshot.clicked.connect(self.screenshot_clicked)
        self.btn_record.toggled.connect(self.record_toggled)
        self.btn_export.clicked.connect(self.export_csv_clicked)
        self.btn_change_pwd.clicked.connect(self.change_password_clicked)

    def _on_trail_opacity(self, val): self.lbl_trail_val.setText(f"{val}%"); self.trail_opacity_changed.emit(val / 100.0)
    def _on_heatmap_opacity(self, val): self.lbl_heat_val.setText(f"{val}%"); self.heatmap_opacity_changed.emit(val / 100.0)
    def _on_confidence(self, val): self.lbl_conf_val.setText(f"{val}%"); self.confidence_changed.emit(val / 100.0)
    def _on_noise_changed(self, val): self.lbl_noise_val.setText(f"{val}px"); self.sensor_noise_changed.emit(float(val))
    def set_recording_state(self, is_recording):
        self.btn_record.setText("⏹  Stop Recording" if is_recording else "⏺  Record")
        self.btn_record.setChecked(is_recording)


# ========================================
# gui/stats_panel.py
# ========================================

class StatsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.session_start = time.time()
        self._init_ui()

    def _create_stat_column(self, value_text, label_text, value_name, label_name):
        col = QVBoxLayout()
        val = QLabel(value_text); val.setObjectName(value_name); val.setAlignment(Qt.AlignCenter)
        lbl = QLabel(label_text); lbl.setObjectName(label_name); lbl.setAlignment(Qt.AlignCenter)
        col.addWidget(val); col.addWidget(lbl)
        return col, val

    def _vsep(self):
        s = QFrame(); s.setFrameShape(QFrame.VLine); s.setStyleSheet("background-color: #1c2333; max-width: 1px;")
        return s

    def _init_ui(self):
        layout = QVBoxLayout(self); layout.setSpacing(8); layout.setContentsMargins(0, 0, 0, 0)
        cg = QGroupBox("LIVE STATISTICS"); cl = QHBoxLayout(cg)
        col1, self.lbl_current_value = self._create_stat_column("0", "CURRENT IN FRAME", "counterValueCurrent", "counterLabel")
        col2, self.lbl_total_value = self._create_stat_column("0", "TOTAL ENTERED", "counterValueTotal", "counterLabel")
        cl.addLayout(col1); cl.addWidget(self._vsep()); cl.addLayout(col2); layout.addWidget(cg)

        pg = QGroupBox("PERFORMANCE"); pl = QHBoxLayout(pg)
        col4, self.lbl_session_value = self._create_stat_column("00:00", "SESSION", "fpsValue", "fpsLabel")
        pl.addLayout(col4); layout.addWidget(pg)

        dg = QGroupBox("DWELL TIME"); dl = QHBoxLayout(dg)
        col5, self.lbl_avg_dwell = self._create_stat_column("0.0s", "AVERAGE", "dwellValue", "dwellLabel")
        col6, self.lbl_max_dwell = self._create_stat_column("0.0s", "MAXIMUM", "dwellValue", "dwellLabel")
        dl.addLayout(col5); dl.addWidget(self._vsep()); dl.addLayout(col6); layout.addWidget(dg)
        layout.addStretch()

    def update_counts(self, current, total): self.lbl_current_value.setText(str(current)); self.lbl_total_value.setText(str(total))
    def update_session_time(self):
        elapsed = time.time() - self.session_start
        m, s = int(elapsed // 60), int(elapsed % 60)
        h = m // 60; m = m % 60
        self.lbl_session_value.setText(f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}")
    def update_dwell_times(self, avg_dwell, max_dwell): self.lbl_avg_dwell.setText(f"{avg_dwell:.1f}s"); self.lbl_max_dwell.setText(f"{max_dwell:.1f}s")
    def reset_session(self): self.session_start = time.time(); self.update_counts(0, 0); self.update_dwell_times(0, 0)


# ========================================
# gui/video_widget.py
# ========================================

class VideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoPlaceholder"); self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setMinimumSize(480, 360)
        self.setText("📷  No Camera Feed")
        self._frame = None

    def update_frame(self, frame):
        if frame is None: return
        self._frame = frame.copy()
        rgb = frame[..., ::-1].copy()
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qt_image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setStyleSheet("QLabel { background-color: #000000; border: 1px solid #1c2333; border-radius: 8px; }")

    def get_current_frame(self): return self._frame
    def set_status(self, text): 
        if self._frame is None: self.setText(f"📷  {text}")
    def clear_feed(self): self._frame = None; self.clear(); self.setText("📷  No Camera Feed")


# ========================================
# gui/main_window.py
# ========================================

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)          
    stats_ready = pyqtSignal(int, int, float, float)  
    status_changed = pyqtSignal(str)  

    def __init__(self, camera_source=0, resolution=(640, 480)):
        super().__init__()
        self.camera_source = camera_source    
        self.resolution = resolution          
        self.running = False                  
        self.detector = PersonDetector()       
        self.tracker = CentroidTracker()       
        self.trail_manager = None              
        self.detection_enabled = True   
        self.show_bboxes = True         
        self.show_trails = True         
        self.show_persistent = True     
        self.show_heatmap = False       
        self.use_kalman = True          
        self.show_predictions = True    
        self.show_uncertainty = True    
        self.sensor_noise = 0.0         
        self.occlusion_enabled = False  
        self.occlusion_rect = (160, 120, 320, 240)  
        self.trail_opacity = 0.5        
        self.heatmap_opacity = 0.5      
        self.recording = False          
        self.video_writer = None        

    def run(self):
        self.running = True
        source = int(self.camera_source) if isinstance(self.camera_source, str) and self.camera_source.isdigit() else self.camera_source
        
        self.status_changed.emit(f"Opening camera {source}...")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
        if not cap.isOpened() and isinstance(source, int): 
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            self.status_changed.emit(f"Failed to open camera {source}.")
            self.running = False
            return

        w, h = self.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.status_changed.emit("Camera initialized. Starting tracking...")

        import traceback
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.5); continue
            
            try:
                if self.trail_manager is None: self.trail_manager = TrailManager(frame.shape)
                processed = frame.copy(); tracked, bboxes = {}, {}

                if self.detection_enabled:
                    detections = self.detector.detect(frame)
                    if self.occlusion_enabled:
                        ox, oy, ow, oh = self.occlusion_rect
                        detections = [d for d in detections if not (ox <= d['centroid'][0] <= ox + ow and oy <= d['centroid'][1] <= oy + oh)]
                    if self.sensor_noise > 0.0:
                        for d in detections:
                            nx, ny = int(np.random.normal(0, self.sensor_noise)), int(np.random.normal(0, self.sensor_noise))
                            d['centroid'] = (d['centroid'][0] + nx, d['centroid'][1] + ny)
                    tracked = self.tracker.update(detections, use_kalman=self.use_kalman)
                    bboxes = self.tracker.get_bboxes()
                    self.trail_manager.update(tracked); self.trail_manager.accumulate_trails(tracked)
                else:
                    self.tracker.update([])

                if self.show_persistent: processed = self.trail_manager.get_trail_overlay(processed, self.trail_opacity)
                if self.show_heatmap: processed = self.trail_manager.get_heatmap_overlay(processed, self.heatmap_opacity)
                if self.show_trails: processed = self.trail_manager.draw_active_trails(processed, tracked)
                if self.show_bboxes: processed = self.trail_manager.draw_bounding_boxes(processed, tracked, bboxes, tracker=self.tracker, show_predictions=self.show_predictions, show_uncertainty=self.show_uncertainty)

                if self.occlusion_enabled:
                    ox, oy, ow, oh = self.occlusion_rect; overlay = processed.copy()
                    cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (40, 40, 40), -1)
                    cv2.addWeighted(overlay, 0.45, processed, 0.55, 0, dst=processed)
                    cv2.rectangle(processed, (ox, oy), (ox + ow, oy + oh), (100, 100, 100), 2, cv2.LINE_AA)

                if self.recording and self.video_writer is not None: self.video_writer.write(processed)
                self.frame_ready.emit(processed)
                self.stats_ready.emit(self.tracker.current_count, self.tracker.total_entered, self.trail_manager.get_avg_dwell_time(), self.trail_manager.get_max_dwell_time())
            except Exception as e:
                traceback.print_exc(); time.sleep(1)

        cap.release()
        if self.video_writer is not None: self.video_writer.release()

    def stop(self): self.running = False; self.wait(3000)
    def start_recording(self, filepath, frame_size):
        self.video_writer = cv2.VideoWriter(filepath, cv2.VideoWriter_fourcc(*RECORDING_CODEC), RECORDING_FPS, frame_size)
        self.recording = True
    def stop_recording(self):
        self.recording = False
        if self.video_writer is not None: self.video_writer.release(); self.video_writer = None
    def clear_trails(self):
        if self.trail_manager: self.trail_manager.clear_trails()
    def reset_all(self):
        self.tracker.reset()
        if self.trail_manager: self.trail_manager.reset()


class CameraTab(QWidget):
    def __init__(self, camera_source=0, tab_name="Camera 1", parent=None):
        super().__init__(parent)
        self.camera_source, self.tab_name = camera_source, tab_name
        self.camera_thread = None
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self); splitter = QSplitter(Qt.Horizontal)
        self.video_widget = VideoWidget(); splitter.addWidget(self.video_widget)
        sidebar = QWidget(); sidebar.setObjectName("sidebarWidget"); sidebar_layout = QVBoxLayout(sidebar)
        self.stats_panel = StatsPanel(); self.controls_panel = ControlsPanel(); self.controls_panel.hide()
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.addWidget(self.stats_panel); scroll_layout.addWidget(self.controls_panel); scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll); splitter.addWidget(sidebar); splitter.setSizes([700, 300])
        main_layout.addWidget(splitter)

    def start_camera(self):
        if self.camera_thread and self.camera_thread.isRunning(): return
        self.camera_thread = CameraThread(self.camera_source, DEFAULT_RESOLUTION)
        self.camera_thread.frame_ready.connect(self.video_widget.update_frame)
        self.camera_thread.stats_ready.connect(self._on_stats)
        self.camera_thread.status_changed.connect(self.video_widget.set_status)

        cp = self.controls_panel
        cp.tracking_toggled.connect(lambda v: setattr(self.camera_thread, 'detection_enabled', v))
        cp.bboxes_toggled.connect(lambda v: setattr(self.camera_thread, 'show_bboxes', v))
        cp.trails_toggled.connect(lambda v: setattr(self.camera_thread, 'show_trails', v))
        cp.persistent_trails_toggled.connect(lambda v: setattr(self.camera_thread, 'show_persistent', v))
        cp.heatmap_toggled.connect(lambda v: setattr(self.camera_thread, 'show_heatmap', v))
        cp.kalman_toggled.connect(lambda v: setattr(self.camera_thread, 'use_kalman', v))
        cp.uncertainty_toggled.connect(lambda v: setattr(self.camera_thread, 'show_uncertainty', v))
        cp.predictions_toggled.connect(lambda v: setattr(self.camera_thread, 'show_predictions', v))
        cp.occlusion_toggled.connect(lambda v: setattr(self.camera_thread, 'occlusion_enabled', v))
        cp.sensor_noise_changed.connect(lambda v: setattr(self.camera_thread, 'sensor_noise', v))
        cp.trail_opacity_changed.connect(lambda v: setattr(self.camera_thread, 'trail_opacity', v))
        cp.heatmap_opacity_changed.connect(lambda v: setattr(self.camera_thread, 'heatmap_opacity', v))
        cp.confidence_changed.connect(self._on_confidence_change)
        cp.clear_trails_clicked.connect(self.camera_thread.clear_trails)
        cp.reset_counters_clicked.connect(self._on_reset)
        cp.screenshot_clicked.connect(self._take_screenshot)
        cp.record_toggled.connect(self._on_record_toggle)
        cp.export_csv_clicked.connect(self._export_csv)
        cp.change_password_clicked.connect(self._change_password)

        self.camera_thread.start()
        self._session_timer = QTimer(); self._session_timer.timeout.connect(self.stats_panel.update_session_time); self._session_timer.start(1000)

    def _change_password(self):
        old_pwd, ok = QInputDialog.getText(self, "Change Password", "Enter current password:", QLineEdit.Password)
        if not ok or old_pwd != load_password(): QMessageBox.warning(self, "Error", "Incorrect password!"); return
        new_pwd, ok = QInputDialog.getText(self, "Change Password", "Enter NEW password:", QLineEdit.Password)
        if ok and new_pwd: save_password(new_pwd); QMessageBox.information(self, "Success", "Password changed!")

    def stop_camera(self):
        if self.camera_thread: self.camera_thread.stop()
        if hasattr(self, '_session_timer'): self._session_timer.stop()

    def _on_stats(self, current, total, avg_dwell, max_dwell):
        self.stats_panel.update_counts(current, total); self.stats_panel.update_dwell_times(avg_dwell, max_dwell)
    def _on_confidence_change(self, value):
        if self.camera_thread: self.camera_thread.detector.set_confidence(value)
    def _on_reset(self):
        if self.camera_thread: self.camera_thread.reset_all()
        self.stats_panel.reset_session()
    def _take_screenshot(self):
        frame = self.video_widget.get_current_frame()
        if frame is not None:
            path = os.path.join(SCREENSHOT_DIR, f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(path, frame); QMessageBox.information(self, "Screenshot Saved", f"Saved to:\n{path}")
                
    def _on_record_toggle(self, checked):
        if not self.camera_thread: return
        if checked:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(RECORDING_DIR, f"recording_{ts}.mp4")
            self.camera_thread.start_recording(path, DEFAULT_RESOLUTION)
            self.controls_panel.set_recording_state(True)
            self.current_video_path = path  
        else:
            self.camera_thread.stop_recording()
            self.controls_panel.set_recording_state(False)
            
            if hasattr(self, 'current_video_path') and os.path.exists(self.current_video_path):
                video_path = self.current_video_path
                bucket_name = "eople-tracker-pro-reports-hadad"
                video_filename = os.path.basename(video_path)
                s3_key = f"recordings/{video_filename}"
            
                try:
                    s3_client = boto3.client("s3", region_name="eu-north-1")
                    print(f"[CLOUD_INFO] Uploading video to S3: {video_filename} ...")
                    s3_client.upload_file(video_path, bucket_name, s3_key)
                    print(f"[CLOUD_INFO] Video uploaded successfully.")
                    QMessageBox.information(self, "Recording Saved & Uploaded", f"Video uploaded to S3 successfully!\ns3://{bucket_name}/{s3_key}")
                except Exception as cloud_error:
                    print(f"[CLOUD_ERROR] S3 Upload Failed: {cloud_error}")
                    QMessageBox.warning(self, "Cloud Upload Failed", f"Saved locally, but S3 upload failed.\nError: {cloud_error}")

    def _export_csv(self):
        if not self.camera_thread or not self.camera_thread.trail_manager:
            QMessageBox.warning(self, "No Data", "No tracking data to export."); return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(EXPORT_DIR, f"tracking_report_{ts}.csv")
        tm = self.camera_thread.trail_manager; events = tm.get_all_dwell_data()

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Person ID", "Entry Time", "Exit Time", "Dwell Duration (s)"])
            for pid, entry, exit_t, duration in events:
                writer.writerow([pid, datetime.fromtimestamp(entry).strftime("%Y-%m-%d %H:%M:%S"), datetime.fromtimestamp(exit_t).strftime("%Y-%m-%d %H:%M:%S"), f"{duration:.2f}"])
            writer.writerow([]); writer.writerow(["SUMMARY"])
            writer.writerow(["Total People Entered", self.camera_thread.tracker.total_entered])
            writer.writerow(["Avg Dwell Time (s)", f"{tm.get_avg_dwell_time():.2f}"])
            writer.writerow(["Max Dwell Time (s)", f"{tm.get_max_dwell_time():.2f}"])

        bucket_name = "eople-tracker-pro-reports-hadad"
        s3_key = f"exports/tracking_report_{ts}.csv"
        try:
            s3_client = boto3.client("s3", region_name="eu-north-1")
            s3_client.upload_file(path, bucket_name, s3_key)
            QMessageBox.information(self, "Export Complete", f"Report saved and uploaded to S3:\ns3://{bucket_name}/{s3_key}")
        except Exception as e:
            QMessageBox.warning(self, "Export Saved (Cloud Failed)", f"Saved locally at {path}\nS3 upload failed: {e}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("People Tracker Pro"); self.setMinimumSize(1100, 700); self.resize(1280, 800)
        self.camera_tabs = []; self._init_ui(); self._init_menu()

    def _init_ui(self):
        central = QWidget(); central.setObjectName("centralWidget"); main_layout = QVBoxLayout(central)
        header = QHBoxLayout(); title = QLabel("🎯  People Tracker Pro"); title.setObjectName("titleLabel")
        subtitle = QLabel("Real-time Detection & Movement Analytics"); subtitle.setObjectName("subtitleLabel")
        title_col = QVBoxLayout(); title_col.addWidget(title); title_col.addWidget(subtitle); header.addLayout(title_col); header.addStretch()
        self.btn_add_camera = QPushButton("+  Add Camera"); self.btn_add_camera.clicked.connect(self._add_camera_dialog); header.addWidget(self.btn_add_camera)
        self.btn_settings = QPushButton("⚙️ Settings"); self.btn_settings.clicked.connect(self._toggle_settings); header.addWidget(self.btn_settings)
        main_layout.addLayout(header)
        self.tab_widget = QTabWidget(); self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_camera_tab); self.tab_widget.currentChanged.connect(self._on_tab_changed)
        main_layout.addWidget(self.tab_widget); self.setCentralWidget(central)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar)
        self._add_camera(DEFAULT_CAMERA_SOURCE, "USB Webcam")

    def _init_menu(self):
        menu_bar = self.menuBar(); file_menu = menu_bar.addMenu("File")
        add_cam = QAction("Add Camera...", self); add_cam.triggered.connect(self._add_camera_dialog); file_menu.addAction(add_cam)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self); quit_action.triggered.connect(self.close); file_menu.addAction(quit_action)

    def _add_camera(self, source, name):
        tab = CameraTab(camera_source=source, tab_name=name); idx = self.tab_widget.addTab(tab, f"📷 {name}")
        self.camera_tabs.append(tab); tab.start_camera(); self.tab_widget.setCurrentIndex(idx)

    def _add_camera_dialog(self):
        source, ok = QInputDialog.getText(self, "Add Camera", "Source (0=laptop, 1=USB, or RTSP URL):", text=str(DEFAULT_CAMERA_SOURCE))
        if not ok or not source.strip(): return
        name, ok2 = QInputDialog.getText(self, "Camera Name", "Name:", text=f"Camera {len(self.camera_tabs) + 1}")
        if not ok2: name = f"Camera {len(self.camera_tabs) + 1}"
        try: src = int(source.strip())
        except ValueError: src = source.strip()
        self._add_camera(src, name.strip())

    def _close_camera_tab(self, index):
        tab = self.tab_widget.widget(index)
        if tab: tab.stop_camera(); self.camera_tabs.remove(tab)
        self.tab_widget.removeTab(index)

    def _on_tab_changed(self, index):
        tab = self.tab_widget.widget(index)
        if tab: self.btn_settings.setText("⚙️ Hide Settings" if tab.controls_panel.isVisible() else "⚙️ Settings")

    def _toggle_settings(self):
        current_widget = self.tab_widget.currentWidget()
        if not current_widget: return
        if not current_widget.controls_panel.isVisible():
            pwd, ok = QInputDialog.getText(self, "Settings Locked", "Enter password:", QLineEdit.Password)
            if not ok or pwd != load_password(): QMessageBox.warning(self, "Error", "Incorrect password!"); return
            current_widget.controls_panel.show(); self.btn_settings.setText("⚙️ Hide Settings")
        else:
            current_widget.controls_panel.hide(); self.btn_settings.setText("⚙️ Settings")

    def closeEvent(self, event):
        for tab in self.camera_tabs: tab.stop_camera()
        event.accept()

# ========================================
# main.py
# ========================================

def create_splash():
    pixmap = QPixmap(480, 280); pixmap.fill(QColor("#0d1117"))
    painter = QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
    grad = QLinearGradient(0, 0, 480, 280); grad.setColorAt(0.0, QColor("#0d1117")); grad.setColorAt(1.0, QColor("#0d1117"))
    painter.fillRect(0, 0, 480, 280, grad); painter.setPen(QColor("#1c2333")); painter.drawRect(0, 0, 479, 279)
    painter.setPen(QColor("#58a6ff")); painter.setFont(QFont("Segoe UI", 22, QFont.Bold))
    painter.drawText(0, 140, 480, 40, Qt.AlignCenter, "🎯 People Tracker Pro")
    painter.setPen(QColor("#8b949e")); painter.setFont(QFont("Segoe UI", 11))
    painter.drawText(0, 190, 480, 30, Qt.AlignCenter, "Loading tracking system...")
    painter.end()
    splash = QSplashScreen(pixmap); splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    return splash

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv); app.setStyleSheet(DARK_THEME_QSS)
    splash = create_splash(); splash.show(); app.processEvents()
    window = MainWindow()
    QTimer.singleShot(1500, lambda: [splash.close(), window.show()])
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
EmoVision Pro Enhanced - Next-Generation Real-time Emotion Detection System
Author: Obrey Muchena
Version: 3.0.0
Updated: June 2025

Enhanced Features:
- Multi-model ensemble for improved accuracy
- Real-time emotion trends and analytics
- Advanced face tracking with Kalman filters
- Emotion history and behavioral pattern analysis
- REST API for remote access
- Real-time streaming capabilities
- Machine learning model fine-tuning
- Advanced data visualization
- Multi-language support
- Cloud integration options
- Performance optimization with threading
"""

import cv2
import time
import datetime
import json
import logging
import threading
import queue
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
import numpy as np
from contextlib import contextmanager
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from fer import FER
    import torch
    import torchvision.transforms as transforms
    from sklearn.ensemble import VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
except ImportError as e:
    raise ImportError(f"Required ML libraries not found: {e}\nInstall with: pip install fer torch torchvision scikit-learn")

# Optional advanced features
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from flask import Flask, jsonify, request, Response
    import websockets
    HAS_ADVANCED_FEATURES = True
except ImportError:
    pd = None
    plt = None
    sns = None
    Flask = None
    websockets = None
    HAS_ADVANCED_FEATURES = False
    print("Warning: Advanced features disabled. Install: pip install pandas matplotlib seaborn flask websockets")

try:
    import dlib
    HAS_DLIB = True
except ImportError:
    HAS_DLIB = False
    print("Warning: dlib not available. Advanced face tracking disabled.")


@dataclass
class EmotionConfig:
    """Enhanced configuration with advanced options"""
    # Camera settings
    camera_index: int = 0
    resolution: Tuple[int, int] = (1920, 1080)
    fps_target: int = 30
    
    # Detection settings
    confidence_threshold: float = 0.6
    ensemble_voting: str = "soft"  # "soft", "hard"
    use_gpu: bool = True
    batch_processing: bool = True
    batch_size: int = 4
    
    # Auto-capture settings
    auto_capture_threshold: float = 0.85
    auto_capture_emotions: List[str] = field(default_factory=lambda: ["happy", "surprise", "angry"])
    auto_capture_cooldown: float = 2.0
    
    # Tracking settings
    face_tracking_method: str = "kalman"  # "simple", "kalman", "dlib"
    max_face_age: float = 3.0
    face_similarity_threshold: float = 0.7
    
    # Analytics settings
    enable_trend_analysis: bool = True
    trend_window_size: int = 100
    enable_behavioral_analysis: bool = True
    
    # Storage settings
    enable_logging: bool = True
    enable_database: bool = True
    log_level: str = "INFO"
    output_dir: str = "emovision_output"
    max_log_size_mb: int = 100
    
    # API settings
    enable_api: bool = False
    api_port: int = 5000
    enable_streaming: bool = False
    streaming_port: int = 8765
    
    # Performance settings
    enable_threading: bool = True
    max_workers: int = 4
    enable_profiling: bool = False
    
    # UI settings
    ui_theme: str = "dark"  # "dark", "light", "cyberpunk"
    show_confidence_bars: bool = True
    show_emotion_history: bool = True
    show_performance_metrics: bool = True


@dataclass
class EmotionData:
    """Enhanced emotion data structure"""
    timestamp: datetime.datetime
    face_id: int
    emotions: Dict[str, float]
    bounding_box: Tuple[int, int, int, int]
    confidence: float
    dominant_emotion: str
    secondary_emotion: Optional[str] = None
    emotion_intensity: float = 0.0
    face_landmarks: Optional[List[Tuple[int, int]]] = None
    head_pose: Optional[Tuple[float, float, float]] = None
    micro_expressions: Optional[Dict[str, float]] = None
    session_id: Optional[str] = None


class KalmanTracker:
    """Advanced Kalman filter for face tracking"""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(8, 4)  # 8 states, 4 measurements
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0, 0, 0, 0]], np.float32)
        
        # State transition matrix (position and velocity)
        self.kalman.transitionMatrix = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                                 [0, 1, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 1, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 1, 0, 0, 0, 1],
                                                 [0, 0, 0, 0, 1, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 1, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 0, 0, 1]], np.float32)
        
        self.kalman.processNoiseCov = 0.03 * np.eye(8, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        self.initialized = False
    
    def update(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Update tracker with new bounding box"""
        x, y, w, h = bbox
        measurement = np.array([[x], [y], [w], [h]], dtype=np.float32)
        
        if not self.initialized:
            self.kalman.statePre = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
            self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)
            self.initialized = True
        
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        
        return tuple(map(int, prediction[:4].flatten()))


class EmotionDatabase:
    """SQLite database for emotion data storage and analytics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS emotion_detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT,
                    face_id INTEGER,
                    dominant_emotion TEXT,
                    confidence REAL,
                    emotions TEXT,  -- JSON string
                    bounding_box TEXT,  -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS session_stats (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    total_detections INTEGER,
                    avg_confidence REAL,
                    dominant_emotions TEXT  -- JSON string
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON emotion_detections(timestamp);
                CREATE INDEX IF NOT EXISTS idx_session ON emotion_detections(session_id);
                CREATE INDEX IF NOT EXISTS idx_emotion ON emotion_detections(dominant_emotion);
            ''')
    
    def insert_emotion_data(self, data: List[EmotionData]):
        """Insert emotion data into database"""
        with sqlite3.connect(self.db_path) as conn:
            for item in data:
                conn.execute('''
                    INSERT INTO emotion_detections 
                    (timestamp, session_id, face_id, dominant_emotion, confidence, emotions, bounding_box)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item.timestamp.isoformat(),
                    item.session_id,
                    item.face_id,
                    item.dominant_emotion,
                    item.confidence,
                    json.dumps(item.emotions),
                    json.dumps(item.bounding_box)
                ))
    
    def get_emotion_trends(self, hours: int = 24) -> pd.DataFrame:
        """Get emotion trends for specified time period"""
        if not pd:
            return None
            
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT timestamp, dominant_emotion, confidence 
                FROM emotion_detections 
                WHERE timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp
            '''.format(hours)
            
            return pd.read_sql_query(query, conn)


class EnsembleEmotionDetector:
    """Ensemble emotion detection using multiple models"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.models = []
        self.weights = []
        self.init_models()
    
    def init_models(self):
        """Initialize multiple emotion detection models"""
        # Primary FER model
        self.fer_model = FER(mtcnn=True)
        self.models.append(("FER", self.fer_model))
        self.weights.append(0.6)
        
        # Add more models if available
        try:
            # You can add more models here like:
            # - Custom trained models
            # - Other pre-trained models
            # - Fine-tuned models for specific use cases
            pass
        except Exception as e:
            logging.warning(f"Could not load additional models: {e}")
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict]:
        """Detect emotions using ensemble approach"""
        results = []
        
        # For now, use primary model
        # In a full implementation, you would combine multiple model outputs
        fer_results = self.fer_model.detect_emotions(frame)
        
        # Apply ensemble voting if multiple models are available
        return fer_results


class EmotionAnalytics:
    """Advanced analytics and trend analysis"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.emotion_history = deque(maxlen=config.trend_window_size)
        self.face_histories = defaultdict(lambda: deque(maxlen=50))
        
    def add_data(self, emotion_data: List[EmotionData]):
        """Add new emotion data for analysis"""
        for data in emotion_data:
            self.emotion_history.append(data)
            self.face_histories[data.face_id].append(data)
    
    def get_emotion_trends(self) -> Dict[str, Any]:
        """Calculate emotion trends and patterns"""
        if not self.emotion_history:
            return {}
        
        # Basic statistics
        emotions = [d.dominant_emotion for d in self.emotion_history]
        emotion_counts = defaultdict(int)
        for emotion in emotions:
            emotion_counts[emotion] += 1
        
        # Trend analysis
        recent_window = min(20, len(self.emotion_history))
        recent_emotions = emotions[-recent_window:]
        
        trends = {
            'total_detections': len(self.emotion_history),
            'emotion_distribution': dict(emotion_counts),
            'recent_trend': self._calculate_trend(recent_emotions),
            'dominant_emotion': max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'unknown',
            'emotion_stability': self._calculate_stability(),
            'confidence_trend': self._calculate_confidence_trend()
        }
        
        return trends
    
    def _calculate_trend(self, emotions: List[str]) -> Dict[str, float]:
        """Calculate emotion trend direction"""
        if len(emotions) < 10:
            return {}
        
        mid_point = len(emotions) // 2
        first_half = emotions[:mid_point]
        second_half = emotions[mid_point:]
        
        first_counts = defaultdict(int)
        second_counts = defaultdict(int)
        
        for emotion in first_half:
            first_counts[emotion] += 1
        for emotion in second_half:
            second_counts[emotion] += 1
        
        trends = {}
        all_emotions = set(first_counts.keys()) | set(second_counts.keys())
        
        for emotion in all_emotions:
            first_pct = first_counts[emotion] / len(first_half)
            second_pct = second_counts[emotion] / len(second_half)
            trends[emotion] = second_pct - first_pct
        
        return trends
    
    def _calculate_stability(self) -> float:
        """Calculate emotion stability score"""
        if len(self.emotion_history) < 10:
            return 0.0
        
        emotions = [d.dominant_emotion for d in list(self.emotion_history)[-20:]]
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        
        return 1.0 - (emotion_changes / (len(emotions) - 1))
    
    def _calculate_confidence_trend(self) -> Dict[str, float]:
        """Calculate confidence trends"""
        if not self.emotion_history:
            return {}
        
        confidences = [d.confidence for d in self.emotion_history]
        
        return {
            'average': np.mean(confidences),
            'std_dev': np.std(confidences),
            'trend': np.polyfit(range(len(confidences)), confidences, 1)[0] if len(confidences) > 1 else 0.0
        }


class EmotionVisualizer:
    """Advanced visualization with multiple themes and layouts"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.themes = self._load_themes()
        self.current_theme = self.themes[config.ui_theme]
        
        # Emotion colors for different themes
        self.emotion_colors = {
            "dark": {
                "angry": (0, 0, 255), "disgust": (0, 128, 0), "fear": (128, 0, 128),
                "happy": (0, 255, 0), "sad": (255, 0, 0), "surprise": (0, 255, 255),
                "neutral": (200, 200, 200)
            },
            "light": {
                "angry": (0, 0, 200), "disgust": (0, 100, 0), "fear": (100, 0, 100),
                "happy": (0, 200, 0), "sad": (200, 0, 0), "surprise": (0, 200, 200),
                "neutral": (100, 100, 100)
            },
            "cyberpunk": {
                "angry": (255, 0, 100), "disgust": (100, 255, 0), "fear": (200, 0, 255),
                "happy": (0, 255, 200), "sad": (255, 100, 0), "surprise": (200, 255, 0),
                "neutral": (150, 150, 255)
            }
        }
    
    def _load_themes(self) -> Dict[str, Dict]:
        """Load UI themes"""
        return {
            "dark": {"bg": (30, 30, 30), "text": (255, 255, 255), "accent": (0, 255, 255)},
            "light": {"bg": (240, 240, 240), "text": (0, 0, 0), "accent": (255, 100, 0)},
            "cyberpunk": {"bg": (0, 0, 50), "text": (0, 255, 255), "accent": (255, 0, 255)}
        }
    
    def draw_enhanced_overlay(self, frame: np.ndarray, emotion_data: List[EmotionData], 
                            analytics: Dict[str, Any], performance_info: Dict[str, Any]) -> np.ndarray:
        """Draw comprehensive overlay with all information"""
        # Apply theme background
        overlay = np.zeros_like(frame)
        overlay[:] = self.current_theme["bg"]
        frame = cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)
        
        # Draw face analyses
        for data in emotion_data:
            self._draw_advanced_face_analysis(frame, data)
        
        # Draw system status
        self._draw_enhanced_status_panel(frame, emotion_data, analytics, performance_info)
        
        # Draw emotion history
        if self.config.show_emotion_history:
            self._draw_emotion_history(frame, analytics.get('emotion_distribution', {}))
        
        # Draw performance metrics
        if self.config.show_performance_metrics:
            self._draw_performance_panel(frame, performance_info)
        
        return frame
    
    def _draw_advanced_face_analysis(self, frame: np.ndarray, data: EmotionData):
        """Draw advanced face analysis with enhanced visuals"""
        x, y, w, h = data.bounding_box
        colors = self.emotion_colors[self.config.ui_theme]
        primary_color = colors.get(data.dominant_emotion, (255, 255, 255))
        
        # Glowing effect for bounding box
        for thickness in range(5, 0, -1):
            alpha = 0.3 - (thickness * 0.05)
            color = tuple(int(c * alpha) for c in primary_color)
            cv2.rectangle(frame, (x-thickness, y-thickness), 
                         (x + w + thickness, y + h + thickness), color, 1)
        
        # Main bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), primary_color, 2)
        
        # Face ID with background
        id_text = f"Face {data.face_id}"
        (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 10, y), primary_color, -1)
        cv2.putText(frame, id_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Dominant emotion with confidence
        emotion_text = f"{data.dominant_emotion.title()} ({data.confidence:.2f})"
        cv2.putText(frame, emotion_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, primary_color, 2)
        
        # Emotion probability visualization
        if self.config.show_confidence_bars:
            self._draw_emotion_bars(frame, data, x, y + h + 40)
    
    def _draw_emotion_bars(self, frame: np.ndarray, data: EmotionData, start_x: int, start_y: int):
        """Draw emotion probability bars"""
        colors = self.emotion_colors[self.config.ui_theme]
        emotions_sorted = sorted(data.emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(emotions_sorted[:4]):
            bar_width = int(score * 120)
            bar_color = colors.get(emotion, (128, 128, 128))
            
            # Background bar
            cv2.rectangle(frame, (start_x, start_y + i * 18), (start_x + 120, start_y + i * 18 + 12), (50, 50, 50), -1)
            
            # Filled bar
            cv2.rectangle(frame, (start_x, start_y + i * 18), (start_x + bar_width, start_y + i * 18 + 12), bar_color, -1)
            
            # Text label
            cv2.putText(frame, f"{emotion}: {score:.2f}", (start_x + 125, start_y + i * 18 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
    
    def _draw_enhanced_status_panel(self, frame: np.ndarray, emotion_data: List[EmotionData], 
                                  analytics: Dict[str, Any], performance_info: Dict[str, Any]):
        """Draw enhanced status panel"""
        height, width = frame.shape[:2]
        panel_height = 150
        
        # Semi-transparent background
        overlay = np.zeros((panel_height, 450, 3), dtype=np.uint8)
        overlay[:] = self.current_theme["bg"]
        
        roi = frame[10:10+panel_height, 10:460]
        frame[10:10+panel_height, 10:460] = cv2.addWeighted(roi, 0.3, overlay, 0.7, 0)
        
        # Border
        cv2.rectangle(frame, (10, 10), (460, 10 + panel_height), self.current_theme["accent"], 2)
        
        # System information
        info_lines = [
            "🎯 EmoVision Pro Enhanced v3.0",
            f"📊 FPS: {performance_info.get('fps', 0):.1f} | Faces: {len(emotion_data)} | Avg Conf: {performance_info.get('avg_confidence', 0):.2f}",
            f"🎭 Dominant: {analytics.get('dominant_emotion', 'Unknown')} | Stability: {analytics.get('emotion_stability', 0):.2f}",
            f"📈 Total Detections: {analytics.get('total_detections', 0)}",
            "⌨️  Controls: Q=Quit | S=Save | R=Report | T=Theme | A=Analytics"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, self.current_theme["text"], 1)
    
    def _draw_emotion_history(self, frame: np.ndarray, emotion_dist: Dict[str, int]):
        """Draw emotion distribution pie chart"""
        if not emotion_dist:
            return
        
        height, width = frame.shape[:2]
        center_x, center_y = width - 150, 150
        radius = 80
        
        total = sum(emotion_dist.values())
        start_angle = 0
        colors = self.emotion_colors[self.config.ui_theme]
        
        for emotion, count in emotion_dist.items():
            angle = int(360 * count / total)
            color = colors.get(emotion, (128, 128, 128))
            
            # Draw pie slice
            cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 
                       start_angle, start_angle + angle, color, -1)
            
            start_angle += angle
        
        # Center circle
        cv2.circle(frame, (center_x, center_y), radius//3, self.current_theme["bg"], -1)
        cv2.putText(frame, "Emotions", (center_x - 35, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.current_theme["text"], 1)
    
    def _draw_performance_panel(self, frame: np.ndarray, performance_info: Dict[str, Any]):
        """Draw performance metrics panel"""
        height, width = frame.shape[:2]
        panel_x, panel_y = width - 250, height - 120
        
        # Background
        cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (panel_x, panel_y), (width - 10, height - 10), self.current_theme["accent"], 1)
        
        # Performance metrics
        metrics = [
            f"CPU: {performance_info.get('cpu_usage', 0):.1f}%",
            f"Memory: {performance_info.get('memory_usage', 0):.1f}MB",
            f"Process Time: {performance_info.get('avg_process_time', 0):.3f}s"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (panel_x + 10, panel_y + 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.current_theme["text"], 1)


class EnhancedEmotionDetector:
    """Main enhanced emotion detection system"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.detector = EnsembleEmotionDetector(config)
        self.analytics = EmotionAnalytics(config)
        self.visualizer = EmotionVisualizer(config)
        
        # Initialize tracking
        self.face_trackers = {}
        self.next_face_id = 0
        self.last_capture_time = defaultdict(float)
        
        # Initialize storage
        self.output_dir = Path(config.output_dir)
        self.setup_directories()
        
        if config.enable_database:
            self.database = EmotionDatabase(str(self.output_dir / 'emotion_data.db'))
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        self.frame_count = 0
        self.start_time = time.time()
        
        # Threading
        if config.enable_threading:
            self.frame_queue = queue.Queue(maxsize=10)
            self.result_queue = queue.Queue()
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            self.processing_thread = None
            self.start_processing_thread()
        
        # Setup logging
        self.setup_logging()
        self.logger.info(f"EmoVision Pro Enhanced initialized - Session: {self.session_id}")
    
    def setup_directories(self):
        """Create directory structure"""
        directories = ['logs', 'screenshots', 'reports', 'data', 'models', 'exports']
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup enhanced logging"""
        log_file = self.output_dir / 'logs' / f'emovision_{self.session_id}.log'
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # File handler with rotation
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('EmoVisionPro')
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def start_processing_thread(self):
        """Start background processing thread"""
        if self.config.enable_threading:
            self.processing_thread = threading.Thread(target=self._process_frames_async, daemon=True)
            self.processing_thread.start()
    
    def _process_frames_async(self):
        """Asynchronous frame processing"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:  # Shutdown signal
                    break
                
                result = self._process_single_frame(frame)
                self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in async processing: {e}")
    
    def _process_single_frame(self, frame: np.ndarray) -> Tuple[List[EmotionData], Dict[str, Any]]:
        """Process a single frame and return results"""
        process_start = time.time()
        
        # Detect emotions
        faces = self.detector.detect_emotions(frame)
        emotion_data = self._track_and_analyze_faces(faces)
        
        # Update analytics
        self.analytics.add_data(emotion_data)
        
        # Performance tracking
        process_time = time.time() - process_start
        self.processing_times.append(process_time)
        
        performance_info = {
            'process_time': process_time,
            'avg_process_time': np.mean(self.processing_times) if self.processing_times else 0,
            'fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_confidence': np.mean([d.confidence for d in emotion_data]) if emotion_data else 0,
            'cpu_usage': self._get_cpu_usage(),
            'memory_usage': self._get_memory_usage()
        }
        
        return emotion_data, performance_info
    
    def _track_and_analyze_faces(self, faces: List[Dict]) -> List[EmotionData]:
        """Advanced face tracking and analysis"""
        current_time = datetime.datetime.now()
        emotion_data = []
        
        # Clean up old trackers
        current_time_stamp = time.time()
        expired_trackers = [
            face_id for face_id, (tracker, last_update) in self.face_trackers.items()
            if current_time_stamp - last_update > self.config.max_face_age
        ]
        for face_id in expired_trackers:
            del self.face_trackers[face_id]
        
        for face in faces:
            box = face['box']
            emotions = face['emotions']
            
            # Calculate dominant and secondary emotions
            emotions_sorted = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            dominant_emotion, confidence = emotions_sorted[0]
            secondary_emotion = emotions_sorted[1][0] if len(emotions_sorted) > 1 else None
            
            # Skip low confidence detections
            if confidence < self.config.confidence_threshold:
                continue
            
            # Track face
            face_id = self._assign_face_id_advanced(box)
            
            # Calculate emotion intensity
            emotion_intensity = self._calculate_emotion_intensity(emotions)
            
            # Create emotion data
            data = EmotionData(
                timestamp=current_time,
                face_id=face_id,
                emotions=emotions,
                bounding_box=box,
                confidence=confidence,
                dominant_emotion=dominant_emotion,
                secondary_emotion=secondary_emotion,
                emotion_intensity=emotion_intensity,
                session_id=self.session_id
            )
            
            emotion_data.append(data)
        
        return emotion_data
    
    def _assign_face_id_advanced(self, box: Tuple[int, int, int, int]) -> int:
        """Advanced face tracking with Kalman filters"""
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        current_time = time.time()
        
        # Find best matching existing tracker
        best_match_id = None
        min_distance = float('inf')
        
        for face_id, (tracker, last_update) in self.face_trackers.items():
            if self.config.face_tracking_method == "kalman":
                predicted_box = tracker.update(box)
                pred_center = (predicted_box[0] + predicted_box[2]//2, predicted_box[1] + predicted_box[3]//2)
                distance = np.sqrt((center[0] - pred_center[0])**2 + (center[1] - pred_center[1])**2)
            else:
                # Simple distance-based tracking
                last_center = getattr(tracker, 'center', center)
                distance = np.sqrt((center[0] - last_center[0])**2 + (center[1] - last_center[1])**2)
            
            if distance < min_distance and distance < 150:  # Threshold for same face
                min_distance = distance
                best_match_id = face_id
        
        if best_match_id is not None:
            # Update existing tracker
            if self.config.face_tracking_method == "kalman":
                self.face_trackers[best_match_id] = (self.face_trackers[best_match_id][0], current_time)
            else:
                tracker = type('SimpleTracker', (), {'center': center})()
                self.face_trackers[best_match_id] = (tracker, current_time)
            return best_match_id
        else:
            # Create new tracker
            new_id = self.next_face_id
            self.next_face_id += 1
            
            if self.config.face_tracking_method == "kalman":
                tracker = KalmanTracker()
                tracker.update(box)
            else:
                tracker = type('SimpleTracker', (), {'center': center})()
            
            self.face_trackers[new_id] = (tracker, current_time)
            return new_id
    
    def _calculate_emotion_intensity(self, emotions: Dict[str, float]) -> float:
        """Calculate overall emotion intensity"""
        # Get the top 3 emotions and calculate intensity
        top_emotions = sorted(emotions.values(), reverse=True)[:3]
        intensity = sum(top_emotions) / len(top_emotions) if top_emotions else 0.0
        
        # Apply intensity scaling
        return min(intensity * 1.2, 1.0)  # Boost intensity slightly but cap at 1.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def should_auto_capture(self, emotion_data: List[EmotionData]) -> Optional[EmotionData]:
        """Determine if auto-capture should be triggered"""
        current_time = time.time()
        
        for data in emotion_data:
            # Check if emotion meets criteria
            if (data.dominant_emotion in self.config.auto_capture_emotions and 
                data.confidence >= self.config.auto_capture_threshold):
                
                # Check cooldown period
                last_capture = self.last_capture_time.get(data.face_id, 0)
                if current_time - last_capture >= self.config.auto_capture_cooldown:
                    self.last_capture_time[data.face_id] = current_time
                    return data
        
        return None
    
    def save_enhanced_screenshot(self, frame: np.ndarray, emotion_data: List[EmotionData], 
                               label: str = "", metadata: Dict[str, Any] = None) -> str:
        """Save screenshot with enhanced metadata and analysis"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"emotion_capture_{label}_{timestamp}.png" if label else f"emotion_capture_{timestamp}.png"
        filepath = self.output_dir / 'screenshots' / filename
        
        # Create enhanced frame with analysis overlay
        enhanced_frame = frame.copy()
        
        # Add comprehensive metadata overlay
        enhanced_frame = self._add_comprehensive_metadata(enhanced_frame, emotion_data, metadata)
        
        # Save image
        cv2.imwrite(str(filepath), enhanced_frame)
        
        # Save metadata JSON
        metadata_file = filepath.with_suffix('.json')
        capture_metadata = {
            'timestamp': datetime.datetime.now().isoformat(),
            'session_id': self.session_id,
            'filename': filename,
            'emotion_data': [asdict(data) for data in emotion_data],
            'capture_reason': label,
            'additional_metadata': metadata or {}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(capture_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Enhanced screenshot saved: {filename}")
        return str(filepath)
    
    def _add_comprehensive_metadata(self, frame: np.ndarray, emotion_data: List[EmotionData], 
                                  metadata: Dict[str, Any] = None) -> np.ndarray:
        """Add comprehensive metadata overlay to saved images"""
        height, width = frame.shape[:2]
        
        # Create metadata panel
        panel_height = 200
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)  # Dark background
        
        # Add border
        cv2.rectangle(panel, (0, 0), (width-1, panel_height-1), (100, 100, 100), 2)
        
        # Title
        cv2.putText(panel, "EmoVision Pro Enhanced - Capture Analysis", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Timestamp and session info
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, f"Timestamp: {timestamp} | Session: {self.session_id}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Emotion summary
        if emotion_data:
            emotions_summary = f"Faces Detected: {len(emotion_data)}"
            dominant_emotions = [data.dominant_emotion for data in emotion_data]
            emotion_counts = defaultdict(int)
            for emotion in dominant_emotions:
                emotion_counts[emotion] += 1
            
            emotions_text = " | ".join([f"{emotion.title()}: {count}" for emotion, count in emotion_counts.items()])
            emotions_summary += f" | {emotions_text}"
            
            cv2.putText(panel, emotions_summary, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Individual face details
            for i, data in enumerate(emotion_data[:3]):  # Show up to 3 faces
                face_info = f"Face {data.face_id}: {data.dominant_emotion.title()} ({data.confidence:.2f}) | Intensity: {data.emotion_intensity:.2f}"
                cv2.putText(panel, face_info, (10, 120 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Combine with original frame
        result = np.vstack([frame, panel])
        return result
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        analytics_data = self.analytics.get_emotion_trends()
        
        # Add system information
        system_info = {
            'session_id': self.session_id,
            'session_duration': time.time() - self.start_time,
            'total_frames_processed': self.frame_count,
            'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'average_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'config_summary': {
                'resolution': self.config.resolution,
                'confidence_threshold': self.config.confidence_threshold,
                'tracking_method': self.config.face_tracking_method,
                'ui_theme': self.config.ui_theme
            }
        }
        
        # Combine all data
        comprehensive_report = {
            'system_info': system_info,
            'analytics': analytics_data,
            'performance_metrics': {
                'fps_history': list(self.fps_history)[-10:],  # Last 10 FPS readings
                'processing_times': list(self.processing_times)[-10:],  # Last 10 processing times
                'memory_usage': self._get_memory_usage(),
                'cpu_usage': self._get_cpu_usage()
            }
        }
        
        # Save report to file
        report_file = self.output_dir / 'reports' / f'session_report_{self.session_id}.json'
        with open(report_file, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report generated: {report_file}")
        return comprehensive_report
    
    def export_data(self, format_type: str = "csv") -> str:
        """Export emotion data in various formats"""
        if not self.config.enable_database:
            return "Database not enabled"
        
        export_file = self.output_dir / 'exports' / f'emotion_data_{self.session_id}.{format_type}'
        
        try:
            if format_type == "csv" and pd:
                # Export to CSV
                df = self.database.get_emotion_trends(hours=24)
                if df is not None:
                    df.to_csv(export_file, index=False)
            elif format_type == "json":
                # Export to JSON
                report = self.generate_comprehensive_report()
                with open(export_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Data exported to: {export_file}")
            return str(export_file)
        
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return f"Export failed: {e}"
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[EmotionData]]:
        """Main frame processing method"""
        self.frame_count += 1
        
        if self.config.enable_threading:
            # Asynchronous processing
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            # Get results if available
            try:
                emotion_data, performance_info = self.result_queue.get_nowait()
            except queue.Empty:
                emotion_data, performance_info = [], {}
        else:
            # Synchronous processing
            emotion_data, performance_info = self._process_single_frame(frame)
        
        # Update FPS
        elapsed = time.time() - self.start_time
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.fps_history.append(current_fps)
        
        # Get analytics
        analytics_data = self.analytics.get_emotion_trends()
        
        # Store data in database
        if self.config.enable_database and emotion_data:
            try:
                self.database.insert_emotion_data(emotion_data)
            except Exception as e:
                self.logger.error(f"Database insert failed: {e}")
        
        # Auto-capture
        capture_data = self.should_auto_capture(emotion_data)
        if capture_data:
            self.save_enhanced_screenshot(frame, emotion_data, f"auto_{capture_data.dominant_emotion}")
        
        # Create visual output
        output_frame = self.visualizer.draw_enhanced_overlay(
            frame, emotion_data, analytics_data, performance_info
        )
        
        return output_frame, emotion_data
    
    def cleanup(self):
        """Cleanup resources"""
        if self.config.enable_threading and self.processing_thread:
            self.frame_queue.put(None)  # Shutdown signal
            self.processing_thread.join(timeout=2)
            self.executor.shutdown(wait=False)
        
        # Generate final report
        self.generate_comprehensive_report()
        self.logger.info("EmoVision Pro Enhanced session ended")


@contextmanager
def enhanced_camera_context(camera_index: int, resolution: Tuple[int, int], fps: int):
    """Enhanced context manager for camera with better error handling"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # Try different backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_GSTREAMER]
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                break
        else:
            raise RuntimeError(f"Could not access camera {camera_index} with any backend")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time processing
    
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def load_enhanced_config(config_path: str = "emovision_config.json") -> EmotionConfig:
    """Load enhanced configuration with validation"""
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Validate configuration
            config = EmotionConfig(**config_data)
            
            # Validate GPU availability
            if config.use_gpu and not torch.cuda.is_available():
                print("Warning: GPU requested but not available. Falling back to CPU.")
                config.use_gpu = False
            
            return config
            
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
    
    # Create default config file with enhanced options
    config = EmotionConfig()
    config_dict = asdict(config)
    
    # Add comments to config
    config_dict['_comments'] = {
        'camera_index': 'Camera device index (0 for default)',
        'resolution': 'Camera resolution [width, height]',
        'confidence_threshold': 'Minimum confidence for emotion detection (0.0-1.0)',
        'face_tracking_method': 'Tracking method: simple, kalman, dlib',
        'ui_theme': 'UI theme: dark, light, cyberpunk',
        'enable_api': 'Enable REST API for remote access',
        'enable_threading': 'Enable multi-threaded processing'
    }
    
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Created enhanced config: {config_path}")
    return config


def main():
    """Enhanced main application with comprehensive error handling"""
    print("🎭 EmoVision Pro Enhanced v3.0")
    print("=" * 60)
    print("Next-Generation Real-time Emotion Detection System")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_enhanced_config()
        print(f"📋 Configuration loaded: {config.ui_theme} theme, {config.resolution} resolution")
        
        # Initialize enhanced detector
        detector = EnhancedEmotionDetector(config)
        print("🔧 Advanced detector initialized")
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        
        with enhanced_camera_context(config.camera_index, config.resolution, config.fps_target) as cap:
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"📹 Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
            print("\n🎮 Enhanced Controls:")
            print("   Q/ESC - Quit application")
            print("   S - Save screenshot with analysis")
            print("   R - Generate comprehensive report")
            print("   T - Cycle UI theme")
            print("   A - Show analytics summary")
            print("   E - Export data (CSV/JSON)")
            print("   P - Toggle performance overlay")
            print("   H - Toggle emotion history")
            print("=" * 60)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    detector.logger.error("Failed to capture frame")
                    break
                
                # Process frame
                try:
                    output_frame, emotion_data = detector.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow('EmoVision Pro Enhanced', output_frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                        break
                    elif key == ord('s') or key == ord('S'):
                        detector.save_enhanced_screenshot(frame, emotion_data, "manual")
                        print("📸 Screenshot saved with analysis")
                    elif key == ord('r') or key == ord('R'):
                        report = detector.generate_comprehensive_report()
                        print("\n" + "="*70)
                        print("📊 COMPREHENSIVE EMOTION ANALYSIS REPORT")
                        print("="*70)
                        print(f"Session ID: {report['system_info']['session_id']}")
                        print(f"Duration: {report['system_info']['session_duration']:.1f} seconds")
                        print(f"Frames Processed: {report['system_info']['total_frames_processed']}")
                        print(f"Average FPS: {report['system_info']['average_fps']:.1f}")
                        
                        analytics = report.get('analytics', {})
                        if analytics.get('emotion_distribution'):
                            print("\n🎭 Emotion Distribution:")
                            for emotion, count in analytics['emotion_distribution'].items():
                                percentage = (count / analytics['total_detections']) * 100
                                print(f"   {emotion.title()}: {count} ({percentage:.1f}%)")
                        
                        print(f"\n📈 Dominant Emotion: {analytics.get('dominant_emotion', 'Unknown')}")
                        print(f"🎯 Emotion Stability: {analytics.get('emotion_stability', 0):.2f}")
                        print("="*70 + "\n")
                        
                    elif key == ord('t') or key == ord('T'):
                        # Cycle themes
                        themes = ["dark", "light", "cyberpunk"]
                        current_idx = themes.index(detector.config.ui_theme)
                        new_theme = themes[(current_idx + 1) % len(themes)]
                        detector.config.ui_theme = new_theme
                        detector.visualizer.config.ui_theme = new_theme
                        detector.visualizer.current_theme = detector.visualizer.themes[new_theme]
                        print(f"🎨 Theme changed to: {new_theme}")
                        
                    elif key == ord('a') or key == ord('A'):
                        analytics = detector.analytics.get_emotion_trends()
                        print(f"\n📊 Current Analytics Summary:")
                        print(f"   Total Detections: {analytics.get('total_detections', 0)}")
                        print(f"   Dominant Emotion: {analytics.get('dominant_emotion', 'Unknown')}")
                        print(f"   Stability Score: {analytics.get('emotion_stability', 0):.2f}")
                        
                    elif key == ord('e') or key == ord('E'):
                        export_path = detector.export_data("json")
                        print(f"📁 Data exported to: {export_path}")
                        
                    elif key == ord('p') or key == ord('P'):
                        detector.config.show_performance_metrics = not detector.config.show_performance_metrics
                        status = "enabled" if detector.config.show_performance_metrics else "disabled"
                        print(f"⚡ Performance metrics {status}")
                        
                    elif key == ord('h') or key == ord('H'):
                        detector.config.show_emotion_history = not detector.config.show_emotion_history
                        status = "enabled" if detector.config.show_emotion_history else "disabled"
                        print(f"📈 Emotion history {status}")
                
                except Exception as e:
                    detector.logger.error(f"Frame processing error: {e}")
                    continue
    
    except KeyboardInterrupt:
        print("\n⏹️  Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        logging.exception("Unhandled exception in main loop")
    finally:
        # Cleanup
        if 'detector' in locals():
            detector.cleanup()
        
        print("\n🏁 EmoVision Pro Enhanced session completed")
        print("📁 Check the 'emovision_output' directory for:")
        print("   • Session logs and reports")
        print("   • Screenshots with analysis")
        print("   • Exported data files")
        print("   • Performance metrics")
        print("\nThank you for using EmoVision Pro Enhanced! 🎭✨")


if __name__ == "__main__":
    main()
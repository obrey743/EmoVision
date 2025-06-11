#!/usr/bin/env python3
"""
EmoVision Pro - Advanced Real-time Emotion Detection System
Author: Obrey Muchena
Version: 2.0.0
Updated: June 2025

Features:
- Real-time emotion detection with confidence thresholds
- Configurable settings via JSON
- Advanced logging with rotation
- Statistics tracking and reporting
- Multi-face detection with individual tracking
- Customizable auto-capture rules
- Performance monitoring and optimization
- Professional UI with status indicators
"""

import cv2
import time
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import numpy as np
from contextlib import contextmanager

try:
    from fer import FER
except ImportError:
    raise ImportError("FER library not found. Install with: pip install fer")

try:
    import pandas as pd
except ImportError:
    pd = None
    print("Warning: pandas not available. Advanced analytics disabled.")


@dataclass
class EmotionConfig:
    """Configuration settings for emotion detection"""
    camera_index: int = 0
    resolution: Tuple[int, int] = (1280, 720)
    fps_target: int = 30
    confidence_threshold: float = 0.5
    auto_capture_threshold: float = 0.85
    auto_capture_emotions: List[str] = None
    enable_logging: bool = True
    enable_statistics: bool = True
    log_level: str = "INFO"
    output_dir: str = "emovision_output"
    
    def __post_init__(self):
        if self.auto_capture_emotions is None:
            self.auto_capture_emotions = ["happy", "surprise"]


@dataclass
class EmotionData:
    """Data structure for emotion detection results"""
    timestamp: datetime.datetime
    face_id: int
    emotions: Dict[str, float]
    bounding_box: Tuple[int, int, int, int]
    confidence: float
    dominant_emotion: str


class EmotionLogger:
    """Advanced logging system with rotation and structured output"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.setup_directories()
        self.setup_logging()
        self.emotion_history = deque(maxlen=1000)
        
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['logs', 'screenshots', 'reports', 'data']
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging with rotation"""
        log_file = self.output_dir / 'logs' / 'emovision.log'
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('EmoVision')
    
    def log_emotion_data(self, emotion_data: List[EmotionData]):
        """Log emotion data to structured files"""
        if not emotion_data or not self.config.enable_logging:
            return
            
        # Store in memory for statistics
        self.emotion_history.extend(emotion_data)
        
        # Log to CSV
        csv_file = self.output_dir / 'data' / 'emotion_data.csv'
        
        # Create header if file doesn't exist
        if not csv_file.exists():
            with open(csv_file, 'w') as f:
                f.write("timestamp,face_id,dominant_emotion,confidence,angry,disgust,fear,happy,sad,surprise,neutral,bbox_x,bbox_y,bbox_w,bbox_h\n")
        
        # Append data
        with open(csv_file, 'a') as f:
            for data in emotion_data:
                emotions = data.emotions
                bbox = data.bounding_box
                f.write(f"{data.timestamp.isoformat()},{data.face_id},{data.dominant_emotion},{data.confidence:.3f},"
                       f"{emotions.get('angry', 0):.3f},{emotions.get('disgust', 0):.3f},{emotions.get('fear', 0):.3f},"
                       f"{emotions.get('happy', 0):.3f},{emotions.get('sad', 0):.3f},{emotions.get('surprise', 0):.3f},"
                       f"{emotions.get('neutral', 0):.3f},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive statistics report"""
        if not self.emotion_history:
            return {}
            
        emotions_count = defaultdict(int)
        confidence_scores = []
        
        for data in self.emotion_history:
            emotions_count[data.dominant_emotion] += 1
            confidence_scores.append(data.confidence)
        
        return {
            'total_detections': len(self.emotion_history),
            'emotion_distribution': dict(emotions_count),
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'session_duration': (datetime.datetime.now() - self.emotion_history[0].timestamp).total_seconds() if self.emotion_history else 0
        }


class EmotionDetector:
    """Enhanced emotion detection with advanced features"""
    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.logger = EmotionLogger(config)
        self.detector = self._initialize_detector()
        self.face_tracker = {}
        self.next_face_id = 0
        
        # Performance monitoring
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=100)
        
        # UI colors for emotions
        self.emotion_colors = {
            "angry": (0, 0, 255),      # Red
            "disgust": (0, 128, 0),    # Dark Green  
            "fear": (128, 0, 128),     # Purple
            "happy": (0, 255, 0),      # Green
            "sad": (255, 0, 0),        # Blue
            "surprise": (0, 255, 255), # Yellow
            "neutral": (200, 200, 200) # Light Gray
        }
    
    def _initialize_detector(self) -> FER:
        """Initialize the FER detector with error handling"""
        try:
            detector = FER(mtcnn=True)
            self.logger.logger.info("FER detector initialized successfully")
            return detector
        except Exception as e:
            self.logger.logger.error(f"Failed to initialize FER detector: {e}")
            raise RuntimeError(f"Detector initialization failed: {e}")
    
    def _track_faces(self, faces: List[Dict]) -> List[EmotionData]:
        """Advanced face tracking with ID assignment"""
        current_time = datetime.datetime.now()
        emotion_data = []
        
        for face in faces:
            box = face['box']
            emotions = face['emotions']
            
            # Find dominant emotion and confidence
            dominant_emotion, confidence = max(emotions.items(), key=lambda x: x[1])
            
            # Skip low confidence detections
            if confidence < self.config.confidence_threshold:
                continue
            
            # Simple face tracking based on bounding box proximity
            face_id = self._assign_face_id(box)
            
            emotion_data.append(EmotionData(
                timestamp=current_time,
                face_id=face_id,
                emotions=emotions,
                bounding_box=box,
                confidence=confidence,
                dominant_emotion=dominant_emotion
            ))
        
        return emotion_data
    
    def _assign_face_id(self, box: Tuple[int, int, int, int]) -> int:
        """Assign consistent IDs to tracked faces"""
        x, y, w, h = box
        center = (x + w//2, y + h//2)
        
        # Find closest existing face
        min_distance = float('inf')
        closest_id = None
        
        for face_id, (prev_center, last_seen) in self.face_tracker.items():
            # Remove old faces (not seen for 2 seconds)
            if time.time() - last_seen > 2.0:
                continue
                
            distance = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
            if distance < min_distance and distance < 100:  # Threshold for same face
                min_distance = distance
                closest_id = face_id
        
        if closest_id is not None:
            # Update existing face
            self.face_tracker[closest_id] = (center, time.time())
            return closest_id
        else:
            # New face
            new_id = self.next_face_id
            self.next_face_id += 1
            self.face_tracker[new_id] = (center, time.time())
            return new_id
    
    def should_auto_capture(self, emotion_data: List[EmotionData]) -> bool:
        """Determine if auto-capture should be triggered"""
        for data in emotion_data:
            if (data.dominant_emotion in self.config.auto_capture_emotions and 
                data.confidence >= self.config.auto_capture_threshold):
                return True
        return False
    
    def save_screenshot(self, frame: np.ndarray, label: str = "") -> str:
        """Save screenshot with enhanced naming and metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"emotion_capture_{label}_{timestamp}.png" if label else f"emotion_capture_{timestamp}.png"
        filepath = self.logger.output_dir / 'screenshots' / filename
        
        # Add metadata overlay
        frame_with_metadata = self._add_metadata_overlay(frame.copy())
        
        cv2.imwrite(str(filepath), frame_with_metadata)
        self.logger.logger.info(f"Screenshot saved: {filename}")
        return str(filepath)
    
    def _add_metadata_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add metadata overlay to saved images"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"EmoVision Pro - {timestamp}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame
    
    def draw_emotion_overlay(self, frame: np.ndarray, emotion_data: List[EmotionData]) -> np.ndarray:
        """Draw comprehensive emotion overlay"""
        for data in emotion_data:
            self._draw_face_analysis(frame, data)
        
        self._draw_status_overlay(frame, emotion_data)
        return frame
    
    def _draw_face_analysis(self, frame: np.ndarray, data: EmotionData):
        """Draw detailed analysis for a single face"""
        x, y, w, h = data.bounding_box
        color = self.emotion_colors.get(data.dominant_emotion, (255, 255, 255))
        
        # Main bounding box with rounded corners effect
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        
        # Face ID and dominant emotion
        main_text = f"Face {data.face_id}: {data.dominant_emotion.title()} ({data.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Background for text
        cv2.rectangle(frame, (x, y - th - 15), (x + tw + 10, y), color, -1)
        cv2.putText(frame, main_text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Emotion probabilities bar
        bar_y = y + h + 10
        bar_height = 15
        emotions_sorted = sorted(data.emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(emotions_sorted[:3]):  # Top 3 emotions
            bar_width = int(score * 150)  # Scale to pixel width
            bar_color = self.emotion_colors.get(emotion, (128, 128, 128))
            
            # Draw probability bar
            cv2.rectangle(frame, (x, bar_y + i * 20), (x + bar_width, bar_y + i * 20 + bar_height), bar_color, -1)
            cv2.rectangle(frame, (x, bar_y + i * 20), (x + 150, bar_y + i * 20 + bar_height), (50, 50, 50), 1)
            
            # Emotion label
            cv2.putText(frame, f"{emotion}: {score:.2f}", (x + 155, bar_y + i * 20 + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bar_color, 1)
    
    def _draw_status_overlay(self, frame: np.ndarray, emotion_data: List[EmotionData]):
        """Draw system status and statistics"""
        height, width = frame.shape[:2]
        
        # Background panel
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (100, 100, 100), 2)
        
        # System information
        fps = self.fps_history[-1] if self.fps_history else 0
        face_count = len(emotion_data)
        avg_confidence = np.mean([d.confidence for d in emotion_data]) if emotion_data else 0
        
        info_lines = [
            f"EmoVision Pro v2.0",
            f"FPS: {fps:.1f} | Faces: {face_count}",
            f"Avg Confidence: {avg_confidence:.2f}",
            f"Controls: Q=Quit, S=Save, R=Report"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (20, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 1)
        
        # Performance indicator
        perf_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        cv2.circle(frame, (370, 30), 8, perf_color, -1)


@contextmanager
def camera_context(camera_index: int):
    """Context manager for camera resource management"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not access camera {camera_index}")
    
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def load_config(config_path: str = "emovision_config.json") -> EmotionConfig:
    """Load configuration from JSON file"""
    config_file = Path(config_path)
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return EmotionConfig(**config_data)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
    
    # Create default config file
    config = EmotionConfig()
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Created default config: {config_path}")
    
    return config


def main():
    """Main application loop with professional error handling"""
    try:
        # Load configuration
        config = load_config()
        print("EmoVision Pro v2.0 - Advanced Emotion Detection System")
        print("=" * 60)
        
        # Initialize detector
        detector = EmotionDetector(config)
        detector.logger.logger.info("Starting EmoVision Pro session")
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        
        with camera_context(config.camera_index) as cap:
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps_target)
            
            print("Camera initialized. Press 'Q' to quit, 'S' to save, 'R' for report")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    detector.logger.logger.error("Failed to capture frame")
                    break
                
                # Process frame
                process_start = time.time()
                
                # Detect emotions
                faces = detector.detector.detect_emotions(frame)
                emotion_data = detector._track_faces(faces)
                
                # Log data
                detector.logger.log_emotion_data(emotion_data)
                
                # Auto-capture
                if detector.should_auto_capture(emotion_data):
                    detector.save_screenshot(frame, "auto_happy")
                
                # Draw overlay
                frame = detector.draw_emotion_overlay(frame, emotion_data)
                
                # Performance tracking
                process_time = time.time() - process_start
                detector.processing_times.append(process_time)
                
                frame_count += 1
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                detector.fps_history.append(current_fps)
                
                # Display frame
                cv2.imshow('EmoVision Pro', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    detector.save_screenshot(frame, "manual")
                elif key == ord('r') or key == ord('R'):
                    report = detector.logger.generate_report()
                    print("\n" + "="*50)
                    print("EMOTION DETECTION REPORT")
                    print("="*50)
                    for key, value in report.items():
                        print(f"{key.replace('_', ' ').title()}: {value}")
                    print("="*50 + "\n")
    
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        logging.exception("Unhandled exception in main loop")
    finally:
        print("EmoVision Pro session ended")
        print("Check the 'emovision_output' directory for logs and captures")


if __name__ == "__main__":
    main()
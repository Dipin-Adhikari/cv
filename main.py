import cv2
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from ultralytics import YOLO

from boxmot.trackers.botsort.botsort import BotSort

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PadelAnalyzer:
    def __init__(self, yolo_model_path='yolov8n.pt', pose_model_path='pose_landmarker_lite.task'):
        #Initialize YOLO, BoxMOT (OSNet), and MediaPipe.
        print("Loading YOLOv8...")
        self.model = YOLO(yolo_model_path)
        
        print("Loading BotSort with OSNet ReID...")
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.tracker = BotSort(
            reid_weights=Path('osnet_x1_0_msmt17.pt'), 
            device=self.device,
            half=False,
            track_high_thresh=0.4,   
            track_low_thresh=0.1,    
            new_track_thresh=0.5,    
            track_buffer=60          
        )
        
        print("Loading MediaPipe PoseLandmarker...")
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = vision.PoseLandmarker
        PoseLandmarkerOptions = vision.PoseLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_poses=1  # Focuses on primary player in frame
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)
        
        self.data_log = []
        
        # Analytics Tracking
        self.shot_counts = {"Forehand": 0, "Backhand": 0, "Smash": 0}
        self.bounces = 0
        self.last_shot_type = "None"
        
        # Cooldowns and Histories for Rule-Based Logic
        self.bounce_cooldown = 0
        self.shot_cooldown = 0  
        self.ball_y_history = [] 

    def get_shot_type(self, landmarks):
        """Simple heuristic logic based on pose landmarks."""
        if not landmarks:
            return "Unknown"
        
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        r_wrist = landmarks[16]

        if r_wrist.y < nose.y:
            return "Smash"
        
        if r_wrist.x < l_shoulder.x:
            return "Backhand"
        else:
            return "Forehand"

    def process_video(self, video_path, output_path='output_result.mp4'):
        print(f"Opening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO Detection
            results = self.model(frame, classes=[0, 32, 38], conf=0.1, verbose=False) 
            
            person_dets = []
            ball_center = None
            
            if results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().item()
                    cls = int(box.cls[0].cpu().item())
                    
                    if cls == 0:  # Person
                        person_dets.append([x1, y1, x2, y2, conf, cls])
                            
                    elif cls == 32 and conf > 0.3:  # Ball
                        ball_center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, "Ball", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                    elif cls == 38 and conf > 0.3:  # Racket
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, "Racket", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if len(person_dets) == 0:
                person_dets = np.empty((0, 6))
            else:
                person_dets = np.array(person_dets)

            player_id = "Unknown"

            # OSNet ReID Tracking 
            tracks = self.tracker.update(person_dets, frame)
            
            for track in tracks:
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = track[:5]
                    player_id = int(track_id)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                    cv2.putText(frame, f"Player ID: {player_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # RULE-BASED LOGIC: BOUNCE DETECTION 
            if ball_center:
                self.ball_y_history.append(ball_center[1])
                if len(self.ball_y_history) > 5:
                    self.ball_y_history.pop(0)
                
                if len(self.ball_y_history) == 5 and self.bounce_cooldown == 0:
                    if self.ball_y_history[0] < self.ball_y_history[2] and self.ball_y_history[2] > self.ball_y_history[4]:
                        self.bounces += 1
                        self.bounce_cooldown = int(fps * 0.5) 
                        cv2.putText(frame, "BOUNCE!", (width//2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            if self.bounce_cooldown > 0:
                self.bounce_cooldown -= 1

            # SHOT DETECTION (MediaPipe Full-Frame Integration)
            current_shot = "None"
            
            # Convert frame for MediaPipe Tasks API
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect Pose on the full frame continuously
            pose_results = self.pose_landmarker.detect(mp_image)

            # Analyze Pose if detected
            if pose_results.pose_landmarks and len(pose_results.pose_landmarks) > 0:
                current_shot = self.get_shot_type(pose_results.pose_landmarks[0])

            # Update Analytics with Debouncing (Cooldown)
            shot_detected_this_frame = False
            if current_shot in self.shot_counts and current_shot != "None":
                if self.shot_cooldown == 0: 
                    self.shot_counts[current_shot] += 1
                    self.last_shot_type = current_shot
                    shot_detected_this_frame = True
                    self.shot_cooldown = int(fps * 1.5) # Wait 1.5 seconds before counting another shot
                    
                    cv2.putText(frame, f"HIT: {current_shot}", (width//2 - 100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
            
            if self.shot_cooldown > 0:
                self.shot_cooldown -= 1

            # VISUALIZE DASHBOARD 
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 20), (350, 220), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            cv2.putText(frame, f"Last Shot: {self.last_shot_type}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Forehands: {self.shot_counts['Forehand']}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Backhands: {self.shot_counts['Backhand']}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Smashes:   {self.shot_counts['Smash']}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Bounces:   {self.bounces}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # --- 6. LOG DATA ---
            self.data_log.append({
                "frame": frame_count,
                "timestamp": round(frame_count / fps, 2),
                "player_id": player_id,
                "shot_detected": shot_detected_this_frame,
                "shot_type": current_shot,
                "total_bounces_so_far": self.bounces
            })
            
            out.write(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

        cap.release()
        out.release()
        self.save_data()

    def save_data(self):
        print("Saving structured data...")
        df = pd.DataFrame(self.data_log)
        
        df.to_csv("shot_analysis.csv", index=False)
        
        with open("shot_analysis.json", "w") as f:
            json.dump(self.data_log, f, indent=4)
        
        summary_data = {
            "Total Bounces": self.bounces,
            "Shot Breakdown": self.shot_counts
        }
        with open("match_summary.json", "w") as f:
            json.dump(summary_data, f, indent=4)
        
        print("\n--- Game Analytics Summary ---")
        print(f"Total Bounces Detected: {self.bounces}")
        for shot, count in self.shot_counts.items():
            print(f"{shot}: {count}")
        print("Outputs saved: shot_analysis.csv, shot_analysis.json, match_summary.json, and the output MP4.")

if __name__ == "__main__":
    analyzer = PadelAnalyzer(
        yolo_model_path='yolov8n.pt', 
        pose_model_path='pose_landmarker_lite.task'
    )
    analyzer.process_video('input_sample_video.mp4', 'final_output_video.mp4')
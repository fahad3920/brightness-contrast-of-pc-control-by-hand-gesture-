import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
import queue
import pyaudio
import speech_recognition as sr
from collections import deque

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Kalman Filter class for smoothing
class KalmanFilter1D:
    def __init__(self, process_variance=1e-3, estimated_measurement_variance=1e-1):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

# Voice command thread
class VoiceCommandThread(threading.Thread):
    def __init__(self, command_queue):
        super().__init__()
        self.command_queue = command_queue
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = True

    def run(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        while self.running:
            with self.microphone as source:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    command = self.recognizer.recognize_google(audio).lower()
                    self.command_queue.put(command)
                except (sr.WaitTimeoutError, sr.UnknownValueError, sr.RequestError):
                    continue

    def stop(self):
        self.running = False

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def draw_slider(frame, pos, length, thickness, val, min_val, max_val, color, label):
    cv2.line(frame, pos, (pos[0] + length, pos[1]), (50, 50, 50), thickness)
    slider_pos = int((val - min_val) / (max_val - min_val) * length)
    cv2.line(frame, pos, (pos[0] + slider_pos, pos[1]), color, thickness)
    cv2.putText(frame, f"{label}: {val:.2f}" if isinstance(val, float) else f"{label}: {val}",
                (pos[0], pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def save_snapshot(frame, count):
    filename = f'snapshot_{count}.png'
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved as {filename}")

def ambient_light_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

brightness_kf = KalmanFilter1D()
contrast_kf = KalmanFilter1D()

brightness = 0
contrast = 1.0

snapshot_count = 0
last_snapshot_time = 0
snapshot_cooldown = 2  # seconds

command_queue = queue.Queue()
voice_thread = VoiceCommandThread(command_queue)
voice_thread.start()

instructions = [
    "Gesture Controls:",
    "Right Hand Thumb-Index: Adjust Brightness",
    "Left Hand Thumb-Index: Adjust Contrast",
    "Pinch Right Thumb-Middle: Save Snapshot",
    "Voice Commands: 'increase brightness', 'decrease contrast', etc.",
    "Keyboard Shortcuts:",
    "  q: Quit",
    "  b/B: Decrease/Increase Brightness",
    "  c/C: Decrease/Increase Contrast"
]

def process_voice_command(cmd, brightness, contrast):
    if 'increase brightness' in cmd:
        brightness = min(50, brightness + 5)
    elif 'decrease brightness' in cmd:
        brightness = max(-50, brightness - 5)
    elif 'increase contrast' in cmd:
        contrast = min(2.0, contrast + 0.1)
    elif 'decrease contrast' in cmd:
        contrast = max(0.5, contrast - 0.1)
    elif 'save snapshot' in cmd:
        return brightness, contrast, True
    return brightness, contrast, False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    brightness_new = brightness
    contrast_new = contrast
    save_snapshot_flag = False

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))

            dist_thumb_index = calculate_distance(thumb_pos, index_pos)
            dist_thumb_middle = calculate_distance(thumb_pos, middle_pos)

            dist_thumb_index = max(20, min(200, dist_thumb_index))
            dist_thumb_middle = max(20, min(200, dist_thumb_middle))

            norm_dist_index = (dist_thumb_index - 20) / (200 - 20)

            if i == 0:  # Right hand controls brightness
                brightness_new = int(norm_dist_index * 100) - 50
            elif i == 1:  # Left hand controls contrast
                contrast_new = 0.5 + norm_dist_index * 1.5

            brightness = brightness_kf.input_latest_noisy_measurement(brightness_new)
            contrast = contrast_kf.input_latest_noisy_measurement(contrast_new)

            if dist_thumb_middle < 40 and i == 0:
                save_snapshot_flag = True

            cv2.putText(frame, f'Brightness: {brightness:.0f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f'Contrast: {contrast:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    ambient_light = ambient_light_level(frame)
    ambient_adjustment = (ambient_light - 127) / 127 * 10  # Adjust brightness by ambient light

    brightness = max(-50, min(50, brightness + ambient_adjustment))

    adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=int(brightness))

    draw_slider(adjusted, (10, 100), 300, 10, brightness, -50, 50, (0, 255, 0), 'Brightness')
    draw_slider(adjusted, (10, 140), 300, 10, contrast, 0.5, 2.0, (0, 255, 0), 'Contrast')

    y0, dy = 180, 25
    for i, line in enumerate(instructions):
        y = y0 + i*dy
        cv2.putText(adjusted, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if save_snapshot_flag:
        current_time = time.time()
        if current_time - last_snapshot_time > snapshot_cooldown:
            save_snapshot(adjusted, snapshot_count)
            snapshot_count += 1
            last_snapshot_time = current_time

    while not command_queue.empty():
        cmd = command_queue.get()
        brightness, contrast, save_cmd = process_voice_command(cmd, brightness, contrast)
        if save_cmd:
            save_snapshot(adjusted, snapshot_count)
            snapshot_count += 1

    cv2.imshow('Live Brightness/Contrast Control - Advanced', adjusted)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        brightness = max(-50, brightness - 5)
    elif key == ord('B'):
        brightness = min(50, brightness + 5)
    elif key == ord('c'):
        contrast = max(0.5, contrast - 0.1)
    elif key == ord('C'):
        contrast = min(2.0, contrast + 0.1)

voice_thread.stop()
voice_thread.join()
cap.release()
cv2.destroyAllWindows()
hands.close()

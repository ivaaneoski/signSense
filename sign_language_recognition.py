import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math
from datetime import datetime

# --- Constants ---

# Brand Colors (BGR format for OpenCV)
COLOR_BLACK = (20, 20, 20)
COLOR_WHITE = (240, 240, 240)
COLOR_GREEN = (120, 255, 0)
COLOR_GRAY = (160, 160, 160)
COLOR_AMBER = (0, 165, 255)
COLOR_BLUE = (200, 120, 60)
COLOR_YELLOW = (0, 220, 255)

# Fonts
FONT_DISPLAY = cv2.FONT_HERSHEY_SIMPLEX
FONT_UI = cv2.FONT_HERSHEY_SIMPLEX
FONT_LABEL = cv2.FONT_HERSHEY_PLAIN
FONT_HINTS = cv2.FONT_HERSHEY_PLAIN

# Thresholds
DEBOUNCE_FRAMES = 15
MAX_HISTORY = 8

class HandClassifier:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Custom drawing spec matching guidelines
        self.landmark_spec = self.mp_draw.DrawingSpec(color=COLOR_GREEN, thickness=-1, circle_radius=4)
        self.connection_spec = self.mp_draw.DrawingSpec(color=COLOR_BLUE, thickness=2)

    def get_finger_states(self, landmarks):
        """
        Returns [thumb, index, middle, ring, pinky]
        1 for extended, 0 for folded.
        """
        states = [False] * 5
        
        # Landmark indices setup
        # Thumb: tip=4, IP=3
        # Index: tip=8, PIP=6
        # Middle: tip=12, PIP=10
        # Ring: tip=16, PIP=14
        # Pinky: tip=20, PIP=18
        
        # Fingers 2-5
        states[1] = landmarks.landmark[8].y < landmarks.landmark[6].y
        states[2] = landmarks.landmark[12].y < landmarks.landmark[10].y
        states[3] = landmarks.landmark[16].y < landmarks.landmark[14].y
        states[4] = landmarks.landmark[20].y < landmarks.landmark[18].y
        
        # Thumb (lateral movement check)
        states[0] = landmarks.landmark[4].x < landmarks.landmark[3].x
        
        return states

    def euclidean_distance(self, lm_a, lm_b):
        """Computes 2D distance between two landmarks using normalized x and y."""
        return math.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2)

    def classify(self, landmarks):
        """Rule-based classification for ASL letters."""
        states = self.get_finger_states(landmarks)
        states_int = [int(s) for s in states]
        
        lm = landmarks.landmark
        thumb_tip = lm[4]
        index_tip = lm[8]
        middle_tip = lm[12]
        index_pip = lm[6]
        wrist = lm[0]
        
        # A, T, Y
        if states_int == [1, 0, 0, 0, 0]:
            if self.euclidean_distance(thumb_tip, index_pip) < 0.05:
                return "T"
            return "A"
        
        # Y
        if states_int == [1, 0, 0, 0, 1]:
            return "Y"
            
        # [ILY]
        if states_int == [1, 1, 0, 0, 1]:
            return "[ILY]"
            
        # B
        if states_int == [0, 1, 1, 1, 1]:
            return "B"
            
        # C, E, S
        if states_int == [0, 0, 0, 0, 0]:
            if thumb_tip.x < index_tip.x:
                return "S"
            all_curved = True
            for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
                if self.euclidean_distance(lm[tip], lm[pip]) > 0.12:
                    all_curved = False
            if all_curved:
                return "C"
            return "E" # default full curl fallback
            
        # D, P, X
        if states_int == [0, 1, 0, 0, 0]:
            if index_tip.y > wrist.y:
                return "P"
            if index_tip.y > index_pip.y:
                return "X"
            if self.euclidean_distance(thumb_tip, middle_tip) < 0.06:
                return "D"
            return "D"
            
        # F
        if states_int == [1, 0, 1, 1, 1]:
            if self.euclidean_distance(thumb_tip, index_tip) < 0.05:
                return "F"
            
        # G, L
        if states_int == [1, 1, 0, 0, 0]:
            if abs(thumb_tip.x - index_tip.x) > 0.08:
                return "L"
            return "G"
            
        # H, R, U, V
        if states_int == [0, 1, 1, 0, 0]:
            if abs(index_tip.x - middle_tip.x) > 0.05:
                return "V"
            return "H" # R and U are geometrically very similar to H in 2D
            
        # I
        if states_int == [0, 0, 0, 0, 1]:
            return "I"
            
        # K
        if states_int == [1, 1, 1, 0, 0]:
            return "K"
            
        # W
        if states_int == [0, 1, 1, 1, 0]:
            return "W"
            
        # O
        if self.euclidean_distance(thumb_tip, index_tip) < 0.06:
            all_curved = True
            for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
                if self.euclidean_distance(lm[tip], lm[pip]) > 0.12:
                    all_curved = False
            if all_curved:
                return "O"

        return "?"

class LetterHistory:
    def __init__(self, debounce_frames=DEBOUNCE_FRAMES, max_history=MAX_HISTORY):
        self.debounce_frames = debounce_frames
        self.max_history = max_history
        self.history = []
        self.frame_counter = 0
        self.debounce_buffer = "?"

    def update(self, letter):
        if letter == "?" or letter == "":
            self.frame_counter = 0
            self.debounce_buffer = "?"
            return

        if letter == self.debounce_buffer:
            self.frame_counter += 1
            if self.frame_counter == self.debounce_frames:
                self.history.append(letter)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
        else:
            self.debounce_buffer = letter
            self.frame_counter = 1

    def clear(self):
        self.history = []
        self.frame_counter = 0
        self.debounce_buffer = "?"

    def get_display_string(self):
        return " ".join(self.history)

class Overlay:
    def _draw_panel(self, frame, x, y, w, h):
        """Draws a semi-transparent dark panel with a blue border."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), COLOR_BLACK, -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_BLUE, 1)

    def draw_letter_box(self, frame, letter):
        h, w, _ = frame.shape
        box_w, box_h = 160, 180
        x = w - box_w - 20
        y = 20

        self._draw_panel(frame, x, y, box_w, box_h)
        
        # Label
        cv2.putText(frame, "SIGN DETECTED", (x + 15, y + 25), FONT_UI, 0.45, COLOR_GRAY, 1)

        # Letter
        text_color = COLOR_GREEN if letter != "?" and letter != "" else COLOR_GRAY
        
        if letter == "[ILY]":
            # Adjust scaling for longer string
            cv2.putText(frame, letter, (x + 10, y + 120), FONT_DISPLAY, 1.5, text_color, 4)
        else:
            # Calculate text size to center it
            text_size = cv2.getTextSize(letter, FONT_DISPLAY, 5.0, 7)[0]
            text_x = x + (box_w - text_size[0]) // 2
            text_y = y + 140
            cv2.putText(frame, letter, (text_x, text_y), FONT_DISPLAY, 5.0, text_color, 7)

        return x, y + box_h

    def draw_finger_states(self, frame, states, box_x, box_bottom_y):
        h, w, _ = frame.shape
        panel_w, panel_h = 160, 36
        x = box_x
        y = box_bottom_y + 10

        self._draw_panel(frame, x, y, panel_w, panel_h)
        
        states_int = [int(s) for s in states]
        text = f"T:{states_int[0]} I:{states_int[1]} M:{states_int[2]} R:{states_int[3]} P:{states_int[4]}"
        cv2.putText(frame, text, (x + 10, y + 24), FONT_LABEL, 1.2, COLOR_WHITE, 1)

    def draw_status(self, frame, hand_detected):
        text = "[HAND] Hand Detected" if hand_detected else "No hand in frame"
        color = COLOR_GREEN if hand_detected else COLOR_AMBER
        
        text_size = cv2.getTextSize(text, FONT_UI, 0.7, 2)[0]
        panel_w = text_size[0] + 48
        panel_h = 40
        
        h, w, _ = frame.shape
        x = (w - panel_w) // 2
        y = 20

        self._draw_panel(frame, x, y, panel_w, panel_h)
        cv2.putText(frame, text, (x + 24, y + 27), FONT_UI, 0.7, color, 2)

    def draw_fps(self, frame, fps):
        text = f"FPS: {int(fps)}"
        # No panel, just text with 1px black outline
        cv2.putText(frame, text, (20, 40), FONT_UI, 0.65, COLOR_BLACK, 3) # Outline
        cv2.putText(frame, text, (20, 40), FONT_UI, 0.65, COLOR_WHITE, 1) # Text

    def draw_history_bar(self, frame, history_str):
        h, w, _ = frame.shape
        panel_w = 400
        panel_h = 48
        x = (w - panel_w) // 2
        y = h - panel_h - 20

        self._draw_panel(frame, x, y, panel_w, panel_h)
        
        label = "History: "
        cv2.putText(frame, label, (x + 20, y + 32), FONT_UI, 0.75, COLOR_GRAY, 2)
        
        label_w = cv2.getTextSize(label, FONT_UI, 0.75, 2)[0][0]
        cv2.putText(frame, history_str, (x + 20 + label_w, y + 32), FONT_UI, 0.75, COLOR_GREEN, 2)

    def draw_shortcuts(self, frame):
        h, w, _ = frame.shape
        panel_w = 200
        panel_h = 100
        x = w - panel_w - 20
        y = h - panel_h - 20

        self._draw_panel(frame, x, y, panel_w, panel_h)
        
        lines = [
            "Q / ESC  -> Quit",
            "SPACE    -> Pause",
            "BKSP     -> Clear history",
            "S        -> Screenshot"
        ]
        
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + 12, y + 24 + i*20), FONT_HINTS, 1.1, COLOR_GRAY, 1)

    def draw_pause_indicator(self, frame):
        h, w, _ = frame.shape
        text = "[PAUSE] PAUSED"
        text_size = cv2.getTextSize(text, FONT_UI, 0.8, 2)[0]
        panel_w = text_size[0] + 48
        panel_h = 44
        x = (w - panel_w) // 2
        y = 70

        self._draw_panel(frame, x, y, panel_w, panel_h)
        cv2.putText(frame, text, (x + 24, y + 30), FONT_UI, 0.8, COLOR_YELLOW, 2)

def main():
    print("[SignSense] Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[SignSense] ERROR: Camera not found (index 0). Check connection and try again.")
        return

    print("[SignSense] Camera initialized at {}x{}".format(
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ))

    try:
        classifier = HandClassifier()
        history = LetterHistory()
        overlay = Overlay()
    except Exception as e:
        print(f"[SignSense] ERROR: MediaPipe init failure: {e}")
        return

    print("[SignSense] MediaPipe Hands ready")
    print("[SignSense] Running - press Q to quit")

    prev_time = time.time()
    
    app_state = {
        "is_paused": False,
        "current_sign": "?",
        "finger_states": [False] * 5,
        "hand_detected": False,
        "fps": 0.0
    }

    # Ensure screenshot directory exists
    os.makedirs("./screenshots/", exist_ok=True)

    try:
        while True:
            if not app_state["is_paused"]:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Flip and convert for processing
                frame = cv2.flip(frame, 1) # Mirror
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process
                results = classifier.hands.process(rgb_frame)

                app_state["hand_detected"] = False
                app_state["current_sign"] = "?"
                app_state["finger_states"] = [False] * 5

                if results.multi_hand_landmarks:
                    app_state["hand_detected"] = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw Skeleton
                    classifier.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        classifier.mp_hands.HAND_CONNECTIONS,
                        classifier.landmark_spec,
                        classifier.connection_spec
                    )
                    
                    # Custom wrist drawing
                    wrist_coords = classifier.mp_draw._normalized_to_pixel_coordinates(
                        hand_landmarks.landmark[0].x,
                        hand_landmarks.landmark[0].y,
                        frame.shape[1], frame.shape[0]
                    )
                    if wrist_coords:
                        cv2.circle(frame, wrist_coords, 6, COLOR_WHITE, -1)

                    app_state["finger_states"] = classifier.get_finger_states(hand_landmarks)
                    app_state["current_sign"] = classifier.classify(hand_landmarks)

                history.update(app_state["current_sign"])

                # FPS Calculation
                current_time = time.time()
                app_state["fps"] = 1 / (current_time - prev_time) if current_time > prev_time else 0
                prev_time = current_time

            # Drawing Overlay
            display_frame = frame.copy() if not app_state["is_paused"] else paused_frame.copy()

            box_x, box_bottom_y = overlay.draw_letter_box(display_frame, app_state["current_sign"])
            overlay.draw_finger_states(display_frame, app_state["finger_states"], box_x, box_bottom_y)
            overlay.draw_status(display_frame, app_state["hand_detected"])
            overlay.draw_fps(display_frame, app_state["fps"])
            overlay.draw_history_bar(display_frame, history.get_display_string())
            overlay.draw_shortcuts(display_frame)

            if not app_state["hand_detected"] and not app_state["is_paused"]:
                h, w, _ = display_frame.shape
                msg = "Show your hand to the camera"
                msg_size = cv2.getTextSize(msg, FONT_UI, 0.8, 2)[0]
                cv2.putText(display_frame, msg, ((w - msg_size[0])//2, h//2), FONT_UI, 0.8, COLOR_GRAY, 2)

            if app_state["is_paused"]:
                overlay.draw_pause_indicator(display_frame)

            cv2.imshow("Sign Language Recognition", display_frame)

            # Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: # Q or ESC
                break
            elif key == ord(' '): # SPACE
                app_state["is_paused"] = not app_state["is_paused"]
                if app_state["is_paused"]:
                    paused_frame = frame.copy()
            elif key == 8: # BACKSPACE
                history.clear()
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"./screenshots/signsense_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[SignSense] Screenshot saved: {filename}")

    except KeyboardInterrupt:
        pass
    finally:
        print("[SignSense] Exiting. Goodbye.")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

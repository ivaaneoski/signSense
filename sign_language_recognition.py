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

        # Predefined prototype landmark normalized vectors for Confidence Scoring
        self.prototypes = {
            "A": self._generate_synthetic_prototype([1, 0, 0, 0, 0], thumb="side"),
            "B": self._generate_synthetic_prototype([0, 1, 1, 1, 1], thumb="tucked"),
            "C": self._generate_synthetic_prototype([0, 0, 0, 0, 0], curved=True),
            "L": self._generate_synthetic_prototype([1, 1, 0, 0, 0], thumb="out"),
            "Y": self._generate_synthetic_prototype([1, 0, 0, 0, 1], thumb="side"),
            "[ILY]": self._generate_synthetic_prototype([1, 1, 0, 0, 1], thumb="out")
        }

    def _generate_synthetic_prototype(self, states, thumb="side", curved=False):
        """Generates a synthetic 21x3 flattened array for confidence matching."""
        proto = np.zeros((21, 3))
        # Synthesize basic positions
        for i in range(1, 6): # fingers 1-5
            base_idx = i * 4 - 3 # MCP (approx)
            proto[base_idx] = [(i-3)*0.1, -0.3, 0] # Base spread
            proto[base_idx+1] = [(i-3)*0.1, -0.4, 0] # PIP
            
            is_extended = states[i-1]
            if is_extended:
                proto[base_idx+2] = [(i-3)*0.1, -0.6, 0]
                proto[base_idx+3] = [(i-3)*0.1, -0.8, 0]
            elif curved:
                proto[base_idx+2] = [(i-3)*0.1, -0.5, 0.1]
                proto[base_idx+3] = [(i-3)*0.1, -0.4, 0.2]
            else:
                proto[base_idx+2] = [(i-3)*0.1, -0.2, 0] # Folded
                proto[base_idx+3] = [(i-3)*0.1, -0.1, 0]
            
        # Adjust thumb slightly
        if thumb == "side":
            proto[4] = [0.2, -0.2, 0]
        elif thumb == "out":
            proto[4] = [0.4, -0.4, 0]
        elif thumb == "tucked":
            proto[4] = [-0.2, -0.2, 0]
            
        return proto.flatten()

    def get_finger_states(self, landmarks, handedness_label="Right"):
        """
        Returns [thumb, index, middle, ring, pinky]
        1 for extended, 0 for folded.
        """
        states = [False] * 5
        wrist = landmarks.landmark[0]
        
        # Fingers 2-5: 3D vector length check from wrist to determine extension vs fold
        states[1] = self.euclidean_distance(wrist, landmarks.landmark[8]) > self.euclidean_distance(wrist, landmarks.landmark[6])
        states[2] = self.euclidean_distance(wrist, landmarks.landmark[12]) > self.euclidean_distance(wrist, landmarks.landmark[10])
        states[3] = self.euclidean_distance(wrist, landmarks.landmark[16]) > self.euclidean_distance(wrist, landmarks.landmark[14])
        states[4] = self.euclidean_distance(wrist, landmarks.landmark[20]) > self.euclidean_distance(wrist, landmarks.landmark[18])
        
        # Thumb (lateral movement check) with mirrored logic for Left Hands
        if handedness_label == 'Left':
            states[0] = landmarks.landmark[4].x > landmarks.landmark[3].x
        else:
            states[0] = landmarks.landmark[4].x < landmarks.landmark[3].x
        
        return states

    def euclidean_distance(self, lm_a, lm_b):
        """Computes true 3D Euclidean distance between two landmarks."""
        return math.sqrt((lm_a.x - lm_b.x)**2 + (lm_a.y - lm_b.y)**2 + (lm_a.z - lm_b.z)**2)

    def extract_landmark_vector(self, landmarks):
        """Extracts normalized 63-element landmark vector for distance calculation."""
        wrist = landmarks.landmark[0]
        vec = []
        for lm in landmarks.landmark:
            vec.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
        return np.array(vec)

    def classify(self, landmarks, handedness_label="Right"):
        """Rule-based classification for ASL letters."""
        states = self.get_finger_states(landmarks, handedness_label)
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
        elif states_int == [1, 0, 0, 0, 1]:
            return "Y"
            
        # [ILY]
        elif states_int == [1, 1, 0, 0, 1]:
            return "[ILY]"
            
        # B
        elif states_int == [0, 1, 1, 1, 1]:
            return "B"
            
        # C, E, S
        elif states_int == [0, 0, 0, 0, 0]:
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
        elif states_int == [0, 1, 0, 0, 0]:
            if index_tip.y > wrist.y:
                return "P"
            if index_tip.y > index_pip.y:
                return "X"
            if self.euclidean_distance(thumb_tip, middle_tip) < 0.06:
                return "D"
            return "D"
            
        # F
        elif states_int == [1, 0, 1, 1, 1]:
            if self.euclidean_distance(thumb_tip, index_tip) < 0.05:
                return "F"
            return "?" # added explicit fallback for F failed condition
            
        # G, L
        elif states_int == [1, 1, 0, 0, 0]:
            if abs(thumb_tip.x - index_tip.x) > 0.08:
                return "L"
            return "G"
            
        # H, R, U, V
        elif states_int == [0, 1, 1, 0, 0]:
            if abs(index_tip.x - middle_tip.x) > 0.05:
                return "V"
            return "H" # R and U are geometrically very similar to H in 2D
            
        # I
        elif states_int == [0, 0, 0, 0, 1]:
            return "I"
            
        # K
        elif states_int == [1, 1, 1, 0, 0]:
            return "K"
            
        # W
        elif states_int == [0, 1, 1, 1, 0]:
            return "W"
            
        # O - wrapped in elif at the end to prevent false positives overriding earlier rules
        elif self.euclidean_distance(thumb_tip, index_tip) < 0.06:
            all_curved = True
            for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
                if self.euclidean_distance(lm[tip], lm[pip]) > 0.12:
                    all_curved = False
            if all_curved:
                return "O"

        return "?"

    def compute_confidences(self, landmarks):
        """Computes distances to prototype letters and returns top 3 matches."""
        current_vec = self.extract_landmark_vector(landmarks)
        scores = []
        for letter, proto_vec in self.prototypes.items():
            dist = np.linalg.norm(current_vec - proto_vec)
            # Normalize to a pseudo-confidence [0.0, 1.0]
            conf = max(0.0, 1.0 - (dist * 1.5))
            scores.append((letter, conf))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:3]


class LetterHistory:
    def __init__(self, debounce_frames=DEBOUNCE_FRAMES, max_history=MAX_HISTORY):
        self.debounce_frames = debounce_frames
        self.max_history = max_history
        self.history = []
        self.frame_counter = 0
        self.debounce_buffer = "?"

    def update(self, letter):
        """Updates debounce and history; returns the confirmed letter if debounced, else None."""
        if letter == "?" or letter == "":
            self.frame_counter = 0
            self.debounce_buffer = "?"
            return None

        if letter == self.debounce_buffer:
            self.frame_counter += 1
            if self.frame_counter == self.debounce_frames:
                self.history.append(letter)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
                return letter
        else:
            self.debounce_buffer = letter
            self.frame_counter = 1
            
        return None

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
        
        return y + panel_h

    def draw_confidence_panel(self, frame, scores, box_x, box_bottom_y):
        if not scores:
            return box_bottom_y
            
        panel_w, panel_h = 160, 100
        x = box_x
        y = box_bottom_y + 10
        self._draw_panel(frame, x, y, panel_w, panel_h)
        
        cv2.putText(frame, "CONFIDENCE RANK", (x + 15, y + 20), FONT_UI, 0.45, COLOR_GRAY, 1)
        
        for i, (letter, conf) in enumerate(scores):
            row_y = y + 45 + (i * 20)
            # Draw letter label
            cv2.putText(frame, f"{letter}", (x + 10, row_y), FONT_UI, 0.6, COLOR_WHITE, 1)
            # Draw bar background
            cv2.rectangle(frame, (x + 40, row_y - 12), (x + 140, row_y), COLOR_GRAY, -1)
            # Draw confidence fill
            fill_w = int(100 * conf)
            color = COLOR_GREEN if conf > 0.5 else COLOR_AMBER
            cv2.rectangle(frame, (x + 40, row_y - 12), (x + 40 + fill_w, row_y), color, -1)
            # Draw border
            cv2.rectangle(frame, (x + 40, row_y - 12), (x + 140, row_y), COLOR_BLACK, 1)
            
        return y + panel_h

    def draw_status(self, frame, hand_detected, handedness_label="Right"):
        text = f"[HAND] {handedness_label} Hand Detected" if hand_detected else "No hand in frame"
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

    def draw_history_bar(self, frame, history_str, word_builder_active=False, current_word="", sentence=""):
        h, w, _ = frame.shape
        
        if word_builder_active:
            panel_w = 600
            panel_h = 74
            x = (w - panel_w) // 2
            y = h - panel_h - 20
            self._draw_panel(frame, x, y, panel_w, panel_h)
            
            # Word builder mode UI
            cv2.putText(frame, "[WORD BUILDER MODE]", (x + 10, y + 25), FONT_UI, 0.5, COLOR_AMBER, 1)
            cv2.putText(frame, "Word: ", (x + 180, y + 25), FONT_UI, 0.5, COLOR_GRAY, 1)
            cv2.putText(frame, f"{current_word}_", (x + 230, y + 25), FONT_UI, 0.6, COLOR_AMBER, 2)
            
            cv2.putText(frame, "Sent: ", (x + 10, y + 60), FONT_UI, 0.6, COLOR_GRAY, 1)
            cv2.putText(frame, sentence, (x + 70, y + 60), FONT_UI, 0.6, COLOR_WHITE, 1)
            
        else:
            panel_w = 400
            panel_h = 48
            x = (w - panel_w) // 2
            y = h - panel_h - 20
            self._draw_panel(frame, x, y, panel_w, panel_h)
            
            label = "History: "
            cv2.putText(frame, label, (x + 20, y + 32), FONT_UI, 0.75, COLOR_GRAY, 2)
            
            label_w = cv2.getTextSize(label, FONT_UI, 0.75, 2)[0][0]
            cv2.putText(frame, history_str, (x + 20 + label_w, y + 32), FONT_UI, 0.75, COLOR_GREEN, 2)

    def draw_shortcuts(self, frame, word_builder_active=False):
        h, w, _ = frame.shape
        panel_w = 230
        panel_h = 135 if word_builder_active else 125
        x = w - panel_w - 20
        y = h - panel_h - 20

        self._draw_panel(frame, x, y, panel_w, panel_h)
        
        if word_builder_active:
            lines = [
                "ESC      -> Quit",
                "TAB      -> Exit Word Mode",
                "SPACE    -> Add to Sent",
                "BKSP     -> Del Letter",
                "S        -> Screenshot"
            ]
        else:
            lines = [
                "Q / ESC  -> Quit",
                "TAB      -> Word Mode",
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
    
    # Init paused_frame fix
    paused_frame = None
    
    app_state = {
        "is_paused": False,
        "current_sign": "?",
        "finger_states": [False] * 5,
        "hand_detected": False,
        "handedness_label": "",
        "fps": 0.0,
        "word_builder_active": False,
        "current_word": "",
        "sentence": "",
        "confidence_scores": []
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
                app_state["handedness_label"] = ""
                app_state["confidence_scores"] = []

                if results.multi_hand_landmarks and results.multi_handedness:
                    app_state["hand_detected"] = True
                    hand_landmarks = results.multi_hand_landmarks[0]
                    app_state["handedness_label"] = results.multi_handedness[0].classification[0].label
                    
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

                    app_state["finger_states"] = classifier.get_finger_states(hand_landmarks, app_state["handedness_label"])
                    app_state["current_sign"] = classifier.classify(hand_landmarks, app_state["handedness_label"])
                    app_state["confidence_scores"] = classifier.compute_confidences(hand_landmarks)

                confirmed_letter = history.update(app_state["current_sign"])
                
                # word builder appends dynamically if valid matching debounced block
                if confirmed_letter and app_state["word_builder_active"] and confirmed_letter != "?":
                    app_state["current_word"] += confirmed_letter

                # FPS Calculation
                current_time = time.time()
                app_state["fps"] = 1 / (current_time - prev_time) if current_time > prev_time else 0
                prev_time = current_time

            # Drawing Overlay
            if app_state["is_paused"] and paused_frame is not None:
                display_frame = paused_frame.copy()
            else:
                display_frame = frame.copy()

            box_x, box_bottom_y = overlay.draw_letter_box(display_frame, app_state["current_sign"])
            box_bottom_y = overlay.draw_finger_states(display_frame, app_state["finger_states"], box_x, box_bottom_y)
            box_bottom_y = overlay.draw_confidence_panel(display_frame, app_state["confidence_scores"], box_x, box_bottom_y)
            
            overlay.draw_status(display_frame, app_state["hand_detected"], app_state["handedness_label"])
            overlay.draw_fps(display_frame, app_state["fps"])
            overlay.draw_history_bar(
                display_frame, 
                history.get_display_string(), 
                app_state["word_builder_active"], 
                app_state["current_word"], 
                app_state["sentence"]
            )
            overlay.draw_shortcuts(display_frame, app_state["word_builder_active"])

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
            elif key == 9: # TAB
                app_state["word_builder_active"] = not app_state["word_builder_active"]
                if app_state["word_builder_active"]:
                    history.clear()
            elif key == ord(' '): # SPACE
                if app_state["word_builder_active"]:
                    if app_state["current_word"]:
                        app_state["sentence"] += app_state["current_word"] + " "
                        app_state["current_word"] = ""
                else:
                    app_state["is_paused"] = not app_state["is_paused"]
                    if app_state["is_paused"]:
                        paused_frame = frame.copy()
            elif key == 8: # BACKSPACE
                if app_state["word_builder_active"]:
                    if len(app_state["current_word"]) > 0:
                        app_state["current_word"] = app_state["current_word"][:-1]
                    elif len(app_state["sentence"]) > 0:
                        app_state["sentence"] = app_state["sentence"].strip()
                        if " " in app_state["sentence"]:
                            app_state["sentence"] = app_state["sentence"].rsplit(" ", 1)[0] + " "
                        else:
                            app_state["sentence"] = ""
                else:
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

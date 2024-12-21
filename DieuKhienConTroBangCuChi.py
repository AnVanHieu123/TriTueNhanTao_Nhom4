import cv2
import mediapipe as mp
import pyautogui
from PIL import Image, ImageDraw, ImageFont  # Thêm Pillow
import os
import numpy as np
from pyuac import main_requires_admin

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize PyAutoGUI for mouse control
pyautogui.FAILSAFE = False

# Set the screen dimensions
screen_width, screen_height = pyautogui.size()
@main_requires_admin
def detect_and_control():
    cap = cv2.VideoCapture(0)
    font_path = "arial.ttf"  # Đường dẫn tới file font chữ. Hãy đảm bảo có font hỗ trợ Unicode
    font = ImageFont.truetype(font_path, 24)  # Tải font với kích thước 24
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB for Pillow
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(img_pil)

        # Detect hands
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        action_text = "Trạng thái: Không thực hiện hành động gì"
        # Detect hands


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the landmarks
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    landmark_list.append((int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])))

                # Determine the state of the hand
                state = "Idle"
                if len(landmark_list) >= 4:
                    # Check if three fingers are raised
                    if landmark_list[4][1] < landmark_list[2][1] and landmark_list[8][1] < landmark_list[7][1] and landmark_list[12][1] < landmark_list[11][1] and landmark_list[16][1] < landmark_list[15][1] and landmark_list[20][1] < landmark_list[19][1]:
                        # Control the mouse cursor
                        x = int(landmark_list[8][0] * screen_width / frame.shape[1])
                        y = int(landmark_list[8][1] * screen_height / frame.shape[0])
                        pyautogui.moveTo(x * 1, y * 1)
                        state = "Cursor Control"
                        action_text = "Trạng thái: Điều khiển con trỏ"
                    # Check if the index finger is bent
                    elif landmark_list[8][1] > landmark_list[7][1]:
                        # Left click
                        pyautogui.click()
                        state = "Left Click"
                        action_text = "Trạng thái: Nhấp chuột trái"
                        cv2.waitKey(250)
                    # Check if the index and middle fingers are bent
                    elif landmark_list[12][1] > landmark_list[11][1]:
                        # Right click
                        pyautogui.rightClick()
                        state = "Right Click"
                        action_text = "Trạng thái: Nhấp chuột phải"
                        cv2.waitKey(500)

                    elif landmark_list[4][1] > landmark_list[3][1]:
                        os.system("start osk")  # Mở bàn phím ảo (On-Screen Keyboard)
                        state = "On-Screen Keyboard"
                        action_text = "Trạng thái: Mở bàn phím ảo"
                        cv2.waitKey(500)

                print(state)

                # Draw the hand landmarks
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for connection in mp_hands.HAND_CONNECTIONS:
                    landmark_index1 = connection[0]
                    landmark_index2 = connection[1]
                    landmark_point1 = landmark_list[landmark_index1]
                    landmark_point2 = landmark_list[landmark_index2]
                    draw.line([landmark_point1, landmark_point2], fill=(255, 0, 0), width=2)  # Màu đỏ, độ dày 2 pixel

        #Create a rectangle to display status
        draw.rectangle([(10, 10), (600, 60)], fill=(0, 255, 0))  # Green background
        draw.text((20, 15), action_text, font=font, fill=(255, 255, 255))  # White text

        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Hand Gesture Controlled Mouse", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_control()
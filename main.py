import mediapipe as mp
import cv2
import pyautogui as auto

def main():
    cam = cv2.VideoCapture(0)
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_width, screen_height = auto.size()
    while True:
        flag, img = cam.read()
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_mesh.process(rgb_frame)
        landmark_pts = out.multi_face_landmarks
        frame_height, frame_width, _ = img.shape
        if landmark_pts:
            landmarks = landmark_pts[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):  # 2D Space
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(img, (x, y), 3, (0, 255, 0))     # Face Landmarks
                if id == 1:
                    screen_x = screen_width / frame_width * x
                    screen_y = screen_height / frame_height * y
                    auto.moveTo(screen_x, screen_y)
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(img, (x, y), 3, (0, 255, 255))
            if left[0].y - left[1].y < 0.004:
                # print("Click Triggered")
                auto.click()
                auto.sleep(1)
        if not flag:
            print("Error Capturing Frames")
            break

        cv2.imshow("Webcam", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

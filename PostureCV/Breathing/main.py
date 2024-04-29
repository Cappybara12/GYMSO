import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Initialize MediaPipe Pose and Hands models
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Initialize VideoCapture
cap = cv2.VideoCapture(0)

# Initialize variables
previous_shoulder_y = 0
breath_in = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Pose
    pose_results = pose.process(frame_rgb)

    # Process the image with MediaPipe Hands
    hands_results = hands.process(frame_rgb)

    # Extract nose landmark
    if pose_results.pose_landmarks:
        nose_landmark = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

        # Get the coordinates of the nose landmark
        nose_x = int(nose_landmark.x * frame.shape[1])
        nose_y = int(nose_landmark.y * frame.shape[0])

        # Draw circle on nose landmark
        cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)

    # Check if any finger is close to the nose
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            for finger_tip in hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP:mp_hands.HandLandmark.PINKY_TIP + 1]:
                finger_tip_x = int(finger_tip.x * frame.shape[1])
                finger_tip_y = int(finger_tip.y * frame.shape[0])
                cv2.circle(frame, (finger_tip_x, finger_tip_y), 5, (0, 255, 0), -1)
                distance_to_nose = ((finger_tip_x - nose_x) ** 2 + (finger_tip_y - nose_y) ** 2) ** 0.5
                if distance_to_nose < 30:  # Adjust this threshold as needed
                    cv2.putText(frame, "Breathe In", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    breath_in = True
                    break
                else:
                    breath_in = False
                    cv2.putText(frame, "Breathe Out", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Yoga Breathing Pose Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoCapture and close all windows
cap.release()
cv2.destroyAllWindows()

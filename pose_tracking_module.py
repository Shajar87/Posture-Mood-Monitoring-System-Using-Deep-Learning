import mediapipe as mp
import time
import math
import cv2


class PoseTracker:
    def __init__(self, min_detection_confidence=0.4, min_tracking_confidence=0.4):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.mpDraw = mp.solutions.drawing_utils
     
        
    def detectPose(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and get the pose landmarks
        self.results = self.pose.process(rgb_frame)

        # Check if pose landmarks are available
        if self.results.pose_landmarks:
            # You can access individual landmarks using results.pose_landmarks.landmark[index]
            # For example, results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
       
            # Draw connections between landmarks
            self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks, 
                                       self.mp_pose.POSE_CONNECTIONS)

        return frame
    def trackPose(self, frame, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #if lm.visibility > 0.5:
                    h, w, c = frame.shape   
                    # print(id, lm)
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                #else:
                   # continue
            
        return self.lmList
    def findAngle(self, frame, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, draw=True):
        # Check if lmList is not empty and has enough elements
        if len(self.lmList) > max(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11):
            # Get the landmarks
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            x3, y3 = self.lmList[p3][1:]
            x4, y4 = self.lmList[p4][1:]
            x5, y5 = self.lmList[p5][1:]
            x6, y6 = self.lmList[p6][1:]     
            x7, y7 = self.lmList[p7][1:]
            x8, y8 = self.lmList[p8][1:]
            x9, y9 = self.lmList[p9][1:]
            x10, y10 = self.lmList[p10][1:]
            x11, y11 = self.lmList[p11][1:]
            
            # Calculate the Angle
            angle1 = math.degrees(math.atan2(y2 - y3, x2 - x3) -
                                math.atan2(y2 - y1, x2 - x1))
            angle2 = math.degrees(math.atan2(y1 - y3, x1 - x3) -
                                math.atan2(y2 - y3, x2 - x3))
            angle3 = math.degrees(math.atan2(y2 - y3, x2 - x3) -
                                math.atan2(y2 - y5, x2 - x5))
            angle4 = math.degrees(math.atan2(y4 - y3, x4 - x3) -
                                math.atan2(y2 - y3, x2 - x3))
            angle5 = math.degrees(math.atan2(y4 - y1, x4 - x1) -
                                math.atan2(y5 - y1, x5 - x1))
            angle6 = math.degrees(math.atan2(y4 - y6, x4 - x6) -
                                math.atan2(y6 - y8, x6 - x8))
            angle7 = math.degrees(math.atan2(y5 - y7, x5 - x7) -
                                math.atan2(y7 - y9, x7 - x9))
            angle8 = math.degrees(math.atan2(y2 - y3, x2 - x3) -
                                math.atan2(y2 - y10, x2 - x10))
            angle9 = math.degrees(math.atan2(y3 - y11, x3 - x11) -
                                math.atan2(y3 - y2, x3 - x2))

            angle_list = [angle1, angle2, angle3, angle4, angle5, angle6, angle7, angle8, angle9]
            angle_list = [a + 360 if a < 0 else a for a in angle_list]

            # Draw
            if draw:
                cv2.line(frame, (x1, y1), (x3, y3), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.line(frame, (x5, y5), (x2, y2), (0, 0, 255), 2)
                cv2.line(frame, (x4, y4), (x3, y3), (0, 0, 255), 2)
                cv2.line(frame, (x4, y4), (x1, y1), (255, 0, 255), 2)
                cv2.line(frame, (x5, y5), (x1, y1), (255, 0, 255), 2)
                cv2.line(frame, (x4, y4), (x6, y6), (0, 255, 255), 2)
                cv2.line(frame, (x6, y6), (x8, y8), (0, 255, 255), 2)
                cv2.line(frame, (x5, y5), (x7, y7), (0, 255, 255), 2)
                cv2.line(frame, (x7, y7), (x9, y9), (0, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x10, y10), (255, 255, 255), 2)
                cv2.line(frame, (x3, y3), (x11, y11), (255, 255, 255), 2)



                """cv2.circle(frame, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x4, y4), 10, (0, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x5, y5), 10, (0, 0, 255), cv2.FILLED)"""

                cv2.putText(frame, str(int(angle_list[0])), (x2 - 50, y2+10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
                cv2.putText(frame, str(int(angle_list[1])), (x3 - 50, y3-50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 3)
                cv2.putText(frame, str(int(angle_list[2])), (x2 - 60, y2-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                cv2.putText(frame, str(int(angle_list[3])), (x3 + 20, y2),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
                """cv2.putText(frame, str(int(angle_list[4])), (x1-20, y1 - 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                cv2.putText(frame, str(int(angle_list[5])), (x8-40, y8 - 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
                cv2.putText(frame, str(int(angle_list[6])), (x9+40, y9 - 40),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)"""
                cv2.putText(frame, str(int(angle_list[7])), (x2+80, y2 - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
                cv2.putText(frame, str(int(angle_list[8])), (x3-80, y3 - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            # cv2.putText(frame, str(int(angle_list[5])), (x1-20, y1+10),
                    #  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
            return angle_list
        else:
        # Handle the case where lmList is not populated enough
         return []
    
    
    
def main():
    cap = cv2.VideoCapture(0)
    detector = PoseTracker()

    p_Time = 0  # Initialize ptime outside the loop
    prevPose = None
    ptime = time.time()
    pos_time = 0
    itime = 0  # Initialize total incorrect posture time
    ctime = 0  # Initialize total correct posture time
    ref_time = time.time()  # Record the time when posture changes
    sitting_time = 0

    # Set the desired window width and height
    window_width = 550
    window_height = 480

    while True:
        success, frame = cap.read()
        # Check if the video has reached its end
        if not success:
            # Set the frame position back to the beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = detector.detectPose(frame)
        lmList = detector.trackPose(frame, draw=True)
        angle_list = detector.findAngle(frame, 0, 12, 11, 8, 7, 6, 3, 5, 2, 10, 9)
        # Resize the window
        frame_resized = cv2.resize(frame, (window_width, window_height))

        if len(lmList) != 0:
            lx = (lmList[12][1] + lmList[11][1]) / 2
            ly = (lmList[12][2] + lmList[11][2]) / 2
            x, y = lmList[0][1], lmList[0][2]
            numerator = abs((lmList[11][2] - lmList[12][2]) * x - (lmList[11][1] - lmList[12][1]) * y + lmList[11][1] * lmList[12][2] - lmList[11][2] * lmList[12][1])
            denominator = math.sqrt((lmList[11][2] - lmList[12][2])**2 + (lmList[11][1] - lmList[12][1])**2)
            distance = numerator / denominator
            print(distance)
            #print(lmList[11][1])
        
            if angle_list != [0, 0]:
                # Calculate the total angle sum
                total_angle_sum = angle_list[0] + angle_list[1]

                if total_angle_sum < 75 or angle_list[0] < 20 or angle_list[1] < 20 or \
                angle_list[0] >300 or angle_list[1] > 300:# or \
                #lmList[12][1] <75 or lmList[11][1] >570:
                #distance < 130:
                
                    if prevPose == "Correct":
                        ptime = time.time()  # Start tracking incorrect posture time from now
                    prevPose = "Incorrect"
                    pos = "Incorrect Posture"
                    print(pos)
                    cv2.putText(frame_resized, pos, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                else:
                    if prevPose == "Incorrect":
                        ptime = time.time()  # Start tracking correct posture time from now
                    prevPose = "Correct"
                    pos = "Correct Posture"
                    print(pos)
                    cv2.putText(frame_resized, pos, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

                # Calculate posture time based on the difference between the current time and time when posture was detected
                pos_time = time.time() - ptime

                if pos == "Correct Posture":
                    ctime += (time.time() - ref_time)
                    ref_time = time.time()
                    cv2.putText(frame_resized, "Total Correct Posture Time: " + str(round(ctime, 1)) + " seconds", (5, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 0, 100), 1)
                else:
                    itime += (time.time() - ref_time)
                    ref_time = time.time()
                    cv2.putText(frame_resized, "Total Incorrect Posture Time: " + str(round(itime, 1)) + " seconds", (5, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 0, 100), 1)

    
                # Display posture time on the frame
                cv2.putText(frame_resized, "Posture Time: " + str(round(pos_time, 1)) + " seconds", (5, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 0, 200), 1)    
                if pos_time > 5 and pos == "Incorrect Posture":
                    cv2.putText(frame_resized, "Alert: Correct Your Sitting Posture.", (5,120), cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255),1) 
                  
                  
        #Pomodore timer
        sitting_time = itime + ctime
        cv2.putText(frame_resized, "Total Sitting Time:" + str(round(sitting_time,2)), (5,150), cv2.FONT_HERSHEY_COMPLEX,0.75,(200,100,50),1)
        if sitting_time > 500:
            cv2.putText(frame_resized, "It's rest time." + str(round(sitting_time,2)), (5,200), cv2.FONT_HERSHEY_COMPLEX,0.75,(200,100,50),1)
        # Check if lmList is not empty and if the first keypoint exists
        if lmList and len(lmList) > 0 and lmList[0] is None:
            sitting_time = 0
            itime =0
            ctime =0

            
        
        #frame rate
        c_Time = time.time()
        fps = 1 / (c_Time - p_Time)
        p_Time = c_Time

        #cv2.putText(frame_resized, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Frame", frame_resized)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

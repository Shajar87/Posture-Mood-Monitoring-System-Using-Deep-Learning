import streamlit as st
import time
import cv2
from pose_tracking_module import PoseTracker
from mood_module import Emotions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Declare global variables for time-related measurements
global ctime, itime, pos_time, ref_time, ref_time2 

def home_section():
    global ctime, itime, pos_time, ref_time, ref_time2

    st.subheader('Your Posture and Mood Insights')    

    # Initialize pose tracker
    detector = PoseTracker()

    # Initialize variables
    p_Time = 0
    prevPose = None
    ptime = time.time()
    pos_time = 0
    itime = 0
    ctime = 0
    ref_time = time.time()
    ref_time2 = time.time()
    sitting_time = 0
    pos = None
    frames = None

    # Set the desired window width and height
    window_width = 500
    window_height = 450

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create a container for the blocks
    container = st.container()

    # Create containers for display
    posture_container = st.empty()
    mood_container = st.empty()
    video_container = st.empty()
    correct_percentage_container = st.empty()
    incorrect_percentage_container = st.empty()

    # Display posture container
    with container:
        posture_container.write("Posture: ")

    # Display mood container
    with container:
        mood_container.write("Mood: ")

    # Display video container
    with container:
        video_container.write("Video Feed: ")

    # Display correct posture percentage
    with container:
        correct_percentage_container.write("Correct Posture Percentage: ")

    # Display incorrect posture percentage
    with container:
        incorrect_percentage_container.write("Incorrect Posture Percentage: ")

    # Display total sitting time
    with container:
        total_sitting_time = st.empty()

    # Display correct posture time
    with container:
        correct_posture_time = st.empty()

    # Display incorrect posture time
    with container:
        incorrect_posture_time = st.empty()

    with container:
        ipose_time = st.empty()

    # Pose alert block
    with container:
        alert_block1 = st.empty()

    # Rest alert block
    with container: 
        alert_block2 = st.empty()

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect emotions
        prediction_label, frame = Emotions.detect_emotions(frame)

        # Update mood container
        if prediction_label:
           mood_container.write(f'Mood: {prediction_label}')

        # Detect posture
        frame = detector.detectPose(frame)
        lmList = detector.trackPose(frame, draw=True)
        angle_list = detector.findAngle(frame, 0, 12, 11, 8, 7, 6, 3, 5, 2, 10, 9)

        # Resize frame
        frame_resized = cv2.resize(frame, (window_width, window_height))

        # Display video feed
        video_container.image(frame_resized, channels='BGR', width=480)

        # Process posture
        if len(lmList) != 0:
            if angle_list != [0, 0]:
                total_angle_sum = angle_list[0] + angle_list[1]

                if total_angle_sum < 75 or angle_list[0] < 20 or angle_list[1] < 20 or \
                        angle_list[0] > 300 or angle_list[1] > 300:
                    pos = "Incorrect Posture"
                else:
                    pos = "Correct Posture"
                
                posture_container.write(pos)

                if pos == "Correct Posture":
                    ctime += (time.time() - ref_time)
                else:
                    itime += (time.time() - ref_time)

                ref_time = time.time()

        pos_time += time.time() - ref_time2
        ipose_time.write(pos_time)
        ref_time2 = time.time()
        if pos == "Correct Posture":
            pos_time = 0

        # Incorrect posture message
        incorrect_posture_message = ""
        if pos_time > 5 and pos == "Incorrect Posture":
            incorrect_posture_message = "<span style='color:red'>Alert: Please Correct Your Posture.</span>"
            # st.warning(incorrect_posture_message)
        if pos == "Correct Posture":
                incorrect_posture_message = ""
        alert_block1.write(incorrect_posture_message, unsafe_allow_html=True)

        correct_posture_time.write(f'Correct Posture Time: {ctime} seconds')
        incorrect_posture_time.write(f'Incorrect Posture Time: {itime} seconds')

        # Calculate total sitting time based on updated values
        sitting_time = itime + ctime
        total_sitting_time.write(f'Total Sitting Time: {sitting_time} seconds')

        # Calculate percentages
        total_time_percentage = 100
        correct_percentage = (ctime / sitting_time) * 100
        incorrect_percentage = (itime / sitting_time) * 100

        # Display correct posture percentage
        correct_percentage_container.write(f'Correct Posture Percentage: {correct_percentage:.2f}%')

        # Display incorrect posture percentage
        incorrect_percentage_container.write(f'Incorrect Posture Percentage: {incorrect_percentage:.2f}%')

        rest_time_message = ""
        if sitting_time > 10:
            # st.info("It's rest time.")
            rest_time_message = "<span style='color:red'>Alert: Please Take Rest.</span>"

        if lmList is None or len(lmList) == 0:
            sitting_time = 0
            itime = 0
            ctime = 0
            rest_time_message = ""  # Reset the rest time message when conditions are met
        # Display the rest time message
        alert_block2.write(rest_time_message, unsafe_allow_html=True)


        # Frame rate calculation
        c_Time = time.time()
        fps = 1 / (c_Time - p_Time)
        p_Time = c_Time

        # Display frame rate on the resized frame
        cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 10), 2)

        # Display the resized frame
        cv2.imshow("Frame", frame_resized)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames, frame_resized, ctime, itime

def contact_us_section():
    st.title('Contact Us')

    # Add form for reaching out
    name = st.text_input('Name')
    email = st.text_input('Email')
    message = st.text_area('Message')
    submit_button = st.button('Submit')

    if submit_button:
        # Process the form submission
        # You can add your logic here to send the message or store it in a database
        st.success('Message Sent!')

def posture_insights(ctime, itime):
    st.header('Posture Insights')

    # Create a pie chart for correct and incorrect posture time
    posture_data = {'Category': ['Correct Posture Time', 'Incorrect Posture Time'],
                    'Time': [ctime, itime]}  # You may need to adjust the data source based on your implementation

    # Create a DataFrame from the posture data
    df = pd.DataFrame(posture_data)

    # Plot the posture insights using a pie chart
    fig, ax = plt.subplots()
    ax.pie(df['Time'], labels=df['Category'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the pie chart in the Streamlit app
    st.pyplot(fig)

def main():
    st.title('Posture and Mood Monitoring App')

    # Add a navigation bar
    nav_selection = st.sidebar.radio('Navigation', ['Home', 'Contact Us', 'Posture Insights'])
    global ctime,itime
    itime = 0
    ctime = 0
    # Display sections based on navigation selection
    if nav_selection == 'Home':
        _, _, ctime, itime = home_section()
        print(ctime)
    elif nav_selection == 'Contact Us':
        contact_us_section()
    elif nav_selection == 'Posture Insights':
        posture_insights(ctime, itime)

if __name__ == '__main__':
    main()
import streamlit as st
import time
import cv2
from posture_module import PoseTracker
from mood_module import Emotions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Declare global variables for time-related measurements
st.session_state.setdefault('ctime', 0)
st.session_state.setdefault('itime', 0)
st.session_state.setdefault('pos_time', 0)
st.session_state.setdefault('ref_time', time.time())
st.session_state.setdefault('ref_time2', time.time())
st.session_state.setdefault('p_Time', 0)
st.session_state.setdefault('sitting_time', 0)
st.session_state.setdefault('pos', None)
st.session_state.setdefault('frames', None)


def home_section():
    st.subheader('Your Posture and Mood Insights')

    # Initialize pose tracker
    detector = PoseTracker()

    # Initialize emotions detector
    emotions_detector = Emotions()

    # Set the desired window width and height
    window_width = 500
    window_height = 450

    # Initialize variables
    p_Time = st.session_state.p_Time
    pos = st.session_state.pos
    pos_time = st.session_state.pos_time

    # Create a container for the blocks
    container = st.container()

    # Create containers for display
    posture_container = st.empty()
    mood_container = st.empty()
    alert_block1 = st.empty()
    alert_block2 = st.empty()
    video_container = st.empty()
    correct_percentage_container = st.empty()
    incorrect_percentage_container = st.empty()
    total_sitting_time = st.empty()
    correct_posture_time = st.empty()
    incorrect_posture_time = st.empty()
    ipose_time_container = st.empty()  # Changed variable name to avoid confusion

    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()

        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Detect mood
        prediction_label, frame = emotions_detector.detect_emotions(frame)

        # Update mood container
        if prediction_label:
            mood_container.write(f'Mood: {prediction_label}')

        # Detect posture
        frame = detector.detectPose(frame)
        lmList = detector.trackPose(frame, draw=True)
        angle_list = detector.findAngle(frame, 0, 12, 11, 8, 7)

        # Resize frame
        frame_resized = cv2.resize(frame, (window_width, window_height))

        # Display video feed
        video_container.image(frame_resized, channels='BGR', width=480)

        # Process posture
        if len(lmList) != 0:
            if angle_list != [0, 0]:
                total_angle_sum = angle_list[0] + angle_list[1]

                if total_angle_sum < 75 or angle_list[0] < 25 or angle_list[1] < 25 or \
                        angle_list[0] > 300 or angle_list[1] > 300:
                    pos = "Incorrect Posture"
                else:
                    pos = "Correct Posture"

                posture_container.write(pos)

                if pos == "Correct Posture":
                    st.session_state.ctime += (time.time() - st.session_state.ref_time)
                else:
                    st.session_state.itime += (time.time() - st.session_state.ref_time)

                st.session_state.ref_time = time.time()

        st.session_state.pos_time += time.time() - st.session_state.ref_time2
        ipose_time = round(st.session_state.pos_time, 2)  # Correctly assigning ipose_time here
        ipose_time_container.write(ipose_time)  # Writing ipose_time to container
        st.session_state.ref_time2 = time.time()
        if pos == "Correct Posture":
            st.session_state.pos_time = 0

        # Incorrect posture message
        incorrect_posture_message = ""
        if st.session_state.pos_time > 5 and pos == "Incorrect Posture":
            incorrect_posture_message = "<span style='color:red'>Alert: Please Correct Your Posture.</span>"
        if pos == "Correct Posture":
            incorrect_posture_message = ""
        alert_block1.write(incorrect_posture_message, unsafe_allow_html=True)

        # Display correct and incorrect posture time
        correct_posture_time.write(f'Correct Posture Time: {round(st.session_state.ctime, 2)} seconds')

        incorrect_posture_time.write(f'Incorrect Posture Time: {round(st.session_state.itime, 2)} seconds')

        # Calculate total sitting time based on updated values
        sitting_time = st.session_state.itime + st.session_state.ctime
        total_sitting_time.write(f'Total Sitting Time: {round(sitting_time, 2)} seconds')

        # Calculate percentages
        if sitting_time != 0:
            correct_percentage = (st.session_state.ctime / sitting_time) * 100
            correct_percentage_container.write(f'Correct Posture Percentage: {correct_percentage:.2f}%')
        if sitting_time != 0:
            incorrect_percentage = (st.session_state.itime / sitting_time) * 100
            incorrect_percentage_container.write(f'Incorrect Posture Percentage: {incorrect_percentage:.2f}%')

        # Rest alert
        rest_time_message = ""
        if sitting_time > 20:
            rest_time_message = "<span style='color:red'>Alert: It's Time to Take a Rest.</span>"
        alert_block2.write(rest_time_message, unsafe_allow_html=True)

        if lmList is None or len(lmList) == 0:
            sitting_time = 0
            st.session_state.itime = 0
            st.session_state.ctime = 0
            rest_time_message = ""
        alert_block2.write(rest_time_message, unsafe_allow_html=True)

        # Display correct and incorrect time
        ctime = round(st.session_state.ctime, 2)
        itime = round(st.session_state.itime, 2)

        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame_resized


def posture_insights(ctime, itime):
    st.subheader('Posture Insights')

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


def contact_us_section():
    st.subheader('Contact Us')

    # Add form for reaching out
    name = st.text_input('Name')
    email = st.text_input('Email')
    message = st.text_area('Message')
    submit_button = st.button('Submit')

    if submit_button:
        # Process the form submission
        # You can add your logic here to send the message or store it in a database
        st.success('Message Sent!')


def main():
    st.title('Posture and Mood Monitoring App')

    # Add a navigation bar
    nav_selection = st.sidebar.radio('Navigation', ['Home', 'Posture Insights', 'Contact Us'])

    # Display sections based on navigation selection
    if nav_selection == 'Home':
        home_section()
    elif nav_selection == 'Contact Us':
        contact_us_section()
    elif nav_selection == 'Posture Insights':
        posture_insights(st.session_state.ctime, st.session_state.itime)


if __name__ == '__main__':
    main()

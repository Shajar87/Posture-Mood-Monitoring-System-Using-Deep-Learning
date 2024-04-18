# frame rate calculation
        c_Time = time.time()
        fps = 1 / (c_Time - p_Time)
        p_Time = c_Time

        # Display frame rate on the resized frame
        cv2.putText(frame_resized, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 10), 2)

        # Display the resized frame
        cv2.imshow("Frame", frame_resized)
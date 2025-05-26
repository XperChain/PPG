# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import tempfile

st.title("📷 PPG 기반 실시간 심박수 측정 (Streamlit 데모)")

roi_radius = 50

uploaded_video = st.camera_input("👉 카메라로 짧은 영상 촬영 후 업로드 (플래시 켜주세요)")

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.getvalue())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    brightness_data_g = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        green = frame[:, :, 1]
        mask = np.zeros_like(green, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), roi_radius, 255, -1)
        mean_g = cv2.mean(green, mask=mask)[0]
        brightness_data_g.append(mean_g)
    cap.release()

    # peak detection
    min_distance = int(fps * 0.5)
    peaks, _ = find_peaks(brightness_data_g, distance=min_distance, prominence=1.5)
    duration_seconds = len(brightness_data_g) / fps
    bpm = len(peaks) * 60 / duration_seconds

    # 결과 출력
    st.markdown(f"### 💓 추정 심박수: `{bpm:.1f} bpm`")
    st.markdown(f"총 프레임 수: {len(brightness_data_g)}, 피크 개수: {len(peaks)}")

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(brightness_data_g, color='green', label='Green Brightness')
    ax.plot(peaks, [brightness_data_g[i] for i in peaks], 'ro', label='Peaks')
    ax.set_title("Green Channel Brightness & R-Peaks")
    ax.set_ylabel("Brightness (0–255)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

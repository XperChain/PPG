import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tempfile

st.title("📷 PPG 기반 실시간 심박수 측정 (Streamlit 데모)")
roi_radius = 50

# 📁 사용자 영상 업로드
uploaded_video = st.file_uploader("📹 라이트 켜고 손가락을 비춘 영상을 업로드하세요", type=["mp4"])

# 🔧 신호 처리 함수
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def bandpass_filter(data, fs, lowcut=0.75, highcut=3.0, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_video.getvalue())
        video_path = tfile.name

    # 🎥 영상 처리
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # 일부 파일에서 fps 정보가 없을 경우
        fps = 30
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

    # ✅ 신호 전처리: 필터링
    filtered = bandpass_filter(brightness_data_g, fs=fps)

    # ✅ prominence 슬라이더 설정 (기본값 = 0.3 * std)
    estimated_prominence = round(0.3 * np.std(filtered), 2)
    st.sidebar.markdown("🔧 **Peak Detection 설정**")
    user_prominence = st.sidebar.slider(
        "Peak prominence", 
        min_value=0.1, 
        max_value=5.0, 
        value=estimated_prominence, 
        step=0.1
    )

    # ⛰️ 피크 감지
    min_distance = int(fps * 0.5)
    peaks, _ = find_peaks(filtered, distance=min_distance, prominence=user_prominence)

    # 💓 심박수 계산
    duration_seconds = len(filtered) / fps
    bpm = len(peaks) * 60 / duration_seconds

    # 📊 결과 출력
    st.markdown(f"### 💓 추정 심박수: `{bpm:.1f} bpm`")
    st.markdown(f"총 프레임 수: {len(brightness_data_g)}, 감지된 피크 수: {len(peaks)}")
    st.markdown(f"사용된 prominence 값: `{user_prominence}` (추정값: `{estimated_prominence}`)")

    # 📈 시각화
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered, color='green', label='Filtered Signal')
    ax.plot(peaks, [filtered[i] for i in peaks], 'ro', label='R-Peaks')
    ax.set_title("Green Channel Brightness with Filtering & Peak Detection")
    ax.set_ylabel("Brightness")
    ax.set_xlabel("Frame Index")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

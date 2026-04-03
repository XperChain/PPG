import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tempfile
import pandas as pd

# --- 1. 신호 처리 및 ROI 탐색 함수 ---
def bandpass_filter(data, fs, lowcut=0.75, highcut=3.0, order=4):
    if len(data) < 30: return data
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_auto_roi(first_frame):
    green = first_frame[:, :, 1]
    mask = cv2.inRange(green, 100, 230)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            _, radius = cv2.minEnclosingCircle(largest)
            return (cx, cy), int(radius * 0.4)
    return (first_frame.shape[1]//2, first_frame.shape[0]//2), 50

# --- 2. 페이지 설정 ---
st.set_page_config(page_title="Advanced PPG Analyzer", layout="wide")
st.title("📷 PPG 기반 심박수 측정")

# --- 사이드바 옵션 설정 ---
st.sidebar.header("⚙️ 분석 설정")
remove_outliers = st.sidebar.checkbox("IBI 이상치 제거 사용", value=True, help="직전 박동 대비 간격이 급변(20% 이상)한 데이터를 제외합니다.")
outlier_threshold = st.sidebar.slider("이상치 판단 임계치 (%)", 10, 100, 50, step=5) if remove_outliers else 0

# --- 3. 파일 업로드 및 신호 추출 (이전과 동일) ---
uploaded_video = st.file_uploader("📹 라이트 켜고 손가락을 비춘 영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.getvalue())
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 100: fps = 30
    
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    ret, first_frame = cap.read()
    if ret:
        (cx, cy), auto_radius = get_auto_roi(first_frame)
        roi_radius = st.sidebar.slider("ROI 반지름", 10, 150, auto_radius)
        
        raw_brightness = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = 2 
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success: break
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (cx, cy), roi_radius, 255, -1)
            mean_g = cv2.mean(frame[:, :, 1], mask=mask)[0]
            raw_brightness.append(mean_g)
            progress_bar.progress(i / total_frames)
        
        cap.release()
        progress_bar.empty()
        
        # --- 4. 신호 처리 ---
        effective_fps = fps / step
        full_filtered = bandpass_filter(raw_brightness, effective_fps)
        
        exclude_sec, window_sec = 2, 10
        exclude_frames = int(exclude_sec * effective_fps)
        window_frames = int(window_sec * effective_fps)
        
        best_start, max_sqi = exclude_frames, -1
        for start in range(exclude_frames, len(full_filtered) - exclude_frames - window_frames, int(effective_fps)):
            subset = full_filtered[start : start + window_frames]
            sqi = np.std(subset)
            if sqi > max_sqi:
                max_sqi, best_start = sqi, start
        
        opt_filtered = full_filtered[best_start : best_start + window_frames]
        dyn_prom = np.std(opt_filtered) * 0.2
        min_dist = int(effective_fps * 0.4)
        peaks, _ = find_peaks(opt_filtered, distance=min_dist, prominence=dyn_prom)
        
        bpm = (len(peaks) / window_sec) * 60

        # --- 6. 시각화 ---
        col1, col2 = st.columns(2)
        col1.metric("💓 추정 심박수", f"{bpm:.1f} BPM")
        col2.metric("📏 신호 강도 (SQI)", f"{max_sqi:.2f}")

        full_time_axis = np.arange(len(raw_brightness)) / effective_fps
        start_time = best_start / effective_fps
        opt_time_axis = full_time_axis[best_start : best_start + window_frames]
        
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(full_time_axis, full_filtered, color='#bdc3c7', alpha=0.5, label='Full Signal')
        ax.plot(opt_time_axis, opt_filtered, color='#2ecc71', label='Selected Window', linewidth=2)
        peak_times = opt_time_axis[peaks]
        ax.plot(peak_times, opt_filtered[peaks], "ro", markersize=6, label='Beats')
        ax.axvspan(start_time, start_time + window_sec, color='yellow', alpha=0.1)
        ax.set_title(f"PPG Heart Rate Analysis (FPS: {fps})")
        ax.set_xlabel("Time (seconds)")
        ax.legend()
        st.pyplot(fig)

        # --- 7. 피크 세부 데이터 분석 (이상치 옵션 적용) ---
        with st.expander("📊 상세 분석 및 피크(Peak) 데이터 보기"):
            if len(peak_times) > 1:
                raw_ibi = np.diff(peak_times)
                
                # 이상치 제거 적용 여부 판단
                if remove_outliers:
                    valid_ibi = [raw_ibi[0]]
                    threshold = outlier_threshold / 100.0
                    for i in range(1, len(raw_ibi)):
                        change_rate = abs(raw_ibi[i] - valid_ibi[-1]) / valid_ibi[-1]
                        if change_rate <= threshold:
                            valid_ibi.append(raw_ibi[i])
                    display_ibi = np.array(valid_ibi)
                    outliers_count = len(raw_ibi) - len(valid_ibi)
                    st.write(f"### 💓 IBI 분석 (이상치 제거 적용: {outlier_threshold}%)")
                else:
                    display_ibi = raw_ibi
                    outliers_count = 0
                    st.write("### 💓 IBI 분석 (원본 데이터)")

                ibi_df = pd.DataFrame({"IBI(초)": display_ibi})
                ibi_df.index = ibi_df.index + 1
                ibi_df.index.name = "박동 번호"
                st.line_chart(ibi_df, y="IBI(초)")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 IBI", f"{np.mean(display_ibi):.3f}s")
                c2.metric("SDNN (변이도)", f"{np.std(display_ibi)*1000:.1f}ms")
                c3.metric("제거된 이상치", f"{outliers_count}개")

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import tempfile

# --- 1. 신호 처리 및 ROI 탐색 함수 ---
def bandpass_filter(data, fs, lowcut=0.75, highcut=3.0, order=4):
    if len(data) < 30: return data
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def get_auto_roi(first_frame):
    """첫 프레임에서 혈류 신호가 가장 잘 보일 지점을 자동 탐색"""
    green = first_frame[:, :, 1]
    # 밝기 100~230 사이(적절한 노출) 영역 마스킹
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

# --- 3. 파일 업로드 ---
uploaded_video = st.file_uploader("📹 라이트 켜고 손가락을 비춘 영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.getvalue())
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 100: fps = 30
    
    # [아이디어 4] 실시간 피드백 요소
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    # 1단계: 첫 프레임으로 ROI 자동 설정
    ret, first_frame = cap.read()
    if ret:
        (cx, cy), auto_radius = get_auto_roi(first_frame)
        st.sidebar.info(f"📍 자동 감지 ROI: ({cx}, {cy}) r={auto_radius}")
        
        # 사용자가 수동 조절 원할 경우 대비
        roi_radius = st.sidebar.slider("ROI 반지름", 10, 150, auto_radius)
        
        raw_brightness = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = 2 
        
        # 2단계: 신호 추출
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
        
        # --- 4. 신호 처리 및 최적 구간 선택 (중요: 전체 필터링 먼저) ---
        effective_fps = fps / step
        # 전체 신호를 먼저 필터링하여 위상 어긋남 방지
        full_filtered = bandpass_filter(raw_brightness, effective_fps)
        
        exclude_sec = 2 # 앞뒤 제외 시간
        window_sec = 10 # 분석 윈도우
        exclude_frames = int(exclude_sec * effective_fps)
        window_frames = int(window_sec * effective_fps)
        
        best_start, max_sqi = exclude_frames, -1
        
        # 품질 계산 및 최적 구간 탐색
        for start in range(exclude_frames, len(full_filtered) - exclude_frames - window_frames, int(effective_fps)):
            subset = full_filtered[start : start + window_frames]
            sqi = np.std(subset) # 단순 SQI (진폭 기반)
            if sqi > max_sqi:
                max_sqi, best_start = sqi, start
        
        # 최종 분석용 신호 (이미 필터링된 데이터에서 슬라이싱)
        opt_filtered = full_filtered[best_start : best_start + window_frames]
        
        # --- 5. 피크 검출 ---
        dyn_prom = np.std(opt_filtered) * 0.2
        min_dist = int(effective_fps * 0.4)
        peaks, _ = find_peaks(opt_filtered, distance=min_dist, prominence=dyn_prom)
        
        bpm = (len(peaks) / window_sec) * 60

        # --- 6. 시각화 (x축 시간 단위) ---
        col1, col2 = st.columns(2)
        col1.metric("💓 추정 심박수", f"{bpm:.1f} BPM")
        col2.metric("📏 신호 강도 (SQI)", f"{max_sqi:.2f}")

        full_time_axis = np.arange(len(raw_brightness)) / effective_fps
        start_time = best_start / effective_fps
        end_time = start_time + window_sec
        
        fig, ax = plt.subplots(figsize=(15, 5))
        # 전체 신호 (연한 색)
        ax.plot(full_time_axis, full_filtered, color='#bdc3c7', alpha=0.5, label='Full Signal')
        
        # 최적 구간 (필터링된 전체 신호 위에 덧그림 - 선이 완벽히 겹침)
        opt_time_axis = full_time_axis[best_start : best_start + window_frames]
        ax.plot(opt_time_axis, opt_filtered, color='#2ecc71', label='Selected Window', linewidth=2)
        
        # 피크 표시
        peak_times = opt_time_axis[peaks]
        ax.plot(peak_times, opt_filtered[peaks], "ro", markersize=6, label='Beats')
        
        ax.axvspan(start_time, end_time, color='yellow', alpha=0.1)
        ax.set_title(f"PPG Heart Rate Analysis (FPS: {fps})")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        st.pyplot(fig)

        # --- 7. 피크 세부 데이터 산출 ---
        peak_amplitudes = opt_filtered[peaks] # 각 피크의 높이(강도)
        
        # 3. 상세 분석 데이터 보기 확장
        with st.expander("📊 상세 분석 및 피크(Peak) 데이터 보기"):
            # 요약 지표
            m1, m2, m3 = st.columns(3)
            m1.metric("최대 피크 강도", f"{np.max(peak_amplitudes):.3f}")
            m2.metric("평균 피크 강도", f"{np.mean(peak_amplitudes):.3f}")
            m3.metric("피크 표준편차", f"{np.std(peak_amplitudes):.3f}")
            
            st.divider()
            
            # 피크 간 간격(IBI) 분석 (추가 지표)
            if len(peak_times) > 1:
                st.write("### 💓 IBI (Inter-Beat Interval) 분석")
                ibi_values = np.diff(peak_times)
                st.line_chart(ibi_values)
                st.write(f"평균 IBI: `{np.mean(ibi_values):.3f}초`, 심박 변이도(SDNN): `{np.std(ibi_values)*1000:.1f}ms`")

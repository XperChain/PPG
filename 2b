import streamlit as st
import random
import time

# 초기 세션 상태 설정
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'current' not in st.session_state:
    st.session_state.current = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'round' not in st.session_state:
    st.session_state.round = 0
if 'message' not in st.session_state:
    st.session_state.message = ""

# 숫자 업데이트 간격 및 제한 시간
INTERVAL = 5  # 초
MAX_ROUNDS = 24  # 2분 동안 5초 간격이면 24회

st.title("🧠 2-Back 작업")
st.write("현재 숫자가 **2단계 전** 숫자와 같으면 '일치', 아니면 '불일치'를 클릭하세요.")

# 숫자 표시
if time.time() - st.session_state.last_update >= INTERVAL and st.session_state.round < MAX_ROUNDS:
    new_number = random.randint(0, 9)
    st.session_state.sequence.append(new_number)
    st.session_state.current = new_number
    st.session_state.last_update = time.time()
    st.session_state.round += 1
    st.session_state.message = ""

# 현재 숫자 표시
if st.session_state.current is not None:
    st.header(f"### 👉 숫자: {st.session_state.current}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ 일치"):
            if len(st.session_state.sequence) >= 3:
                if st.session_state.sequence[-1] == st.session_state.sequence[-3]:
                    st.session_state.score += 1
                    st.session_state.message = "정답입니다! 🎯"
                else:
                    st.session_state.message = "틀렸습니다. 😢"
            else:
                st.session_state.message = "아직 비교할 수 없습니다."
    with col2:
        if st.button("❌ 불일치"):
            if len(st.session_state.sequence) >= 3:
                if st.session_state.sequence[-1] != st.session_state.sequence[-3]:
                    st.session_state.score += 1
                    st.session_state.message = "정답입니다! 🎯"
                else:
                    st.session_state.message = "틀렸습니다. 😢"
            else:
                st.session_state.message = "아직 비교할 수 없습니다."

# 피드백 메시지 및 점수 표시
st.write(st.session_state.message)
st.info(f"🔢 현재 점수: {st.session_state.score} / {st.session_state.round}")

# 종료 메시지
if st.session_state.round >= MAX_ROUNDS:
    st.success(f"✅ 테스트 종료! 총 점수: {st.session_state.score} / {MAX_ROUNDS}")

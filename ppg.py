"""PPG 심박수 측정 + XperChain LogBook 등록 (Streamlit).

실행:  streamlit run ppg_xperchain_app.py

설정 (.env, 환경변수, 또는 .streamlit/secrets.toml):
    XPER_NODE_URL                = http://ai-labs.duckdns.org:5000
    XPER_CONTRACT_ADDRESS        = C...(40 hex)        # logbook_v1 컨트랙트
    RATING_WALLET_PRIVATE_KEY_HEX= ...(64 hex)          # 수수료를 내는 서비스 지갑
    ALLOWED_PARTICIPANT_CODES    = P01,P02,...          # (선택) 허용 참가자 코드
    TX_CONFIRM_LOOKBACK_BLOCKS   = 1000                 # (선택) 확인 조회 범위

개인키는 코드에 넣지 말고 반드시 위 설정 중 하나로만 주입하세요.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scipy.signal import butter, filtfilt, find_peaks

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# 0. 설정
# ============================================================
def config(name: str, default: str = '') -> str:
    """st.secrets → 환경변수 순서로 읽는다."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # secrets.toml 이 없으면 예외가 난다
        pass
    return os.environ.get(name, default).strip()


CONTRACT_ADDR = config('XPER_CONTRACT_ADDRESS', 'Ced62f5ada9c3bb783ce71a05db811a937d9a8f19')
NODE_URL = config('XPER_NODE_URL', 'http://ai-labs.duckdns.org:5000').rstrip('/')
SCHEMA = 'heart_rate_v1'
ACCEPTED_SCHEMAS = (SCHEMA,)
TITLE = 'PPG Heart Rate'
MAX_TX_DATA_BYTES = 526          # 노드의 MAX_TX_DATA_BYTES
BPM_RANGE = (30.0, 220.0)        # 이 범위를 벗어나면 등록하지 않음
VIDEO_HASH_LEN = 16              # 영상 SHA-256 앞 16자리(중복 등록 방지용)

PARTICIPANT_RE = re.compile(r'^P\d{2,3}$')
TX_ID_RE = re.compile(r'^[a-fA-F0-9]{64}$')
ADDRESS_RE = re.compile(r'^C[a-fA-F0-9]{40}$')
WALLET_RE = re.compile(r'^W[a-fA-F0-9]{40}$')
VIDEO_HASH_RE = re.compile(rf'^[a-f0-9]{{{VIDEO_HASH_LEN}}}$')


@st.cache_resource
def get_submit_lock() -> threading.Lock:
    """Streamlit은 매 상호작용마다 스크립트를 다시 실행하므로
    nonce 직렬화용 락은 cache_resource 로 프로세스 전체에서 하나만 유지한다."""
    return threading.Lock()


# ============================================================
# 1. 신호 처리 및 ROI 탐색
# ============================================================
def bandpass_filter(data, fs, lowcut=0.75, highcut=3.0, order=4):
    data = np.asarray(data, dtype=float)
    if len(data) < 30:
        return data
    nyquist = 0.5 * fs
    low, high = lowcut / nyquist, min(highcut / nyquist, 0.99)
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
    return (first_frame.shape[1] // 2, first_frame.shape[0] // 2), 50


def _open_video(video_bytes: bytes):
    """임시 파일로 저장 후 VideoCapture를 연다 (Windows 호환을 위해 닫은 뒤 연다)."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        tfile.write(video_bytes)
    finally:
        tfile.close()
    return cv2.VideoCapture(tfile.name), tfile.name


@st.cache_data(show_spinner=False, max_entries=4)
def probe_video(video_bytes: bytes):
    cap, path = _open_video(video_bytes)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps > 100:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if not ret:
            return None
        (cx, cy), auto_radius = get_auto_roi(first_frame)
        return {'fps': float(fps), 'total_frames': total_frames,
                'center': (cx, cy), 'auto_radius': auto_radius}
    finally:
        cap.release()
        os.unlink(path)


@st.cache_data(show_spinner="영상에서 PPG 신호 추출 중...", max_entries=8)
def extract_brightness(video_bytes: bytes, cx: int, cy: int, radius: int, step: int = 2):
    """ROI 안의 녹색 채널 평균 밝기를 step 프레임마다 추출 (순차 읽기)."""
    cap, path = _open_video(video_bytes)
    try:
        values, mask, idx = [], None, 0
        while True:
            if idx % step:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                if mask is None:
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    cv2.circle(mask, (cx, cy), radius, 255, -1)
                values.append(cv2.mean(frame[:, :, 1], mask=mask)[0])
            idx += 1
        return values
    finally:
        cap.release()
        os.unlink(path)


def analyze_signal(raw_brightness, effective_fps, exclude_sec=2, window_sec=10):
    full_filtered = bandpass_filter(raw_brightness, effective_fps)
    exclude_frames = int(exclude_sec * effective_fps)
    window_frames = int(window_sec * effective_fps)
    if len(full_filtered) < 2 * exclude_frames + window_frames:
        raise ValueError(
            f'영상이 너무 짧습니다. 최소 {2 * exclude_sec + window_sec}초 이상 촬영해 주세요.')

    best_start, max_sqi = exclude_frames, -1.0
    last_start = len(full_filtered) - exclude_frames - window_frames
    for start in range(exclude_frames, last_start + 1, max(1, int(effective_fps))):
        sqi = float(np.std(full_filtered[start:start + window_frames]))
        if sqi > max_sqi:
            max_sqi, best_start = sqi, start

    opt_filtered = full_filtered[best_start:best_start + window_frames]
    dyn_prom = np.std(opt_filtered) * 0.2
    min_dist = max(1, int(effective_fps * 0.4))
    peaks, _ = find_peaks(opt_filtered, distance=min_dist, prominence=dyn_prom)
    return {
        'full_filtered': full_filtered,
        'best_start': best_start,
        'window_frames': window_frames,
        'window_sec': window_sec,
        'opt_filtered': opt_filtered,
        'peaks': peaks,
        'bpm': (len(peaks) / window_sec) * 60,
        'sqi': max_sqi,
    }


def clean_ibi(peak_times, remove_outliers: bool, threshold_pct: float):
    raw_ibi = np.diff(peak_times)
    if len(raw_ibi) == 0:
        return raw_ibi, 0
    if not remove_outliers:
        return raw_ibi, 0
    threshold = threshold_pct / 100.0
    valid = [raw_ibi[0]]
    for value in raw_ibi[1:]:
        if abs(value - valid[-1]) / valid[-1] <= threshold:
            valid.append(value)
    return np.array(valid), len(raw_ibi) - len(valid)


# ============================================================
# 2. XperChain 연동 (Flask 버전과 동일한 서명/전송 방식)
# ============================================================
class NodeError(Exception):
    """노드 요청 실패 또는 사용할 수 없는 응답."""


def node_request(method: str, path: str, **kwargs) -> dict:
    try:
        response = requests.request(method, f'{NODE_URL}{path}', timeout=8, **kwargs)
        try:
            data = response.json()
        except ValueError as exc:
            content_type = response.headers.get('Content-Type', 'unknown')
            raise NodeError(
                f'XperChain이 {method.upper()} {path} 요청에 JSON 대신 '
                f'HTTP {response.status_code} ({content_type})를 반환했습니다. XPER_NODE_URL을 확인하세요.'
            ) from exc
        if not isinstance(data, dict):
            raise NodeError(f'XperChain이 {method.upper()} {path}에 예상치 못한 JSON을 반환했습니다.')
        if response.status_code >= 400 or not data.get('ok', True):
            raise NodeError(f"노드가 요청을 거부했습니다: {data.get('error') or f'HTTP {response.status_code}'}")
        return data
    except requests.RequestException as exc:
        raise NodeError('XperChain 노드에 연결할 수 없습니다.') from exc


def service_signing_key() -> Ed25519PrivateKey:
    value = config('RATING_WALLET_PRIVATE_KEY_HEX')
    if not value:
        raise NodeError('서버 서명 지갑(RATING_WALLET_PRIVATE_KEY_HEX)이 설정되지 않았습니다.')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise NodeError('RATING_WALLET_PRIVATE_KEY_HEX 형식 오류 (32바이트 hex 필요).')
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(value))


def wallet_address(key: Ed25519PrivateKey) -> tuple[str, bytes]:
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    return 'W' + hashlib.sha256(public).hexdigest()[:40], public


def canonical_tx_bytes(tx: dict) -> bytes:
    """노드의 Transaction.canon_payload/canon_bytes 와 동일."""
    fields = ('sender', 'recipient', 'amount_xbit', 'fee_xbit', 'nonce',
              'timestamp', 'tx_type', 'data')
    payload = {field: tx[field] for field in fields}
    return json.dumps(payload, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False).encode('utf-8')


def validate_participant_id(value, enforce_allowlist: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError('참가자 코드를 입력하세요 (예: P01).')
    pid = value.strip().upper()
    if not PARTICIPANT_RE.fullmatch(pid) or int(pid[1:]) == 0:
        raise ValueError('참가자 코드는 P01, P02, ..., P999 형식이어야 합니다.')
    allowed = config('ALLOWED_PARTICIPANT_CODES')
    if enforce_allowlist and allowed and pid not in {v.strip().upper() for v in allowed.split(',')}:
        raise ValueError('이 참가자 코드는 등록이 허용되지 않았습니다.')
    return pid


def _finite(value, lo, hi, name) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} 값이 숫자가 아닙니다.')
    value = float(value)
    if not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{name} 값이 허용 범위({lo}~{hi})를 벗어났습니다.')
    return value


def validate_measurement(m) -> dict:
    """체인에 올리거나 체인에서 읽은 측정값을 검증·정규화."""
    if not isinstance(m, dict):
        raise ValueError('측정값 형식 오류')
    video = m.get('video_sha256')
    if not isinstance(video, str) or not VIDEO_HASH_RE.fullmatch(video):
        raise ValueError('영상 해시 형식 오류')
    return {
        'bpm': round(_finite(m.get('bpm'), *BPM_RANGE, 'BPM'), 1),
        'ibi_ms': round(_finite(m.get('ibi_ms'), 200, 3000, '평균 IBI'), 1),
        'sdnn_ms': round(_finite(m.get('sdnn_ms'), 0, 1000, 'SDNN'), 1),
        'beats': int(_finite(m.get('beats'), 2, 1000, '박동 수')),
        'sqi': round(_finite(m.get('sqi'), 0, 1e6, 'SQI'), 3),
        'fps': round(_finite(m.get('fps'), 1, 240, 'FPS'), 2),
        'video_sha256': video,
    }


def normalized_utc_time(value) -> str:
    if not isinstance(value, str) or len(value) > 40:
        return ''
    try:
        instant = datetime.fromisoformat(value)
        if instant.tzinfo is None:
            return ''
        return instant.astimezone(timezone.utc).isoformat(timespec='seconds')
    except ValueError:
        return ''


def build_tx_data(record: dict) -> dict:
    text = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return {'contract_addr': CONTRACT_ADDR, 'method': 'upload_text',
            'args': {'title': TITLE, 'text': text}}


def tx_data_size(tx_data: dict) -> int:
    return len(json.dumps(tx_data, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def get_contract_snapshot() -> tuple[dict, dict]:
    if not ADDRESS_RE.fullmatch(CONTRACT_ADDR):
        raise NodeError('XPER_CONTRACT_ADDRESS 설정이 올바르지 않습니다.')
    status = node_request('GET', '/status')
    if status.get('is_synced') is not True:
        raise NodeError('XperChain 노드가 아직 동기화 중입니다. 잠시 후 다시 시도하세요.')
    snapshot = node_request('GET', f'/contract/{CONTRACT_ADDR}')
    contract = snapshot.get('contract') or {}
    try:
        metadata = json.loads(contract.get('code', '{}'))
    except (TypeError, ValueError) as exc:
        raise NodeError('컨트랙트 메타데이터를 JSON으로 읽을 수 없습니다.') from exc
    if metadata.get('template') != 'logbook_v1':
        raise NodeError('지정한 컨트랙트가 logbook_v1 이 아닙니다.')
    return metadata, snapshot.get('state') or {}


def decode_heart_records(items) -> list[dict]:
    records, seen_videos = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            content = json.loads(item.get('text', ''))
            if not isinstance(content, dict) or content.get('schema') not in ACCEPTED_SCHEMAS:
                continue
            pid = validate_participant_id(content.get('participant_id'))
            m = validate_measurement(content)
            sender = str(item.get('by', ''))
            records.append({
                'participant_id': pid,
                **m,
                'submitted_at_utc': normalized_utc_time(content.get('submitted_at_utc')),
                'logbook_index': item.get('index') if isinstance(item.get('index'), int) else None,
                'block_timestamp': item.get('at') if isinstance(item.get('at'), (int, float)) else None,
                'sender_wallet': sender if WALLET_RE.fullmatch(sender) else '',
                'duplicate_video': m['video_sha256'] in seen_videos,
            })
            seen_videos.add(m['video_sha256'])
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    return records


def get_confirmed_records() -> tuple[dict, list[dict]]:
    metadata, state = get_contract_snapshot()
    return metadata, decode_heart_records(state.get('data_items', []))


def pending_video_hashes() -> set[str]:
    """아직 블록에 포함되지 않은 등록 요청 (중복 클릭 방지)."""
    mempool = node_request('GET', '/mempool').get('mempool', [])
    hashes = set()
    for tx in mempool if isinstance(mempool, list) else []:
        if not isinstance(tx, dict) or tx.get('tx_type') != 'CONTRACT_CALL':
            continue
        data = tx.get('data', {})
        if data.get('contract_addr') != CONTRACT_ADDR or data.get('method') != 'upload_text':
            continue
        try:
            payload = json.loads(data.get('args', {}).get('text', ''))
            if payload.get('schema') in ACCEPTED_SCHEMAS and isinstance(payload.get('video_sha256'), str):
                hashes.add(payload['video_sha256'])
        except (ValueError, TypeError, AttributeError):
            continue
    return hashes


def send_logbook_record(record: dict) -> str:
    key = service_signing_key()
    wallet, public = wallet_address(key)
    tx_data = build_tx_data(record)
    if tx_data_size(tx_data) > MAX_TX_DATA_BYTES:
        raise NodeError(f'측정 기록이 노드의 {MAX_TX_DATA_BYTES}바이트 제한을 초과합니다.')

    nonce_info = node_request('GET', f'/account/{wallet}/nonce')
    estimate = node_request('POST', '/fee/estimate', json={'tx_type': 'CONTRACT_CALL', 'data': tx_data})
    tx = {
        'sender': wallet,
        'recipient': CONTRACT_ADDR,
        'amount_xbit': 0,
        'fee_xbit': int(estimate['required_fee_xbit']),
        'nonce': int(nonce_info['next_nonce']),
        'timestamp': time.time(),
        'tx_type': 'CONTRACT_CALL',
        'data': tx_data,
    }
    signature = key.sign(canonical_tx_bytes(tx))
    result = node_request('POST', '/contract/call', json={
        'contract_addr': CONTRACT_ADDR,
        'method': 'upload_text',
        'args': tx_data['args'],
        'sender': tx['sender'],
        'sender_pub_b64': base64.b64encode(public).decode('ascii'),
        'sig_b64': base64.b64encode(signature).decode('ascii'),
        'value_xbit': 0,
        'fee_xbit': tx['fee_xbit'],
        'nonce': tx['nonce'],
        'timestamp': tx['timestamp'],
    })
    tx_id = str(result.get('tx_id', ''))
    if not TX_ID_RE.fullmatch(tx_id):
        raise NodeError('노드가 요청을 받았지만 올바른 트랜잭션 ID를 반환하지 않았습니다.')
    return tx_id


def submit_measurement(participant_id: str, measurement: dict) -> str:
    """검증 → 컨트랙트 확인 → 중복 확인 → 서명/전송. tx_id 반환."""
    pid = validate_participant_id(participant_id, enforce_allowlist=True)
    record = {
        'schema': SCHEMA,
        'participant_id': pid,
        **validate_measurement(measurement),
        'submitted_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    with get_submit_lock():
        metadata, confirmed = get_confirmed_records()
        if 'upload_text' not in metadata.get('methods', []):
            raise NodeError('이 컨트랙트는 upload_text 메서드를 허용하지 않습니다.')
        used = {r['video_sha256'] for r in confirmed} | pending_video_hashes()
        if record['video_sha256'] in used:
            raise ValueError('이 영상의 측정 결과는 이미 블록체인에 등록(또는 대기)되어 있습니다.')
        return send_logbook_record(record)


def tx_status(tx_id: str) -> dict:
    """mempool → 최근 블록 순으로 트랜잭션 확인."""
    if not TX_ID_RE.fullmatch(tx_id):
        raise ValueError('잘못된 트랜잭션 ID입니다.')
    pool = node_request('GET', '/mempool').get('mempool', [])
    if any(isinstance(tx, dict) and str(tx.get('hash', '')).lower() == tx_id.lower()
           for tx in pool if isinstance(pool, list)):
        return {'status': 'pending'}
    length = int(node_request('GET', '/status').get('length', 0))
    try:
        lookback = int(config('TX_CONFIRM_LOOKBACK_BLOCKS', '1000'))
    except ValueError:
        lookback = 1000
    lookback = max(1, min(lookback, 10000))
    floor, end = max(0, length - lookback), length
    while end > floor:
        start = max(floor, end - 200)
        blocks = node_request('GET', f'/chain/from/{start}', params={'limit': end - start}).get('chain', [])
        if not isinstance(blocks, list):
            raise NodeError('XperChain이 올바르지 않은 체인 응답을 반환했습니다.')
        for block in reversed(blocks):
            if not isinstance(block, dict):
                continue
            txs = block.get('transactions', [])
            if isinstance(txs, list) and any(
                    isinstance(tx, dict) and str(tx.get('hash', '')).lower() == tx_id.lower() for tx in txs):
                return {'status': 'confirmed', 'block': block.get('index')}
        end = start
    return {'status': 'unknown'}


# ============================================================
# 3. 화면
# ============================================================
STATUS_LABEL = {
    'pending': '⏳ 대기 중 (mempool) — 블록 생성 후 확정됩니다.',
    'confirmed': '✅ 블록에 포함되어 확정되었습니다.',
    'unknown': '❔ 최근 블록에서 찾지 못했습니다. 잠시 후 다시 확인하세요.',
}


def render_measure_tab(remove_outliers: bool, outlier_threshold: int):
    uploaded_video = st.file_uploader(
        "📹 라이트 켜고 손가락을 비춘 영상을 업로드하세요", type=["mp4", "mov", "avi"])
    if uploaded_video is None:
        return

    video_bytes = uploaded_video.getvalue()
    video_hash = hashlib.sha256(video_bytes).hexdigest()[:VIDEO_HASH_LEN]

    info = probe_video(video_bytes)
    if info is None:
        st.error("영상의 첫 프레임을 읽을 수 없습니다.")
        return
    fps, (cx, cy) = info['fps'], info['center']
    auto_radius = int(min(max(info['auto_radius'], 10), 150))
    roi_radius = st.sidebar.slider("ROI 반지름", 10, 150, auto_radius)

    step = 2
    raw_brightness = extract_brightness(video_bytes, cx, cy, roi_radius, step)
    effective_fps = fps / step
    try:
        result = analyze_signal(raw_brightness, effective_fps)
    except ValueError as exc:
        st.error(str(exc))
        return

    bpm, max_sqi, peaks = result['bpm'], result['sqi'], result['peaks']
    best_start, window_frames = result['best_start'], result['window_frames']
    window_sec, opt_filtered = result['window_sec'], result['opt_filtered']

    full_time_axis = np.arange(len(raw_brightness)) / effective_fps
    start_time = best_start / effective_fps
    opt_time_axis = full_time_axis[best_start:best_start + window_frames]
    peak_times = opt_time_axis[peaks]
    display_ibi, outliers_count = clean_ibi(peak_times, remove_outliers, outlier_threshold)
    # 피크 개수×6 방식은 창 경계 때문에 1박 정도 과소추정되므로, IBI가 있으면 60/평균IBI 사용
    if len(display_ibi) > 0:
        bpm = 60.0 / float(np.mean(display_ibi))

    # --- 시각화 ---
    col1, col2 = st.columns(2)
    col1.metric("💓 추정 심박수", f"{bpm:.1f} BPM",
                help=f"평균 IBI 기반. 참고: 피크 개수 기반 {result['bpm']:.1f} BPM")
    col2.metric("📏 신호 강도 (SQI)", f"{max_sqi:.2f}")

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(full_time_axis, result['full_filtered'], color='#bdc3c7', alpha=0.5, label='Full Signal')
    ax.plot(opt_time_axis, opt_filtered, color='#2ecc71', label='Selected Window', linewidth=2)
    ax.plot(peak_times, opt_filtered[peaks], "ro", markersize=6, label='Beats')
    ax.axvspan(start_time, start_time + window_sec, color='yellow', alpha=0.1)
    ax.set_title(f"PPG Heart Rate Analysis (FPS: {fps:.2f})")
    ax.set_xlabel("Time (seconds)")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # --- IBI 분석 ---
    with st.expander("📊 상세 분석 및 피크(Peak) 데이터 보기"):
        if len(display_ibi) > 0:
            title = (f"### 💓 IBI 분석 (이상치 제거 적용: {outlier_threshold}%)"
                     if remove_outliers else "### 💓 IBI 분석 (원본 데이터)")
            st.write(title)
            ibi_df = pd.DataFrame({"IBI(초)": display_ibi})
            ibi_df.index = ibi_df.index + 1
            ibi_df.index.name = "박동 번호"
            st.line_chart(ibi_df, y="IBI(초)")
            c1, c2, c3 = st.columns(3)
            c1.metric("평균 IBI", f"{np.mean(display_ibi):.3f}s")
            c2.metric("SDNN (변이도)", f"{np.std(display_ibi) * 1000:.1f}ms")
            c3.metric("제거된 이상치", f"{outliers_count}개")
        else:
            st.info("검출된 박동이 2개 미만이라 IBI를 계산할 수 없습니다.")

    # --- 블록체인 등록 ---
    st.divider()
    st.subheader("⛓️ 블록체인에 측정 결과 등록")
    if len(display_ibi) == 0:
        st.warning("박동 검출이 부족하여 등록할 수 없습니다. 영상을 다시 촬영해 주세요.")
        return

    measurement = {
        'bpm': round(float(bpm), 1),
        'ibi_ms': round(float(np.mean(display_ibi)) * 1000, 1),
        'sdnn_ms': round(float(np.std(display_ibi)) * 1000, 1),
        'beats': int(len(peaks)),
        'sqi': round(float(max_sqi), 3),
        'fps': round(float(effective_fps), 2),
        'video_sha256': video_hash,
    }
    try:
        validate_measurement(measurement)
        problem = None
    except ValueError as exc:
        problem = str(exc)

    participant_id = st.text_input("참가자 코드", placeholder="P01", max_chars=4)
    with st.expander("등록될 데이터 미리보기"):
        st.json({'schema': SCHEMA, 'participant_id': participant_id.strip().upper() or '(미입력)',
                 **measurement})
        st.caption(f"컨트랙트: `{CONTRACT_ADDR}` · 노드: `{NODE_URL}`")

    submissions = st.session_state.setdefault('submissions', {})
    already = submissions.get(video_hash)

    if problem:
        st.warning(f"측정값이 등록 기준을 벗어나 등록할 수 없습니다: {problem}")
    elif st.button("🔗 블록체인에 등록", type="primary", disabled=already is not None):
        try:
            with st.spinner("트랜잭션 서명 및 전송 중..."):
                tx_id = submit_measurement(participant_id, measurement)
            submissions[video_hash] = {'tx_id': tx_id, 'participant_id': participant_id.strip().upper()}
            already = submissions[video_hash]
            st.success("노드가 트랜잭션을 접수했습니다. 블록 확정을 기다리는 중입니다.")
        except ValueError as exc:
            st.error(str(exc))
        except NodeError as exc:
            st.error(f"블록체인 등록 실패: {exc}")

    if already:
        st.write(f"**참가자** {already['participant_id']} · **트랜잭션 ID**")
        st.code(already['tx_id'], language=None)
        if st.button("🔄 확정 상태 확인"):
            try:
                status = tx_status(already['tx_id'])
                label = STATUS_LABEL.get(status['status'], status['status'])
                if status.get('block') is not None:
                    label += f" (블록 #{status['block']})"
                st.info(label)
            except (NodeError, ValueError) as exc:
                st.error(str(exc))


def render_records_tab():
    st.caption("블록체인 LogBook에 확정된 심박 측정 기록을 조회합니다.")
    if not st.button("📥 기록 불러오기"):
        return
    try:
        with st.spinner("LogBook 조회 중..."):
            _metadata, records = get_confirmed_records()
    except NodeError as exc:
        st.error(str(exc))
        return
    if not records:
        st.info("확정된 심박 측정 기록이 없습니다.")
        return
    df = pd.DataFrame(records)
    unique = df[~df['duplicate_video']]
    c1, c2, c3 = st.columns(3)
    c1.metric("기록 수", len(unique))
    c2.metric("참가자 수", unique['participant_id'].nunique())
    c3.metric("평균 BPM", f"{unique['bpm'].mean():.1f}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "CSV 다운로드", ('﻿' + df.to_csv(index=False)).encode('utf-8'),
        file_name="heart_rate_records.csv", mime="text/csv")


def main():
    st.set_page_config(page_title="PPG Heart Rate → XperChain", layout="wide")
    st.title("📷 PPG 기반 심박수 측정 + ⛓️ 블록체인 등록")

    st.sidebar.header("⚙️ 분석 설정")
    remove_outliers = st.sidebar.checkbox(
        "IBI 이상치 제거 사용", value=True,
        help="직전 박동 대비 간격이 임계치 이상 급변한 데이터를 제외합니다.")
    outlier_threshold = (st.sidebar.slider("이상치 판단 임계치 (%)", 10, 100, 50, step=5)
                         if remove_outliers else 0)

    try:
        st.sidebar.caption(f"서비스 지갑: `{wallet_address(service_signing_key())[0]}`")
    except NodeError as exc:
        st.sidebar.warning(f"{exc} 측정은 가능하지만 등록은 되지 않습니다.")

    tab_measure, tab_records = st.tabs(["💓 측정 및 등록", "📜 블록체인 기록 조회"])
    with tab_measure:
        render_measure_tab(remove_outliers, outlier_threshold)
    with tab_records:
        render_records_tab()


if __name__ == '__main__':
    main()

"""실시간 PPG 측정 Streamlit 앱 (폰 브라우저에서 계산, 숫자만 전송).

    streamlit run ppg_realtime.py

  참가자 폰:   https://<앱 주소>/                 (Streamlit Cloud는 HTTPS 자동 제공)
  실험자 PC:   https://<앱 주소>/?page=monitor    (MONITOR_PASSWORD 필요)

동작
  - 폰: 커스텀 컴포넌트(JS)가 후면 카메라+플래시로 프레임별 평균 밝기(R, G)를 계산하고
        1초마다 Python으로 넘깁니다. 화면에는 신호 품질만 표시합니다(심박 비표시).
  - 서버: 모든 세션 데이터를 프로세스 공용 저장소(st.cache_resource)에 보관합니다.
  - 모니터: 1초마다 실시간 파형/심박/IBI/RMSSD 갱신, 블록 시작·종료, 블록 요약 블록체인 등록,
           원시 데이터·IBI·블록 CSV 다운로드.

설정 (.streamlit/secrets.toml 또는 환경변수)
  MONITOR_PASSWORD = "..."                  # 필수 (실험자 화면)
  RATING_WALLET_PRIVATE_KEY_HEX = "..."     # 블록체인 등록 시 필요
  XPER_NODE_URL, XPER_CONTRACT_ADDRESS      # 선택 (기본값: seat touch 앱과 동일)

주의: Streamlit Cloud의 디스크와 메모리는 앱이 재시작되면 지워집니다.
      세션이 끝나면 모니터 화면에서 CSV를 내려받으세요.
"""
from __future__ import annotations

import base64
import bisect
import hashlib
import io
import csv
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scipy.signal import butter, filtfilt, find_peaks, welch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def config(name: str, default: str = '') -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, default).strip()


CONTRACT_ADDR = config('XPER_CONTRACT_ADDRESS', 'Ced62f5ada9c3bb783ce71a05db811a937d9a8f19')
NODE_URL = config('XPER_NODE_URL', 'http://ai-labs.duckdns.org:5000').rstrip('/')
SCHEMA = 'ppg_block_v1'
TITLE = 'PPG Block'
MAX_TX_DATA_BYTES = 526

FS = 30.0
GAP_MS = 400.0
LIVE_WINDOW_MS = 120_000
MAX_BATCH = 1200
MAX_SESSION_SAMPLES = 400_000
MAX_SESSIONS = 30

PARTICIPANT_RE = re.compile(r'^P\d{2,3}$')
CONDITION_RE = re.compile(r'^[a-z0-9_]{1,32}$')
RUN_ID_RE = re.compile(r'^[a-zA-Z0-9]{8,32}$')
TX_ID_RE = re.compile(r'^[a-fA-F0-9]{64}$')
ADDRESS_RE = re.compile(r'^C[a-fA-F0-9]{40}$')


def now_ms() -> float:
    return time.time() * 1000.0


def utc_iso(ms: float) -> str:
    return datetime.fromtimestamp(ms / 1000.0, timezone.utc).isoformat(timespec='milliseconds')


def validate_pid(value) -> str:
    pid = str(value or '').strip().upper()
    if not PARTICIPANT_RE.fullmatch(pid) or int(pid[1:]) == 0:
        raise ValueError('참가자 코드는 P01~P999 형식이어야 합니다.')
    return pid


# ============================================================
# 1. 신호 처리
# ============================================================
def _bandpass(x, fs=FS, lo=0.7, hi=3.0, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    if len(x) <= 3 * max(len(a), len(b)) + 1:
        return None
    return filtfilt(b, a, x)


def _resample(t_ms, v):
    t = (t_ms - t_ms[0]) / 1000.0
    grid = np.arange(0.0, t[-1], 1.0 / FS)
    return grid, np.interp(grid, t, v)


def spectral_quality(x, fs=FS):
    x = np.asarray(x, float)
    x = x - x.mean()
    if len(x) < fs * 4 or np.std(x) < 1e-6:
        return 0.0, None
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)))
    wide = (f >= 0.5) & (f <= 5.0)
    band = (f >= 0.7) & (f <= 3.5)
    total = p[wide].sum()
    if total <= 0 or not band.any():
        return 0.0, None
    f0 = f[band][np.argmax(p[band])]
    near = (np.abs(f - f0) <= 0.2) & wide
    harm = (np.abs(f - 2 * f0) <= 0.2) & wide
    return float(min(1.0, (p[near].sum() + p[harm].sum()) / total)), float(f0 * 60)


def dominant_period_s(x, fs=FS):
    """스펙트럼 주 피크로 심박 주기 추정 (2배 고조파를 기본파로 착각하지 않도록 f0/2 확인)."""
    if len(x) < fs * 4:
        return None
    f, p = welch(x, fs=fs, nperseg=min(len(x), int(fs * 8)))
    band = (f >= 0.7) & (f <= 3.0)
    if not band.any() or p[band].max() <= 0:
        return None
    f0 = f[band][np.argmax(p[band])]
    if f0 / 2 >= 0.7 and p[np.abs(f - f0 / 2) <= 0.1].max(initial=0.0) >= 0.3 * p[band].max():
        f0 /= 2
    return 1.0 / f0


def finger_status(r, g) -> str:
    r_mean, g_mean = float(np.mean(r)), float(np.mean(g))
    if r_mean < 40:
        return 'dark'
    if r_mean < 1.3 * g_mean:
        return 'nofinger'
    if r_mean > 250 and g_mean > 250:
        return 'saturated'
    return 'ok'


def quality_label(snr: float) -> str:
    return 'good' if snr >= 0.5 else 'fair' if snr >= 0.3 else 'poor'


def analyze(t_ms, r, g):
    t_ms = np.asarray(t_ms, float)
    if len(t_ms) < 2 or t_ms[-1] - t_ms[0] < 6000:
        return None
    grid, rs = _resample(t_ms, np.asarray(r, float))
    _, gs = _resample(t_ms, np.asarray(g, float))
    candidates = [(spectral_quality(sig)[0], name, sig)
                  for name, sig in (('green', gs), ('red', rs)) if np.mean(sig) <= 250]
    if not candidates:
        return None
    q, channel, sig = max(candidates, key=lambda c: c[0])

    filt = _bandpass(-sig)                 # 혈액량↑ → 투과광↓ : 뒤집어서 수축기 = 피크
    if filt is None or np.std(filt) < 1e-9:
        return None
    period = dominant_period_s(filt)       # 중복맥을 두 번째 박동으로 세지 않도록
    min_dist = 0.65 * period if period else 0.33
    peaks, _ = find_peaks(filt, distance=max(1, int(min_dist * FS)), prominence=0.3 * np.std(filt))

    refined = []
    for p in peaks:                        # 포물선 보간으로 33 ms 격자 이하 정밀도
        if 0 < p < len(filt) - 1:
            a, b, c = filt[p - 1], filt[p], filt[p + 1]
            d = a - 2 * b + c
            refined.append(p + (0.5 * (a - c) / d if d != 0 else 0.0))
        else:
            refined.append(float(p))
    peak_ms = t_ms[0] + np.asarray(refined) / FS * 1000.0

    ibi = np.diff(peak_ms)
    valid = (ibi >= 330) & (ibi <= 1500)
    gaps = np.where(np.diff(t_ms) > GAP_MS)[0]
    for i in gaps:                         # 프레임 끊김을 가로지르는 IBI 제외
        valid &= ~((peak_ms[:-1] < t_ms[i + 1]) & (peak_ms[1:] > t_ms[i]))
    if valid.sum() >= 3:
        med = np.median(ibi[valid])
        valid &= np.abs(ibi - med) <= 0.3 * med
    return {
        'channel': channel, 'quality': q,
        'grid_ms': t_ms[0] + grid * 1000.0, 'filt': filt,
        'peak_ms': peak_ms, 'ibi_end_ms': peak_ms[1:], 'ibi_ms': ibi, 'valid': valid,
        'duration_s': (t_ms[-1] - t_ms[0]) / 1000.0, 'gap_count': int(len(gaps)),
    }


def hrv_stats(ibi_ms, valid) -> dict:
    ibi_ms, valid = np.asarray(ibi_ms, float), np.asarray(valid, bool)
    vi = ibi_ms[valid]
    pairs = valid[:-1] & valid[1:]
    succ = np.diff(ibi_ms)[pairs] if len(ibi_ms) > 1 else np.array([])
    return {
        'beats': int(valid.sum()),
        'hr_bpm': round(float(60000.0 / vi.mean()), 1) if len(vi) else None,
        'ibi_ms': round(float(vi.mean()), 1) if len(vi) else None,
        'sdnn_ms': round(float(vi.std()), 1) if len(vi) >= 2 else None,
        'rmssd_ms': round(float(np.sqrt(np.mean(succ ** 2))), 1) if len(succ) >= 2 else None,
    }


# ============================================================
# 2. 공용 저장소 (모든 Streamlit 세션이 공유)
# ============================================================
class IngestError(Exception):
    pass


class PPGSession:
    def __init__(self, pid: str):
        self.pid = pid
        self.created_ms = now_ms()
        self.t: list[float] = []        # 서버 시계 ms
        self.tp: list[float] = []       # 폰 시계 ms (원본)
        self.r: list[float] = []
        self.g: list[float] = []
        self.offsets = deque(maxlen=60)
        self.last_seen_ms = 0.0
        self.run_id = ''
        self.old_runs: set[str] = set()
        self.ack = 0.0                  # 현재 run에서 받은 폰 시각 최댓값
        self.active = False
        self.block: dict | None = None
        self.blocks: list[dict] = []
        self.ibi_rows: list[list] = []  # 블록별 IBI 기록 (CSV 다운로드용)

    @property
    def offset_ms(self) -> float:
        return min(self.offsets) if self.offsets else 0.0

    def window(self, start_ms: float, end_ms: float | None = None):
        i = bisect.bisect_left(self.t, start_ms)
        j = len(self.t) if end_ms is None else bisect.bisect_right(self.t, end_ms)
        return self.t[i:j], self.r[i:j], self.g[i:j]


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.submit_lock = threading.Lock()
        self.sessions: dict[str, PPGSession] = {}

    # ---- 폰 데이터 수신 ----
    def ingest(self, pid: str, value: dict) -> dict:
        if not isinstance(value, dict):
            raise IngestError('데이터 형식 오류')
        run_id, sent_at = value.get('run_id'), value.get('sent_at')
        samples, running = value.get('samples'), value.get('running') is True
        server_now = now_ms()
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise IngestError('측정 ID 오류')
        if isinstance(sent_at, bool) or not isinstance(sent_at, (int, float)) \
                or not math.isfinite(sent_at) or abs(sent_at - server_now) > 86_400_000:
            raise IngestError('폰 시계가 올바르지 않습니다.')
        if not isinstance(samples, list) or len(samples) > MAX_BATCH:
            raise IngestError('샘플 형식 오류')
        parsed = []
        for s in samples:
            if not (isinstance(s, list) and len(s) == 3) or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in s):
                raise IngestError('샘플 값 오류')
            t, r, g = map(float, s)
            if not (0 <= r <= 255 and 0 <= g <= 255) or abs(t - sent_at) > 300_000:
                raise IngestError('샘플 값 범위 오류')
            parsed.append((t, r, g))

        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None:
                if len(self.sessions) >= MAX_SESSIONS:
                    raise IngestError('동시 측정 세션 수를 초과했습니다.')
                sess = self.sessions[pid] = PPGSession(pid)
            if run_id != sess.run_id:
                if run_id in sess.old_runs:
                    raise IngestError('다른 기기에서 같은 참가자 코드로 측정이 시작되어 이 기기의 측정은 중단되었습니다.')
                if sess.run_id:                      # 새 측정 시작(새로고침/다른 기기) → 이전 것은 무효
                    sess.old_runs.add(sess.run_id)
                sess.run_id, sess.ack = run_id, 0.0
                sess.offsets.clear()
            if len(sess.t) + len(parsed) > MAX_SESSION_SAMPLES:
                raise IngestError('세션 최대 길이를 초과했습니다. 새 참가자 코드로 시작하세요.')
            sess.offsets.append(server_now - float(sent_at))
            offset = sess.offset_ms
            last_t = sess.t[-1] if sess.t else -math.inf
            for t, r, g in sorted(parsed):
                sess.ack = max(sess.ack, t)
                ts = t + offset
                if ts <= last_t:                     # 재전송 중복 제거
                    continue
                sess.t.append(ts); sess.tp.append(t); sess.r.append(r); sess.g.append(g)
                last_t = ts
            sess.last_seen_ms = server_now
            sess.active = running
            t8, r8, g8 = sess.window(server_now - 8000)
            ack = sess.ack

        if len(t8) < FS * 4:
            status = 'warming'
        else:
            status = finger_status(r8, g8)
            if status == 'ok':
                ta = np.asarray(t8)
                _, gr = _resample(ta, np.asarray(g8))
                _, rr = _resample(ta, np.asarray(r8))
                snr = max(spectral_quality(gr)[0], spectral_quality(rr)[0] if np.mean(rr) <= 250 else 0.0)
                status = quality_label(snr)
        return {'ack': ack, 'quality': status}

    # ---- 모니터 ----
    def session_list(self) -> list[dict]:
        now = now_ms()
        with self.lock:
            return [{'pid': s.pid, 'connected': s.active and now - s.last_seen_ms < 3500,
                     'block': s.block['condition'] if s.block else None}
                    for s in sorted(self.sessions.values(), key=lambda s: s.pid)]

    def live(self, pid: str) -> dict | None:
        now = now_ms()
        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None:
                return None
            t, r, g = (list(x) for x in sess.window(now - LIVE_WINDOW_MS))
            block = dict(sess.block) if sess.block else None
            last_seen, offset, active = sess.last_seen_ms, sess.offset_ms, sess.active
        recent = [x for x in t if x >= now - 5000]
        out = {
            'connected': active and last_seen > 0 and now - last_seen < 3500,
            'last_seen_s': round((now - last_seen) / 1000, 1) if last_seen else None,
            'fps': round((len(recent) - 1) / ((recent[-1] - recent[0]) / 1000), 1)
            if len(recent) > 2 and recent[-1] > recent[0] else None,
            'clock_offset_ms': round(offset, 1),
            'finger': finger_status(r[-150:], g[-150:]) if len(r) >= 30 else 'warming',
            'block': {'condition': block['condition'], 'elapsed_s': (now - block['start_ms']) / 1000}
            if block else None,
            'res': None,
        }
        res = analyze(t, r, g)
        if res is not None:
            end = res['grid_ms'][-1]
            last15 = res['ibi_end_ms'] >= end - 15_000
            last60 = res['ibi_end_ms'] >= end - 60_000
            out['res'] = res
            out['end_ms'] = end
            out['hr15'] = hrv_stats(res['ibi_ms'][last15], res['valid'][last15])
            out['hr60'] = hrv_stats(res['ibi_ms'][last60], res['valid'][last60])
        return out

    def start_block(self, pid: str, condition: str):
        condition = condition.strip().lower()
        if not CONDITION_RE.fullmatch(condition):
            raise ValueError('조건 이름은 영문 소문자/숫자/밑줄 32자 이내로 입력하세요.')
        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None:
                raise ValueError('세션이 없습니다.')
            if sess.block:
                raise ValueError('이미 진행 중인 블록이 있습니다. 먼저 종료하세요.')
            sess.block = {'id': secrets.token_hex(8), 'condition': condition, 'start_ms': now_ms()}

    def stop_block(self, pid: str) -> dict:
        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None or not sess.block:
                raise ValueError('진행 중인 블록이 없습니다.')
            block, sess.block = sess.block, None
            block['end_ms'] = now_ms()
            t, r, g = sess.window(block['start_ms'], block['end_ms'])
        summary = {
            'block_id': block['id'], 'participant_id': pid, 'condition': block['condition'],
            'started_at_utc': utc_iso(block['start_ms']), 'ended_at_utc': utc_iso(block['end_ms']),
            'duration_s': round((block['end_ms'] - block['start_ms']) / 1000, 1),
            'hr_bpm': None, 'ibi_ms': None, 'sdnn_ms': None, 'rmssd_ms': None, 'beats': 0,
            'coverage': 0.0, 'quality': None, 'channel': None, 'gap_count': None, 'tx_id': '', 'error': '',
        }
        res = analyze(t, r, g)
        rows = []
        if res is None:
            summary['error'] = '분석 가능한 신호가 부족합니다.'
        else:
            summary.update(hrv_stats(res['ibi_ms'], res['valid']))
            summary.update({
                'coverage': round(float(res['ibi_ms'][res['valid']].sum() / 1000 / max(res['duration_s'], 1e-9)), 3),
                'quality': round(res['quality'], 3), 'channel': res['channel'], 'gap_count': res['gap_count'],
            })
            rows = [[pid, block['condition'], block['id'], utc_iso(te), round(float(te), 1),
                     round(float(ibi), 1), bool(ok)]
                    for te, ibi, ok in zip(res['ibi_end_ms'], res['ibi_ms'], res['valid'])]
        with self.lock:
            sess.blocks.append(summary)
            sess.ibi_rows.extend(rows)
        return summary

    def blocks(self, pid: str) -> list[dict]:
        with self.lock:
            sess = self.sessions.get(pid)
            return [dict(b) for b in sess.blocks] if sess else []

    def set_tx(self, pid: str, block_id: str, tx_id: str):
        with self.lock:
            for b in self.sessions[pid].blocks:
                if b['block_id'] == block_id:
                    b['tx_id'] = tx_id

    def csv_raw(self, pid: str) -> bytes:
        with self.lock:
            s = self.sessions[pid]
            rows = list(zip(s.t, s.tp, s.r, s.g))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['t_server_ms', 't_phone_ms', 'red', 'green'])
        w.writerows((round(a, 1), round(b, 1), round(c, 3), round(d, 3)) for a, b, c, d in rows)
        return buf.getvalue().encode('utf-8')

    def csv_ibi(self, pid: str) -> bytes:
        with self.lock:
            rows = list(self.sessions[pid].ibi_rows)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['participant_id', 'condition', 'block_id', 'beat_time_utc', 'beat_t_server_ms', 'ibi_ms', 'valid'])
        w.writerows(rows)
        return buf.getvalue().encode('utf-8')

    def csv_blocks(self) -> bytes:
        fields = ['block_id', 'participant_id', 'condition', 'started_at_utc', 'ended_at_utc', 'duration_s',
                  'hr_bpm', 'ibi_ms', 'sdnn_ms', 'rmssd_ms', 'beats', 'coverage', 'quality', 'channel',
                  'gap_count', 'tx_id', 'error']
        with self.lock:
            rows = [dict(b) for s in self.sessions.values() for b in s.blocks]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
        return ('﻿' + buf.getvalue()).encode('utf-8')


@st.cache_resource
def get_store() -> Store:
    return Store()


# ============================================================
# 3. XperChain 등록
# ============================================================
class NodeError(Exception):
    pass


def node_request(method: str, path: str, **kwargs) -> dict:
    try:
        response = requests.request(method, f'{NODE_URL}{path}', timeout=8, **kwargs)
        try:
            data = response.json()
        except ValueError as exc:
            raise NodeError(f'XperChain이 {method} {path}에 JSON 대신 HTTP {response.status_code}를 반환했습니다.') from exc
        if not isinstance(data, dict):
            raise NodeError('XperChain이 예상치 못한 응답을 반환했습니다.')
        if response.status_code >= 400 or not data.get('ok', True):
            raise NodeError(f"노드가 요청을 거부했습니다: {data.get('error') or response.status_code}")
        return data
    except requests.RequestException as exc:
        raise NodeError('XperChain 노드에 연결할 수 없습니다.') from exc


def service_signing_key() -> Ed25519PrivateKey:
    value = config('RATING_WALLET_PRIVATE_KEY_HEX')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        raise NodeError('RATING_WALLET_PRIVATE_KEY_HEX(64자리 hex)가 설정되지 않았습니다.')
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(value))


def wallet_address(key: Ed25519PrivateKey) -> tuple[str, bytes]:
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return 'W' + hashlib.sha256(public).hexdigest()[:40], public


def canonical_tx_bytes(tx: dict) -> bytes:
    fields = ('sender', 'recipient', 'amount_xbit', 'fee_xbit', 'nonce', 'timestamp', 'tx_type', 'data')
    return json.dumps({k: tx[k] for k in fields}, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False).encode('utf-8')


def build_record(s: dict) -> dict:
    return {'schema': SCHEMA, 'block_id': s['block_id'], 'participant_id': s['participant_id'],
            'condition': s['condition'], 'hr': s['hr_bpm'], 'ibi_ms': s['ibi_ms'], 'sdnn_ms': s['sdnn_ms'],
            'rmssd_ms': s['rmssd_ms'], 'beats': s['beats'], 'dur_s': s['duration_s'], 'cov': s['coverage'],
            'q': s['quality'], 'start_utc': s['started_at_utc'][:19] + 'Z'}


def build_tx_data(record: dict) -> dict:
    text = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return {'contract_addr': CONTRACT_ADDR, 'method': 'upload_text', 'args': {'title': TITLE, 'text': text}}


def tx_data_size(tx_data: dict) -> int:
    return len(json.dumps(tx_data, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def used_block_ids() -> set[str]:
    if not ADDRESS_RE.fullmatch(CONTRACT_ADDR):
        raise NodeError('XPER_CONTRACT_ADDRESS 설정이 올바르지 않습니다.')
    if node_request('GET', '/status').get('is_synced') is not True:
        raise NodeError('노드가 아직 동기화 중입니다.')
    snap = node_request('GET', f'/contract/{CONTRACT_ADDR}')
    try:
        meta = json.loads((snap.get('contract') or {}).get('code', '{}'))
    except (TypeError, ValueError) as exc:
        raise NodeError('컨트랙트 메타데이터를 읽을 수 없습니다.') from exc
    if meta.get('template') != 'logbook_v1' or 'upload_text' not in meta.get('methods', []):
        raise NodeError('upload_text를 지원하는 logbook_v1 컨트랙트가 아닙니다.')
    texts = [i.get('text', '') for i in (snap.get('state') or {}).get('data_items', []) if isinstance(i, dict)]
    for tx in node_request('GET', '/mempool').get('mempool', []):
        if isinstance(tx, dict) and tx.get('tx_type') == 'CONTRACT_CALL':
            d = tx.get('data', {})
            if d.get('contract_addr') == CONTRACT_ADDR and d.get('method') == 'upload_text':
                texts.append((d.get('args') or {}).get('text', ''))
    ids = set()
    for text in texts:
        try:
            c = json.loads(text)
            if isinstance(c, dict) and c.get('schema') == SCHEMA and isinstance(c.get('block_id'), str):
                ids.add(c['block_id'])
        except (TypeError, ValueError):
            continue
    return ids


def register_block(store: Store, summary: dict) -> str:
    if summary.get('tx_id'):
        raise NodeError('이미 등록 요청한 블록입니다.')
    if summary.get('hr_bpm') is None:
        raise NodeError('분석 결과가 없는 블록은 등록할 수 없습니다.')
    record = build_record(summary)
    tx_data = build_tx_data(record)
    if tx_data_size(tx_data) > MAX_TX_DATA_BYTES:
        raise NodeError(f'기록이 {MAX_TX_DATA_BYTES}바이트 제한을 넘습니다. 조건 이름을 줄이세요.')
    key = service_signing_key()
    wallet, public = wallet_address(key)
    with store.submit_lock:
        if summary['block_id'] in used_block_ids():
            raise NodeError('이 블록은 이미 블록체인에 등록(또는 대기)되어 있습니다.')
        nonce = node_request('GET', f'/account/{wallet}/nonce')
        fee = node_request('POST', '/fee/estimate', json={'tx_type': 'CONTRACT_CALL', 'data': tx_data})
        tx = {'sender': wallet, 'recipient': CONTRACT_ADDR, 'amount_xbit': 0,
              'fee_xbit': int(fee['required_fee_xbit']), 'nonce': int(nonce['next_nonce']),
              'timestamp': time.time(), 'tx_type': 'CONTRACT_CALL', 'data': tx_data}
        sig = key.sign(canonical_tx_bytes(tx))
        result = node_request('POST', '/contract/call', json={
            'contract_addr': CONTRACT_ADDR, 'method': 'upload_text', 'args': tx_data['args'],
            'sender': wallet, 'sender_pub_b64': base64.b64encode(public).decode('ascii'),
            'sig_b64': base64.b64encode(sig).decode('ascii'), 'value_xbit': 0,
            'fee_xbit': tx['fee_xbit'], 'nonce': tx['nonce'], 'timestamp': tx['timestamp']})
    tx_id = str(result.get('tx_id', ''))
    if not TX_ID_RE.fullmatch(tx_id):
        raise NodeError('노드가 올바른 트랜잭션 ID를 반환하지 않았습니다.')
    store.set_tx(summary['participant_id'], summary['block_id'], tx_id)
    return tx_id


def tx_status(tx_id: str) -> str:
    pool = node_request('GET', '/mempool').get('mempool', [])
    if any(isinstance(t, dict) and str(t.get('hash', '')).lower() == tx_id.lower() for t in pool):
        return '⏳ 대기 중'
    length = int(node_request('GET', '/status').get('length', 0))
    floor, end = max(0, length - 1000), length
    while end > floor:
        start = max(floor, end - 200)
        blocks = node_request('GET', f'/chain/from/{start}', params={'limit': end - start}).get('chain', [])
        for b in reversed(blocks if isinstance(blocks, list) else []):
            txs = b.get('transactions', []) if isinstance(b, dict) else []
            if any(isinstance(t, dict) and str(t.get('hash', '')).lower() == tx_id.lower() for t in txs):
                return f"✅ 확정 (블록 #{b.get('index')})"
        end = start
    return '❔ 최근 블록에서 찾지 못함'


# ============================================================
# 4. 폰 측정 컴포넌트 (JS)
# ============================================================
COMPONENT_HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<style>
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
         color:#eef1f4; background:#0f1216; text-align:center; }
  .wrap { padding:18px 14px 22px; }
  button { width:100%; margin-top:14px; padding:16px; font-size:18px; border:0; border-radius:10px;
           background:#3b82f6; color:#fff; }
  button.stop { background:#374151; }
  #dot { width:140px; height:140px; border-radius:50%; margin:6px auto 14px; background:#4b5563; transition:background .3s; }
  #msg { font-size:18px; line-height:1.5; min-height:54px; }
  #elapsed { color:#8a96a3; margin-top:8px; font-variant-numeric:tabular-nums; }
  .hint { color:#8a96a3; font-size:14px; line-height:1.6; margin-top:12px; }
  #err { color:#f87171; margin-top:10px; font-size:15px; }
  .hidden { display:none !important; }
  video { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
</style></head>
<body><div class="wrap">
  <div id="setup">
    <div class="hint">휴대폰을 책상이나 받침대에 올려 두고,<br>왼손 검지로 <b>후면 카메라와 플래시를 함께</b> 가볍게 덮어 주세요.<br>너무 세게 누르면 신호가 약해집니다.</div>
    <button id="startBtn">측정 시작</button>
  </div>
  <div id="run" class="hidden">
    <div id="dot"></div>
    <div id="msg">측정 준비 중...</div>
    <div id="elapsed">00:00</div>
    <div class="hint" id="torchNote"></div>
    <button class="stop" id="stopBtn">측정 종료</button>
  </div>
  <div id="err"></div>
</div>
<video id="video" playsinline muted autoplay></video>
<script>
"use strict";
// ---- Streamlit 컴포넌트 프로토콜 (streamlit-component-lib 없이 직접 구현) ----
function post(type, extra) { window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, extra || {}), "*"); }
function setHeight() { post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight }); }
function setValue(v) { post("streamlit:setComponentValue", { value: v, dataType: "json" }); }

const $ = (id) => document.getElementById(id);
const W = 40, H = 30;
const video = $("video");
const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
const ctx = canvas.getContext("2d", { willReadFrequently: true });
let stream = null, track = null, running = false, queue = [], runId = "", seq = 0;
let sendTimer = null, clockTimer = null, fallbackTimer = null, wakeLock = null, startedAt = 0, lastMediaTime = -1;
let lastServerReply = 0;

const MESSAGES = {
  warming:   ["#4b5563", "측정 준비 중... 손가락을 그대로 두세요"],
  good:      ["#22c55e", "신호 좋음 — 그대로 유지하세요"],
  fair:      ["#eab308", "신호 보통 — 손가락을 움직이지 마세요"],
  poor:      ["#ef4444", "신호 약함 — 손가락을 가볍게 다시 올려 주세요"],
  nofinger:  ["#6b7280", "손가락이 감지되지 않습니다 — 카메라와 플래시를 함께 덮어 주세요"],
  dark:      ["#6b7280", "너무 어둡습니다 — 플래시가 켜졌는지 확인하세요"],
  saturated: ["#ef4444", "너무 밝습니다 — 손가락으로 카메라를 완전히 덮어 주세요"],
  network:   ["#a855f7", "서버 응답이 늦습니다 — 측정은 계속되며 데이터는 다시 전송됩니다"],
};
function show(status, text) {
  const [color, def] = MESSAGES[status] || MESSAGES.warming;
  $("dot").style.background = color; $("msg").textContent = text || def;
}

// Python이 매 실행마다 보내주는 값: ack(서버가 받은 마지막 폰 시각), quality, error
window.addEventListener("message", (ev) => {
  const d = ev.data;
  if (!d || d.type !== "streamlit:render") return;
  const a = d.args || {};
  if (a.run_id === runId && typeof a.ack === "number") {
    queue = queue.filter(s => s[0] > a.ack);       // 서버가 받은 샘플은 버림
    lastServerReply = performance.now();
    if (running && a.quality) show(a.quality);
  }
  if (a.error && a.run_id === runId && running) { stopLocal(a.error); }
  setHeight();
});

function sampleFrame(tsPerf) {
  ctx.drawImage(video, 0, 0, W, H);
  const d = ctx.getImageData(0, 0, W, H).data;
  let r = 0, g = 0;
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; }
  const n = d.length / 4;
  queue.push([+(performance.timeOrigin + tsPerf).toFixed(1), +(r / n).toFixed(3), +(g / n).toFixed(3)]);
  if (queue.length > 9000) queue.splice(0, queue.length - 9000);   // 최대 약 5분 보관
}
function onVideoFrame(now, meta) {
  if (!running) return;
  if (meta.mediaTime !== lastMediaTime) {
    lastMediaTime = meta.mediaTime;
    const ct = meta.captureTime;
    sampleFrame(ct && Math.abs(ct - now) < 1000 ? ct : now);
  }
  video.requestVideoFrameCallback(onVideoFrame);
}
function push(isRunning) {
  setValue({ run_id: runId, seq: ++seq, sent_at: performance.timeOrigin + performance.now(),
             running: isRunning, samples: queue.slice(0, 1200) });
  if (isRunning && lastServerReply && performance.now() - lastServerReply > 5000) show("network");
}
function tick() {
  const s = Math.floor((performance.now() - startedAt) / 1000);
  $("elapsed").textContent = String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}
function randomId() { const a = new Uint8Array(12); crypto.getRandomValues(a); return Array.from(a, b => b.toString(16).padStart(2, "0")).join(""); }

async function start() {
  $("err").textContent = "";
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { $("err").textContent = "이 브라우저에서는 카메라를 사용할 수 없습니다."; return; }
  $("startBtn").disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 30, max: 30 } } });
    video.srcObject = stream; await video.play();
    track = stream.getVideoTracks()[0];
    try { await track.applyConstraints({ advanced: [{ torch: true }] }); $("torchNote").textContent = "플래시가 켜졌는지 확인하세요."; }
    catch (e) { $("torchNote").textContent = "이 브라우저에서는 플래시를 켤 수 없습니다. Android는 Chrome을 사용하거나 밝은 조명 아래에서 측정하세요."; }
    runId = randomId(); seq = 0; queue = []; lastMediaTime = -1; lastServerReply = 0;
    running = true; startedAt = performance.now();
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) video.requestVideoFrameCallback(onVideoFrame);
    else fallbackTimer = setInterval(() => sampleFrame(performance.now()), 33);
    sendTimer = setInterval(() => push(true), 1000);
    clockTimer = setInterval(tick, 1000);
    try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {}
    $("setup").classList.add("hidden"); $("run").classList.remove("hidden");
    show("warming"); setHeight();
  } catch (e) {
    $("err").textContent = e && e.name === "NotAllowedError" ? "카메라 권한을 허용해 주세요." : (e.message || String(e));
    stopLocal();
  } finally { $("startBtn").disabled = false; }
}
function stopLocal(reason) {
  running = false;
  [sendTimer, clockTimer, fallbackTimer].forEach(t => t && clearInterval(t));
  sendTimer = clockTimer = fallbackTimer = null;
  if (track) { try { track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) {} }
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = track = null;
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  $("run").classList.add("hidden"); $("setup").classList.remove("hidden");
  if (reason) $("err").textContent = reason;
  setHeight();
}
function stop() {
  push(false);                     // 남은 샘플 + 종료 알림
  stopLocal("측정이 종료되었습니다.");
}
document.addEventListener("visibilitychange", () => {
  if (running && document.visibilityState !== "visible") show("poor", "화면이 꺼지거나 다른 앱으로 전환되면 측정이 멈춥니다");
});
$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", stop);
post("streamlit:componentReady", { apiVersion: 1 });
setHeight();
</script></body></html>
"""


@st.cache_resource
def ppg_component():
    """HTML을 임시 폴더에 써서 양방향 컴포넌트로 등록 (빌드 도구 불필요)."""
    import streamlit.components.v1 as components
    folder = Path(tempfile.gettempdir()) / 'ppg_camera_component'
    folder.mkdir(exist_ok=True)
    index = folder / 'index.html'
    if not index.exists() or index.read_text(encoding='utf-8') != COMPONENT_HTML:
        index.write_text(COMPONENT_HTML, encoding='utf-8')
    return components.declare_component('ppg_camera', path=str(folder))


# ============================================================
# 5. 화면
# ============================================================
def phone_page():
    st.markdown("### 📷 PPG 측정")
    pid_raw = st.text_input("참가자 코드", placeholder="P01", max_chars=4)
    if not pid_raw:
        st.caption("실험자가 알려준 참가자 코드를 입력하세요.")
        return
    try:
        pid = validate_pid(pid_raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    # 체크박스는 항상 컴포넌트보다 위에 두어, 켜고 꺼도 측정 컴포넌트가 다시 만들어지지 않게 한다
    show_results = st.checkbox(
        "측정 결과 함께 보기", value=False,
        help="본 실험에서는 끄세요. 참가자가 자기 심박을 보면 바이오피드백으로 상태가 바뀔 수 있습니다.")
    phone_panel(pid)
    if show_results:
        st.divider()
        live_panel(pid)
        block_controls(pid)
        blocks_panel(pid)


@st.fragment
def phone_panel(pid: str):
    """컴포넌트 값이 바뀔 때(1초마다) 이 조각만 다시 실행된다."""
    ss = st.session_state
    state = ss.setdefault('phone', {})
    component = ppg_component()
    value = component(run_id=state.get('run_id', ''), ack=state.get('ack', 0.0),
                      quality=state.get('quality', 'warming'), error=state.get('error', ''),
                      key=f'ppg_{pid}', default=None)
    if not isinstance(value, dict) or (value.get('run_id'), value.get('seq')) == state.get('last'):
        return
    state['last'] = (value.get('run_id'), value.get('seq'))
    state['run_id'] = value.get('run_id', '')
    try:
        result = get_store().ingest(pid, value)
        state.update(ack=result['ack'], quality=result['quality'], error='')
    except IngestError as exc:
        state['error'] = str(exc)


def experimenter_login(key: str) -> bool:
    """실험자 로그인 여부. 로그인 안 되어 있으면 비밀번호 입력창을 표시."""
    if st.session_state.get('experimenter') is True:
        return True
    password = config('MONITOR_PASSWORD')
    if not password:
        st.error("secrets 또는 환경변수에 MONITOR_PASSWORD를 설정하세요.")
        return False
    entered = st.text_input("실험자 비밀번호", type="password", key=f'pw_{key}')
    if entered:
        if secrets.compare_digest(entered, password):
            st.session_state['experimenter'] = True
            st.rerun()
        else:
            time.sleep(1)
            st.error("비밀번호가 틀렸습니다.")
    return False


def block_controls(pid: str):
    store = get_store()
    st.subheader("블록")
    c1, c2, c3 = st.columns([2, 1, 1])
    condition = c1.text_input("조건", placeholder="baseline / defensive / aggressive / load_only / load_anxiety ...",
                              label_visibility="collapsed", key=f'cond_{pid}')
    if c2.button("▶ 블록 시작", key=f'bstart_{pid}'):
        try:
            store.start_block(pid, condition)
            st.toast(f"블록 시작: {condition}")
        except ValueError as exc:
            st.error(str(exc))
    if c3.button("■ 블록 종료", key=f'bstop_{pid}'):
        try:
            s = store.stop_block(pid)
            st.toast(f"블록 종료: {s['condition']} ({s['duration_s']}초)")
            if s['error']:
                st.warning(s['error'])
        except ValueError as exc:
            st.error(str(exc))


def monitor_page():
    st.title("📈 PPG 실험자 모니터")
    if not experimenter_login('monitor'):
        return

    store = get_store()
    sessions = store.session_list()
    top = st.columns([3, 1])
    if top[1].button("🔄 세션 목록 새로고침"):
        st.rerun()
    if not sessions:
        top[0].info("아직 측정 중인 폰이 없습니다. 참가자 폰에서 측정을 시작한 뒤 새로고침하세요.")
        return
    labels = {s['pid']: f"{s['pid']} {'● 연결됨' if s['connected'] else '○ 끊김'}" for s in sessions}
    pid = top[0].selectbox("참가자", list(labels), format_func=labels.get)

    live_panel(pid)
    block_controls(pid)
    blocks_panel(pid)


@st.fragment(run_every=1.0)
def live_panel(pid: str):
    import altair as alt
    import pandas as pd
    live = get_store().live(pid)
    if live is None:
        st.info("측정을 시작하면 여기에 결과가 표시됩니다.")
        return
    res = live['res']
    m = st.columns(5)
    m[0].metric("폰 연결", "연결됨" if live['connected'] else
                ("데이터 없음" if live['last_seen_s'] is None else f"끊김 ({live['last_seen_s']}s)"))
    m[1].metric("심박 (15초)", f"{live['hr15']['hr_bpm']} bpm" if res and live['hr15']['hr_bpm'] else "-")
    m[2].metric("RMSSD (60초)", f"{live['hr60']['rmssd_ms']} ms" if res and live['hr60']['rmssd_ms'] else "-")
    m[3].metric("신호 품질", f"{res['quality']:.2f} ({quality_label(res['quality'])})" if res else "-")
    m[4].metric("카메라 FPS", live['fps'] or "-")
    block = live['block']
    st.caption(f"손가락: {live['finger']} · 채널: {res['channel'] if res else '-'} · 시계 보정: {live['clock_offset_ms']} ms · "
               + (f"**진행 중인 블록: {block['condition']} ({int(block['elapsed_s'])}초)**" if block else "진행 중인 블록 없음"))
    if not res:
        st.info("분석할 신호가 부족합니다 (6초 이상 필요).")
        return

    end = live['end_ms']
    w = res['grid_ms'] >= end - 10_000
    wave = pd.DataFrame({'t': (res['grid_ms'][w] - end) / 1000, 'y': res['filt'][w]})
    pk = res['peak_ms'] >= end - 10_000
    ytop = np.interp(res['peak_ms'][pk], res['grid_ms'], res['filt'])
    peaks = pd.DataFrame({'t': (res['peak_ms'][pk] - end) / 1000, 'y': ytop})
    x = alt.X('t:Q', title='초 (현재 = 0)', scale=alt.Scale(domain=[-10, 0]))
    wave_chart = (alt.Chart(wave).mark_line(color='#3b82f6').encode(x=x, y=alt.Y('y:Q', title=None, axis=None))
                  + alt.Chart(peaks).mark_circle(color='#ef4444', size=60).encode(x=x, y='y:Q')).properties(height=200)

    tacho = pd.DataFrame({'t': (res['ibi_end_ms'] - end) / 1000, 'ibi': res['ibi_ms'],
                          '상태': np.where(res['valid'], '유효', '제외')})
    tacho_chart = alt.Chart(tacho).mark_circle(size=30).encode(
        x=alt.X('t:Q', title='초 (현재 = 0)', scale=alt.Scale(domain=[-120, 0])),
        y=alt.Y('ibi:Q', title='IBI (ms)', scale=alt.Scale(domain=[400, 1400], clamp=True)),
        color=alt.Color('상태:N', scale=alt.Scale(domain=['유효', '제외'], range=['#22c55e', '#9ca3af']))
    ).properties(height=180)
    c1, c2 = st.columns(2)
    c1.markdown("**실시간 파형 (최근 10초, 빨간 점 = 박동)**")
    c1.altair_chart(wave_chart)
    c2.markdown("**IBI 타코그램 (최근 2분)**")
    c2.altair_chart(tacho_chart)


@st.fragment
def blocks_panel(pid: str):
    import pandas as pd
    store = get_store()
    st.subheader("완료된 블록")
    if st.button("목록 새로고침"):
        st.rerun(scope="fragment")
    blocks = store.blocks(pid)
    if not blocks:
        st.caption("완료된 블록이 없습니다.")
    else:
        cols = ['condition', 'started_at_utc', 'duration_s', 'hr_bpm', 'ibi_ms', 'sdnn_ms', 'rmssd_ms',
                'beats', 'coverage', 'quality', 'tx_id', 'error']
        st.dataframe(pd.DataFrame(blocks)[cols], hide_index=True)

        pending = [b for b in blocks if not b['tx_id'] and b['hr_bpm'] is not None]
        is_experimenter = st.session_state.get('experimenter') is True
        if pending and not is_experimenter:
            # 블록체인 등록은 수수료 지갑을 쓰므로 실험자만 가능
            with st.expander("🔗 블록체인 등록 (실험자 로그인 필요)"):
                experimenter_login(f'reg_{pid}')
        if pending and is_experimenter:
            c1, c2 = st.columns([3, 1])
            chosen = c1.selectbox("블록체인에 등록할 블록", pending,
                                  format_func=lambda b: f"{b['condition']} · {b['started_at_utc'][11:19]} UTC · HR {b['hr_bpm']}")
            if c2.button("🔗 등록"):
                try:
                    tx_id = register_block(store, chosen)
                    st.success(f"등록 요청 완료: {tx_id}")
                except NodeError as exc:
                    st.error(str(exc))
        sent = [b for b in blocks if b['tx_id']]
        if sent:
            c1, c2 = st.columns([3, 1])
            chosen_tx = c1.selectbox("확정 상태 확인", sent, format_func=lambda b: f"{b['condition']} · {b['tx_id'][:16]}…")
            if c2.button("상태 확인"):
                try:
                    st.info(tx_status(chosen_tx['tx_id']))
                except (NodeError, ValueError) as exc:
                    st.error(str(exc))

    if pid not in store.sessions:
        return
    st.caption("앱이 재시작되면 메모리의 데이터가 사라집니다. 세션이 끝나면 내려받으세요.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("원시 신호 CSV", store.csv_raw(pid), f"{pid}_raw.csv", "text/csv", key=f'dl_raw_{pid}')
    d2.download_button("블록별 IBI CSV", store.csv_ibi(pid), f"{pid}_ibi.csv", "text/csv", key=f'dl_ibi_{pid}')
    if st.session_state.get('experimenter') is True:   # 다른 참가자 데이터가 섞여 있으므로 실험자만
        d3.download_button("전체 블록 요약 CSV", store.csv_blocks(), "blocks.csv", "text/csv", key='dl_blocks')


def main():
    is_monitor = st.query_params.get('page') == 'monitor'
    st.set_page_config(page_title="PPG 모니터" if is_monitor else "PPG 측정",
                       layout="wide" if is_monitor else "centered")
    if is_monitor:
        monitor_page()
    else:
        phone_page()


if __name__ == '__main__':
    main()

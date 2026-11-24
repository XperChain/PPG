"""PPG + 휴식/1-back 측정 Streamlit 앱 (한 화면에서 준비 → 측정 → 결과 → 블록체인 저장).

    streamlit run ppg_realtime.py

  참가자 폰:   https://<앱 주소>/                 (Streamlit Cloud는 HTTPS 자동 제공)
  실험자 PC:   https://<앱 주소>/?page=monitor    (선택, MONITOR_PASSWORD 필요)

흐름 (폰 한 화면)
  1) [측정 준비]  카메라+플래시 ON → PPG 파형과 검출된 박동, IBI 그래프를 1초마다 갱신해 신호 확인
  2) [측정 시작]  휴식 블록과 1-back 블록을 무작위 순서로 진행 (휴식→1-back 또는 1-back→휴식)
                  두 블록 모두 숫자 20칸 길이이며 같은 구간에서 IBI를 기록:
                    1~5칸   적응 구간 (기록 안 함)
                    6~15칸  IBI 기록 구간
                    16~20칸 마무리 구간 (기록 안 함)
                  휴식 블록: 화면의 + 를 바라보며 가만히 있기 (응답 없음)
                  1-back 블록: 숫자가 바로 앞 숫자와 같으면 [일치]
                  블록 사이에 안내 + 5초 카운트다운
  3) [블록체인 저장]  두 블록의 IBI + 1-back 결과를 트랜잭션 1개로 XperChain LogBook에 기록
  상단 [블록체인 보기]  컨트랙트의 모든 기록을 표로 보여 주고 엑셀(.xlsx)로 내려받기

블록체인 기록 형식 (schema = prb3, 노드의 526바이트 제한 안에 두 블록을 담도록 압축)
  {"schema":"prb3","mid":측정ID,"pid":"P01","ms":숫자간격 ms,"o":"rb"(진행 순서: r=휴식, b=1-back),
   "t":첫 블록 기록 시작 UTC,
   "r":{"d":t부터 이 블록 기록 시작까지 ms(첫 블록은 생략),"b0":기록 시작~첫 박동 ms,"ibi":압축 IBI,"x":[제외 박동 번호]},
   "b":{ ...휴식과 같음..., "nb":[적중,놓침,오경보,정기각,적중 평균 RT ms]}}
  ibi 압축: 정수 ms IBI를 첫 값은 varint(ibi), 다음부터 varint(zigzag(차이))로 바이트열을 만든 뒤
            Base85(Python base64.b85encode). 앱의 decode_ibis()로 풀 수 있고,
            [블록체인 보기] 엑셀에는 풀린 값이 들어갑니다. x는 이상치로 제외된 박동의 0부터 센 번호.
  심박/SDNN/RMSSD는 IBI에서 다시 계산할 수 있어 저장하지 않습니다.

설정 (.streamlit/secrets.toml 또는 환경변수)
  RATING_WALLET_PRIVATE_KEY_HEX = "..."     # 블록체인 저장 시 필요 (수수료 지갑)
  MONITOR_PASSWORD = "..."                  # 실험자 모니터 화면용
  ALLOWED_PARTICIPANT_CODES = "P01,P02"     # (선택) 저장을 허용할 참가자 코드
  XPER_NODE_URL, XPER_CONTRACT_ADDRESS      # (선택) 기본값은 seat touch 앱과 동일

주의: Streamlit Cloud는 앱이 재시작되면 메모리가 지워집니다. 필요한 CSV는 바로 내려받으세요.
"""
from __future__ import annotations

import base64
import bisect
import csv
import hashlib
import io
import json
import math
import os
import re
import secrets
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import NormalDist

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
SCHEMA = 'prb3'
TITLE = 'PPG'
COND_KEY = {'rest': 'r', '1back': 'b'}       # 기록 안에서 쓰는 짧은 블록 키
KEY_COND = {v: k for k, v in COND_KEY.items()}
CONDITIONS = ('rest', '1back')
COND_LABEL = {'rest': '휴식', '1back': '1-back'}
MAX_TX_DATA_BYTES = 526

# 블록 구성 (숫자 칸 수, 휴식 블록도 같은 길이)
N_PRE, N_REC, N_POST = 5, 10, 5
N_TOTAL = N_PRE + N_REC + N_POST
STIM_MS = 500                      # 숫자 제시 시간
DEFAULT_TRIAL_MS = 2500            # 숫자 간격 (블록당 기록 구간 = 10 × 2.5초 = 25초)
BREAK_S = 5                        # 블록 시작 전 안내 + 카운트다운

# 화면 상단 메뉴 아이콘/이름. 아이콘은 이모지 또는 Streamlit 머티리얼 아이콘(:material/이름:)
#   다른 후보: ':material/ecg_heart:', ':material/cardiology:', '🫀', '💓'
MEASURE_ICON, MEASURE_NAME = ':material/monitor_heart:', '심박 측정'
CHAIN_ICON, CHAIN_NAME = '⛓️', '블록체인 보기'

FS = 30.0
GAP_MS = 400.0
LIVE_WINDOW_MS = 60_000
MAX_BATCH = 1200
MAX_SESSION_SAMPLES = 400_000
MAX_SESSIONS = 30

PARTICIPANT_RE = re.compile(r'^P\d{2,3}$')
RUN_ID_RE = re.compile(r'^[a-f0-9]{8,32}$')
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


def _num(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)


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


def score_nback(trials: list[dict], n: int = 1) -> dict:
    """서버에서 정답을 다시 계산한 n-back 성적."""
    hit = miss = fa = cr = 0
    hit_rts = []
    for i, t in enumerate(trials):
        target = i >= n and trials[i - n]['s'] == t['s']
        t['target'] = target
        if target and t['r']:
            hit += 1
            if t['rt'] is not None:
                hit_rts.append(t['rt'])
        elif target:
            miss += 1
        elif t['r']:
            fa += 1
        else:
            cr += 1
    z = NormalDist().inv_cdf
    h = (hit + 0.5) / (hit + miss + 1)          # log-linear 보정
    f = (fa + 0.5) / (fa + cr + 1)
    return {
        'hit': hit, 'miss': miss, 'false_alarm': fa, 'correct_rejection': cr,
        'accuracy': round((hit + cr) / max(1, len(trials)), 3),
        'd_prime': round(z(h) - z(f), 3),
        'mean_rt_hit_ms': round(sum(hit_rts) / len(hit_rts), 1) if hit_rts else None,
    }


# ============================================================
# 2. 공용 저장소 (모든 Streamlit 세션이 공유)
# ============================================================
class IngestError(Exception):
    pass


class PPGSession:
    def __init__(self, pid: str):
        self.pid = pid
        self.t: list[float] = []        # 서버 시계 ms
        self.tp: list[float] = []       # 폰 시계 ms (1-back 시각과 같은 시계)
        self.r: list[float] = []
        self.g: list[float] = []
        self.offsets = deque(maxlen=60)
        self.last_seen_ms = 0.0
        self.run_id = ''
        self.run_start_idx = 0          # 현재 측정(run)의 첫 샘플 위치
        self.old_runs: set[str] = set()
        self.ack = 0.0
        self.active = False
        self.results: list[dict] = []

    @property
    def offset_ms(self) -> float:
        return min(self.offsets) if self.offsets else 0.0

    def window(self, start_ms: float):
        i = bisect.bisect_left(self.t, start_ms)
        return self.t[i:], self.r[i:], self.g[i:]


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.submit_lock = threading.Lock()
        self.sessions: dict[str, PPGSession] = {}

    def ingest(self, pid: str, value: dict) -> dict:
        if not isinstance(value, dict):
            raise IngestError('데이터 형식 오류')
        run_id, sent_at = value.get('run_id'), value.get('sent_at')
        samples, running = value.get('samples'), value.get('running') is True
        server_now = now_ms()
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise IngestError('측정 ID 오류')
        if not _num(sent_at) or abs(sent_at - server_now) > 86_400_000:
            raise IngestError('폰 시계가 올바르지 않습니다.')
        if not isinstance(samples, list) or len(samples) > MAX_BATCH:
            raise IngestError('샘플 형식 오류')
        parsed = []
        for s in samples:
            if not (isinstance(s, list) and len(s) == 3 and all(_num(v) for v in s)):
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
                    raise IngestError('다른 기기(또는 새 측정)에서 같은 참가자 코드로 측정이 시작되어 이 측정은 중단되었습니다.')
                if sess.run_id:
                    sess.old_runs.add(sess.run_id)
                sess.run_id, sess.ack, sess.run_start_idx = run_id, 0.0, len(sess.t)
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

    def live(self, pid: str, window_ms: float = LIVE_WINDOW_MS) -> dict | None:
        now = now_ms()
        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None:
                return None
            t, r, g = (list(x) for x in sess.window(now - window_ms))
            last_seen, offset, active = sess.last_seen_ms, sess.offset_ms, sess.active
        recent = [x for x in t if x >= now - 5000]
        out = {
            'connected': active and last_seen > 0 and now - last_seen < 3500,
            'last_seen_s': round((now - last_seen) / 1000, 1) if last_seen else None,
            'fps': round((len(recent) - 1) / ((recent[-1] - recent[0]) / 1000), 1)
            if len(recent) > 2 and recent[-1] > recent[0] else None,
            'clock_offset_ms': round(offset, 1),
            'finger': finger_status(r[-150:], g[-150:]) if len(r) >= 30 else 'warming',
            'res': None,
        }
        res = analyze(t, r, g)
        if res is not None:
            end = res['grid_ms'][-1]
            last15 = res['ibi_end_ms'] >= end - 15_000
            out.update(res=res, end_ms=end,
                       hr15=hrv_stats(res['ibi_ms'][last15], res['valid'][last15]),
                       all=hrv_stats(res['ibi_ms'], res['valid']))
        return out

    # ---- 휴식 / 1-back 측정 결과 ----
    def compute_result(self, pid: str, run_id: str, task: dict, waited_s: float) -> dict | None:
        """두 블록 각각의 기록 구간(폰 시계) IBI + 1-back 성적. 데이터가 아직 다 안 왔으면 None."""
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not RUN_ID_RE.fullmatch(task_id):
            raise ValueError('측정 ID 오류')
        trial_ms = task.get('trial_ms')
        if not _num(trial_ms) or not 1000 <= trial_ms <= 6000:
            raise ValueError('숫자 간격 값 오류')
        trial_ms = int(trial_ms)
        order = task.get('order')
        if not isinstance(order, list) or sorted(order) != sorted(CONDITIONS):
            raise ValueError('블록 순서 값 오류')
        raw_blocks = task.get('blocks')
        if not isinstance(raw_blocks, list) or len(raw_blocks) != len(order):
            raise ValueError('블록 데이터 오류')

        blocks, prev_end = [], -math.inf
        for i, (cond, b) in enumerate(zip(order, raw_blocks)):
            if not isinstance(b, dict) or b.get('cond') != cond:
                raise ValueError('블록 순서가 맞지 않습니다.')
            times = [b.get(k) for k in ('start_tp', 'rec_start_tp', 'rec_end_tp', 'end_tp')]
            if not all(_num(v) for v in times):
                raise ValueError('블록 시각 값 오류')
            start, rec_start, rec_end, end = map(float, times)
            expected = (N_PRE * trial_ms, N_REC * trial_ms, N_POST * trial_ms)
            actual = (rec_start - start, rec_end - rec_start, end - rec_end)
            if any(abs(a - e) > 5 for a, e in zip(actual, expected)) or start < prev_end:
                raise ValueError('블록 구간 시각이 설정과 맞지 않습니다.')
            prev_end = end
            trials = None
            if cond == '1back':
                raw_trials = b.get('trials')
                if not isinstance(raw_trials, list) or len(raw_trials) != N_TOTAL:
                    raise ValueError('1-back 시행 데이터 오류')
                trials = []
                for t in raw_trials:
                    sv = str(t.get('s', '')) if isinstance(t, dict) else ''
                    if len(sv) != 1 or sv not in '123456789':
                        raise ValueError('1-back 자극 값 오류')
                    responded = t.get('r') is True
                    rt = t.get('rt')
                    rt = round(float(rt), 1) if responded and _num(rt) and 0 <= rt <= 6000 else None
                    trials.append({'s': sv, 'r': responded, 'rt': rt})
            blocks.append({'cond': cond, 'order': i + 1, 'start': start, 'rec_start': rec_start,
                           'rec_end': rec_end, 'end': end, 'trials': trials})

        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None or sess.run_id != run_id:
                raise ValueError('측정 세션이 바뀌었습니다. 다시 측정하세요.')
            done = next((r for r in sess.results if r.get('task_id') == task_id), None)
            if done:
                return done
            i0 = sess.run_start_idx
            tp = np.asarray(sess.tp[i0:]); r = np.asarray(sess.r[i0:]); g = np.asarray(sess.g[i0:])
            offset = sess.offset_ms
        last_rec_end = max(b['rec_end'] for b in blocks)
        if (len(tp) == 0 or tp[-1] < last_rec_end + 2000) and waited_s < 20:
            return None                                  # 마지막 기록 구간 이후 2초까지 도착을 기다림

        result = {'task_id': task_id, 'participant_id': pid, 'trial_ms': trial_ms, 'order': list(order),
                  'blocks': {}, 'tx_id': ''}
        for b in blocks:
            result['blocks'][b['cond']] = self._block_result(tp, r, g, b, offset)
        with self.lock:
            sess.results.append(result)
        return result

    @staticmethod
    def _block_result(tp, r, g, b, offset) -> dict:
        rec_start, rec_end = b['rec_start'], b['rec_end']
        out = {
            'condition': b['cond'], 'order': b['order'],
            'rec_start_utc': utc_iso(rec_start + offset), 'rec_duration_s': round((rec_end - rec_start) / 1000, 1),
            'ibi_ms': [], 'valid': [], 'ibi_t_s': [], 'first_beat_ms': None,
            'hr_bpm': None, 'mean_ibi_ms': None, 'sdnn_ms': None, 'rmssd_ms': None,
            'beats': 0, 'excluded': 0, 'coverage': 0.0, 'quality': None, 'warning': '',
        }
        if b['trials'] is not None:
            out['nback'] = score_nback(b['trials'], 1)
            out['trials'] = b['trials']
        mask = (tp >= rec_start - 20_000) & (tp <= b['end'] + 2000)   # 필터 안정화를 위해 앞부분 포함
        res = analyze(tp[mask], r[mask], g[mask]) if mask.sum() > 1 else None
        if res is None:
            out['warning'] = '신호가 부족해 IBI를 계산하지 못했습니다.'
            return out
        pk = res['peak_ms']
        sel = (pk[:-1] >= rec_start) & (pk[1:] <= rec_end)
        ibi, valid = res['ibi_ms'][sel], res['valid'][sel]
        stats = hrv_stats(ibi, valid)
        coverage = float(ibi[valid].sum() / (rec_end - rec_start)) if len(ibi) else 0.0
        out.update({
            'ibi_ms': [round(float(v), 1) for v in ibi], 'valid': [bool(v) for v in valid],
            'ibi_t_s': [round(float((e - rec_start) / 1000), 2) for e in pk[1:][sel]],
            'first_beat_ms': round(float(pk[:-1][sel][0] - rec_start), 1) if sel.any() else None,
            'hr_bpm': stats['hr_bpm'], 'mean_ibi_ms': stats['ibi_ms'], 'sdnn_ms': stats['sdnn_ms'],
            'rmssd_ms': stats['rmssd_ms'], 'beats': stats['beats'], 'excluded': int((~valid).sum()),
            'coverage': round(coverage, 3), 'quality': round(res['quality'], 3),
        })
        if coverage < 0.8:
            out['warning'] = f'기록 구간의 {round((1 - coverage) * 100)}%에서 박동을 신뢰성 있게 검출하지 못했습니다.'
        return out

    def get_result(self, pid: str, task_id: str) -> dict | None:
        with self.lock:
            sess = self.sessions.get(pid)
            return next((r for r in sess.results if r['task_id'] == task_id), None) if sess else None

    def set_tx(self, pid: str, task_id: str, tx_id: str):
        with self.lock:
            for r in self.sessions[pid].results:
                if r['task_id'] == task_id:
                    r['tx_id'] = tx_id

    def session_list(self) -> list[dict]:
        now = now_ms()
        with self.lock:
            return [{'pid': s.pid, 'connected': s.active and now - s.last_seen_ms < 3500}
                    for s in sorted(self.sessions.values(), key=lambda s: s.pid)]

    def all_results(self) -> list[dict]:
        with self.lock:
            return [dict(r) for s in self.sessions.values() for r in s.results]

    # ---- CSV ----
    def csv_raw(self, pid: str) -> bytes:
        with self.lock:
            s = self.sessions[pid]
            rows = list(zip(s.t, s.tp, s.r, s.g))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(['t_server_ms', 't_phone_ms', 'red', 'green'])
        w.writerows((round(a, 1), round(b, 1), round(c, 3), round(d, 3)) for a, b, c, d in rows)
        return buf.getvalue().encode('utf-8')


def csv_bytes(header: list[str], rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return ('﻿' + buf.getvalue()).encode('utf-8')


def _blocks(results):
    for r in results:
        for cond in r['order']:
            yield r, r['blocks'][cond]


def csv_summary(results: list[dict]) -> bytes:
    cols = ['rec_start_utc', 'rec_duration_s', 'hr_bpm', 'mean_ibi_ms', 'sdnn_ms', 'rmssd_ms',
            'beats', 'excluded', 'coverage', 'quality']
    nb = ['hit', 'miss', 'false_alarm', 'correct_rejection', 'accuracy', 'd_prime', 'mean_rt_hit_ms']
    return csv_bytes(['participant_id', 'task_id', 'condition', 'order', 'trial_ms'] + cols
                     + ['nback_' + k for k in nb] + ['tx_id', 'warning'],
                     ([r['participant_id'], r['task_id'], b['condition'], b['order'], r['trial_ms']]
                      + [b[c] for c in cols] + [b['nback'][k] if 'nback' in b else '' for k in nb]
                      + [r['tx_id'], b['warning']] for r, b in _blocks(results)))


def csv_ibi(results: list[dict]) -> bytes:
    return csv_bytes(['participant_id', 'task_id', 'condition', 'order', 'beat', 't_from_rec_start_s', 'ibi_ms', 'valid'],
                     ([r['participant_id'], r['task_id'], b['condition'], b['order'], i + 1, t, v, ok]
                      for r, b in _blocks(results)
                      for i, (t, v, ok) in enumerate(zip(b['ibi_t_s'], b['ibi_ms'], b['valid']))))


def csv_trials(results: list[dict]) -> bytes:
    def phase(i):
        return 'pre' if i < N_PRE else 'rec' if i < N_PRE + N_REC else 'post'
    return csv_bytes(['participant_id', 'task_id', 'order', 'trial', 'phase', 'stimulus', 'is_target', 'responded', 'rt_ms'],
                     ([r['participant_id'], r['task_id'], b['order'], i + 1, phase(i), t['s'], t.get('target'), t['r'], t['rt']]
                      for r, b in _blocks(results) if 'trials' in b for i, t in enumerate(b['trials'])))


def _code_version() -> str:
    """이 파일 내용의 해시. 코드가 바뀌면 캐시된 저장소/컴포넌트를 새로 만든다.
    (st.cache_resource는 함수 본문만 보고 캐시를 유지하므로, 클래스가 바뀌어도
     예전 버전 객체가 남아 TypeError 등이 날 수 있다.)"""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    except (NameError, OSError):
        return SCHEMA


CODE_VERSION = _code_version()


@st.cache_resource(max_entries=1)
def _store_for(version: str) -> Store:
    return Store()


def get_store() -> Store:
    return _store_for(CODE_VERSION)


# ============================================================
# 3. XperChain 저장
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


def build_tx_data(record: dict) -> dict:
    text = json.dumps(record, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return {'contract_addr': CONTRACT_ADDR, 'method': 'upload_text', 'args': {'title': TITLE, 'text': text}}


def tx_data_size(tx_data: dict) -> int:
    return len(json.dumps(tx_data, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


# ---- IBI 압축 (트랜잭션 하나에 두 블록을 담기 위해) ----
#   IBI를 정수 ms로 반올림 → 첫 값은 varint(ibi), 다음부터 varint(zigzag(ibi - 직전 ibi))
#   → 바이트열을 Base85(RFC 1924, Python base64.b85encode)로 문자열화.
#   차이가 ±63 ms 이내면 1바이트, 그 이상이면 2바이트. 제외된 박동 번호는 따로 "x" 목록에 둔다.
def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def encode_ibis(ibi_ms) -> str:
    buf, prev = bytearray(), None
    for v in ibi_ms:
        v = int(round(v))
        if prev is None:
            buf += _varint(v)
        else:
            d = v - prev
            buf += _varint(d * 2 if d >= 0 else -d * 2 - 1)
        prev = v
    return base64.b85encode(bytes(buf)).decode('ascii')


def decode_ibis(text: str, max_beats: int = 2000) -> list[int]:
    data = base64.b85decode(text.encode('ascii'))
    values, prev, i = [], None, 0
    while i < len(data):
        n, shift = 0, 0
        while True:
            if i >= len(data) or shift > 28:
                raise ValueError('IBI 데이터가 손상되었습니다.')
            b = data[i]; i += 1
            n |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                break
        v = n if prev is None else prev + (n // 2 if n % 2 == 0 else -(n + 1) // 2)
        values.append(v); prev = v
        if len(values) > max_beats:
            raise ValueError('IBI 개수가 너무 많습니다.')
    return values


def build_record(result: dict) -> dict:
    """휴식 + 1-back을 한 기록으로. 요약 통계는 IBI에서 다시 계산할 수 있으므로 넣지 않는다."""
    first = result['blocks'][result['order'][0]]
    t0 = datetime.fromisoformat(first['rec_start_utc'])
    rec = {'schema': SCHEMA, 'mid': result['task_id'][:12], 'pid': result['participant_id'],
           'ms': result['trial_ms'], 'o': ''.join(COND_KEY[c] for c in result['order']),
           't': first['rec_start_utc'][:19] + 'Z'}
    for cond in result['order']:
        b = result['blocks'][cond]
        blk = {}
        d = int(round((datetime.fromisoformat(b['rec_start_utc']) - t0).total_seconds() * 1000))
        if d:
            blk['d'] = d                                          # 첫 블록 기록 시작부터의 간격(ms)
        if b['first_beat_ms'] is not None:
            blk['b0'] = int(round(b['first_beat_ms']))
        blk['ibi'] = encode_ibis(b['ibi_ms'])
        excluded = [i for i, ok in enumerate(b['valid']) if not ok]
        if excluded:
            blk['x'] = excluded
        if 'nback' in b:
            nb, rt = b['nback'], b['nback']['mean_rt_hit_ms']
            blk['nb'] = [nb['hit'], nb['miss'], nb['false_alarm'], nb['correct_rejection'],
                         int(round(rt)) if rt is not None else None]
        rec[COND_KEY[cond]] = blk
    return rec


def record_size(result: dict) -> int:
    return tx_data_size(build_tx_data(build_record(result)))


def chain_texts() -> list[str]:
    """컨트랙트 상태(확정)와 mempool(대기)의 upload_text 내용."""
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
    return texts


def used_mids() -> set[str]:
    mids = set()
    for text in chain_texts():
        try:
            c = json.loads(text)
            if isinstance(c, dict) and c.get('schema') == SCHEMA and isinstance(c.get('mid'), str):
                mids.add(c['mid'])
        except (TypeError, ValueError):
            continue
    return mids


def save_to_blockchain(store: Store, result: dict) -> str:
    if all(b['hr_bpm'] is None for b in result['blocks'].values()):
        raise NodeError('IBI 결과가 없는 측정은 저장할 수 없습니다.')
    if result.get('tx_id'):
        raise NodeError('이 측정은 이미 블록체인에 저장 요청되었습니다.')
    allowed = config('ALLOWED_PARTICIPANT_CODES')
    if allowed and result['participant_id'] not in {v.strip().upper() for v in allowed.split(',')}:
        raise NodeError('이 참가자 코드는 저장이 허용되지 않았습니다.')
    record = build_record(result)
    tx_data = build_tx_data(record)
    size = tx_data_size(tx_data)
    if size > MAX_TX_DATA_BYTES:
        raise NodeError(f'기록 크기 {size}바이트가 노드 제한 {MAX_TX_DATA_BYTES}바이트를 넘습니다. '
                        '설정에서 숫자 간격을 줄여 다시 측정하세요.')
    key = service_signing_key()
    wallet, public = wallet_address(key)
    with store.submit_lock:
        if record['mid'] in used_mids():
            raise NodeError('이 측정은 이미 블록체인에 저장(또는 대기)되어 있습니다.')
        nonce = node_request('GET', f'/account/{wallet}/nonce')
        fee = node_request('POST', '/fee/estimate', json={'tx_type': 'CONTRACT_CALL', 'data': tx_data})
        tx = {'sender': wallet, 'recipient': CONTRACT_ADDR, 'amount_xbit': 0,
              'fee_xbit': int(fee['required_fee_xbit']), 'nonce': int(nonce['next_nonce']),
              'timestamp': time.time(), 'tx_type': 'CONTRACT_CALL', 'data': tx_data}
        sig = key.sign(canonical_tx_bytes(tx))
        res = node_request('POST', '/contract/call', json={
            'contract_addr': CONTRACT_ADDR, 'method': 'upload_text', 'args': tx_data['args'],
            'sender': wallet, 'sender_pub_b64': base64.b64encode(public).decode('ascii'),
            'sig_b64': base64.b64encode(sig).decode('ascii'), 'value_xbit': 0,
            'fee_xbit': tx['fee_xbit'], 'nonce': tx['nonce'], 'timestamp': tx['timestamp']})
    tx_id = str(res.get('tx_id', ''))
    if not TX_ID_RE.fullmatch(tx_id):
        raise NodeError('노드가 올바른 트랜잭션 ID를 반환하지 않았습니다.')
    store.set_tx(result['participant_id'], result['task_id'], tx_id)
    return tx_id


# ---- 블록체인 보기: 컨트랙트 전체 기록 읽기 / 해석 / 엑셀 ----
def fetch_contract_items() -> list[dict]:
    if not ADDRESS_RE.fullmatch(CONTRACT_ADDR):
        raise NodeError('XPER_CONTRACT_ADDRESS 설정이 올바르지 않습니다.')
    snap = node_request('GET', f'/contract/{CONTRACT_ADDR}')
    items = (snap.get('state') or {}).get('data_items', [])
    return [i for i in items if isinstance(i, dict)] if isinstance(items, list) else []


def _at_utc(value) -> str:
    """블록 시각(at): 초 또는 ms 숫자, 또는 ISO 문자열."""
    if _num(value):
        sec = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(sec, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        except (OverflowError, OSError, ValueError):
            return ''
    return str(value)[:40] if value is not None else ''


def _safe_cell(v):
    """엑셀(XML)에 쓸 수 없는 제어 문자 제거. 수식 방지는 excel_bytes에서 셀 형식으로 처리."""
    if isinstance(v, str):
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE   # openpyxl 없으면 ImportError → CSV로 대체
        return ILLEGAL_CHARACTERS_RE.sub('', v)
    return v


def _check_ibis(ibi) -> list[int]:
    if any(not isinstance(v, int) or not 200 <= v <= 3000 for v in ibi):
        raise ValueError('IBI 값 범위 오류 (손상되었거나 다른 형식의 기록)')
    return ibi


def parse_items(items: list[dict]) -> dict:
    """전체 기록 표 + PPG 기록(현재/이전 형식) 해석 결과."""
    raw, blocks, beats = [], [], []
    legacy: dict[tuple, dict] = {}          # 이전 다중 조각 형식 모으기
    for it in items:
        text = it.get('text', '')
        text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        try:
            content = json.loads(text)
        except (TypeError, ValueError):
            content = None
        schema = content.get('schema', '') if isinstance(content, dict) else ''
        raw.append({'index': it.get('index'), 'record_id': str(it.get('id', '')), 'block_time_utc': _at_utc(it.get('at')),
                    'sender': str(it.get('by', '')), 'title': str(it.get('title', '')),
                    'schema': str(schema), 'text': text})
        if not isinstance(content, dict):
            continue
        try:
            if schema == SCHEMA:
                pid, mid = validate_pid(content.get('pid')), str(content.get('mid', ''))[:32]
                t0 = datetime.fromisoformat(str(content.get('t', '')).replace('Z', '+00:00'))
                for k, key in enumerate(str(content.get('o', ''))[:2], start=1):
                    blk, cond = content.get(key), KEY_COND.get(key)
                    if not isinstance(blk, dict) or cond is None:
                        continue
                    ibi = _check_ibis(decode_ibis(str(blk.get('ibi', ''))))
                    excluded = {i for i in blk.get('x', []) if isinstance(i, int)}
                    valid = [i not in excluded for i in range(len(ibi))]
                    d = blk.get('d', 0) if _num(blk.get('d', 0)) else 0
                    t = (t0 + timedelta(milliseconds=d)).strftime('%Y-%m-%dT%H:%M:%SZ')
                    nb = blk.get('nb') if isinstance(blk.get('nb'), list) else None
                    _add_block(blocks, beats, it, schema, mid, pid, cond, k, t, content.get('ms'), ibi, valid, nb)
            elif schema in ('ppg_rest_1back_v1', 'ppg_1back_v1'):
                cond = content.get('c', '1back') if schema == 'ppg_rest_1back_v1' else '1back'
                key = (schema, str(content.get('mid', '')), cond)
                g = legacy.setdefault(key, {'item': it, 'parts': {}, 'head': None})
                g['parts'][int(content.get('p', 1))] = [int(v) for v in content.get('ibi', [])]
                if int(content.get('p', 1)) == 1:
                    g['head'] = content
        except (ValueError, TypeError, KeyError):
            continue
    for (schema, mid, cond), g in legacy.items():
        h = g['head'] or {}
        try:
            pid = validate_pid(h.get('pid'))
        except ValueError:
            continue
        ibis = [v for p in sorted(g['parts']) for v in g['parts'][p]]
        try:
            _check_ibis([abs(v) for v in ibis])
        except ValueError:
            continue
        nb = h.get('nb')[:5] if isinstance(h.get('nb'), list) else None
        _add_block(blocks, beats, g['item'], schema, mid, pid, cond, h.get('o', 1), h.get('t'),
                   (h.get('nb') or [None] * 6)[5] if isinstance(h.get('nb'), list) and len(h['nb']) > 5 else None,
                   [abs(v) for v in ibis], [v > 0 for v in ibis], nb)
    return {'raw': raw, 'blocks': blocks, 'beats': beats}


def _add_block(blocks, beats, item, schema, mid, pid, cond, order, t, trial_ms, ibi, valid, nb):
    stats = hrv_stats(ibi, valid) if ibi else hrv_stats([], [])
    nb = nb if isinstance(nb, list) and len(nb) >= 4 else None
    blocks.append({
        'participant_id': pid, 'measurement_id': mid, 'condition': cond, 'order': order,
        'rec_start_utc': str(t or '')[:20], 'trial_ms': trial_ms, 'beats_valid': stats['beats'],
        'beats_excluded': len(ibi) - stats['beats'], 'hr_bpm': stats['hr_bpm'], 'mean_ibi_ms': stats['ibi_ms'],
        'sdnn_ms': stats['sdnn_ms'], 'rmssd_ms': stats['rmssd_ms'],
        'nback_hit': nb[0] if nb else None, 'nback_miss': nb[1] if nb else None,
        'nback_false_alarm': nb[2] if nb else None, 'nback_correct_rejection': nb[3] if nb else None,
        'nback_mean_rt_ms': nb[4] if nb and len(nb) > 4 else None,
        'record_index': item.get('index'), 'block_time_utc': _at_utc(item.get('at')), 'schema': schema,
    })
    for k, (v, ok) in enumerate(zip(ibi, valid), start=1):
        beats.append({'participant_id': pid, 'measurement_id': mid, 'condition': cond, 'order': order,
                      'beat': k, 'ibi_ms': v, 'valid': ok})


def rows_csv(rows: list[dict]) -> bytes:
    """엑셀 대신 쓰는 CSV. 체인의 문자열은 신뢰할 수 없으므로 수식으로 해석될 셀은 ' 를 붙인다."""
    if not rows:
        return csv_bytes(['(없음)'], [])
    header = list(rows[0])
    def cell(v):
        return "'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@', '\t', '\r') else v
    return csv_bytes(header, ([cell(r.get(k)) for k in header] for r in rows))


def excel_bytes(parsed: dict) -> bytes:
    import openpyxl  # noqa: F401  (없으면 ImportError → 화면에서 CSV로 대체)
    import pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as xw:
        for name, rows in (('PPG 요약', parsed['blocks']), ('PPG IBI', parsed['beats']), ('전체 기록', parsed['raw'])):
            df = pd.DataFrame(rows) if rows else pd.DataFrame({'(없음)': []})
            df = df.apply(lambda col: col if pd.api.types.is_numeric_dtype(col) else col.map(_safe_cell))
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            for row in ws.iter_rows():          # 체인의 '=...' 문자열이 수식으로 실행되지 않게 텍스트로 고정
                for c in row:
                    if c.data_type == 'f':
                        c.data_type = 's'
            for col in ws.columns:              # 열 너비 대충 맞추기
                width = max(len(str(c.value or '')) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(60, max(8, width + 2))
            ws.freeze_panes = 'A2'
    return buf.getvalue()


# ============================================================
# 4. 폰 컴포넌트 (카메라 PPG + 1-back)
# ============================================================
COMPONENT_HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<style>
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;
         color:#eef1f4; background:#0f1216; user-select:none; -webkit-user-select:none; }
  .wrap { padding:16px 14px 20px; }
  button { width:100%; margin-top:12px; padding:16px; font-size:18px; border:0; border-radius:10px;
           background:#3b82f6; color:#fff; touch-action:manipulation; }
  button.secondary { background:#374151; }
  button:disabled { opacity:.4; }
  .hint { color:#8a96a3; font-size:14px; line-height:1.6; margin-top:10px; text-align:center; }
  .label { color:#8a96a3; font-size:13px; margin:12px 0 4px; }
  .row { display:flex; align-items:center; gap:10px; font-size:16px; }
  .dot { width:18px; height:18px; border-radius:50%; background:#4b5563; flex:none; }
  canvas { width:100%; display:block; background:#12161b; border-radius:8px; }
  #err { color:#f87171; margin-top:10px; font-size:15px; text-align:center; }
  .hidden { display:none !important; }
  #blockTitle { text-align:center; font-size:15px; color:#cbd5e1; }
  #progress { color:#8a96a3; text-align:center; font-variant-numeric:tabular-nums; font-size:14px; min-height:20px; }
  #bar { height:4px; background:#1f2937; border-radius:2px; margin-top:6px; overflow:hidden; }
  #barFill { height:100%; width:0; background:#3b82f6; transition:width .25s linear; }
  #stimBox { height:220px; display:flex; align-items:center; justify-content:center; }
  #stim { font-size:150px; font-weight:700; line-height:1; }
  #fix { font-size:40px; color:#6b7280; }
  #taskMsg { text-align:center; font-size:18px; line-height:1.6; min-height:28px; }
  #matchBtn { height:110px; font-size:30px; font-weight:700; margin-top:16px; }
  #matchBtn.pressed { background:#94a3b8; }
  table { width:100%; border-collapse:collapse; font-size:15px; margin-top:6px; }
  td { padding:6px 4px; border-bottom:1px solid #2a313a; }
  td:not(:first-child) { text-align:right; font-variant-numeric:tabular-nums; }
  th { font-weight:500; color:#8a96a3; text-align:right; padding:4px; font-size:13px; }
  .warn { color:#fbbf24; font-size:14px; margin-top:8px; }
  video { position:absolute; width:1px; height:1px; opacity:0; pointer-events:none; }
</style></head>
<body><div class="wrap">
  <div id="v-idle">
    <div class="hint">휴대폰을 책상이나 받침대에 올려 두고,<br>왼손 검지로 <b>후면 카메라와 플래시를 함께</b> 가볍게 덮어 주세요.<br>너무 세게 누르면 신호가 약해집니다.</div>
    <button id="prepBtn">측정 준비</button>
  </div>

  <div id="v-prep" class="hidden">
    <div class="row"><span id="qdot" class="dot"></span><span id="qmsg">신호 확인 중...</span></div>
    <div class="label">PPG 신호 (최근 10초 · 빨간 점 = 검출된 박동)</div>
    <canvas id="cWave" height="140"></canvas>
    <div class="label">IBI (최근 60초) <span id="hrText"></span></div>
    <canvas id="cIbi" height="120"></canvas>
    <div class="hint">박동마다 빨간 점이 하나씩 찍히고 IBI 점이 고르게 이어지면 [측정 시작]을 누르세요.</div>
    <div class="hint" id="torchNote"></div>
    <div class="hint">측정은 <b>휴식</b>과 <b>1-back</b> 두 블록으로 진행되며, 순서는 무작위입니다.</div>
    <button id="startBtn">측정 시작</button>
    <button id="cancelBtn" class="secondary">취소</button>
  </div>

  <div id="v-task" class="hidden">
    <div id="blockTitle"></div>
    <div id="bar"><div id="barFill"></div></div>
    <div id="progress"></div>
    <div id="stimBox"><span id="stim"></span><span id="fix" class="hidden">+</span></div>
    <div id="taskMsg"></div>
    <button id="matchBtn">일치</button>
  </div>

  <div id="v-done" class="hidden">
    <h3 id="doneTitle" style="margin:4px 0 8px">결과 계산 중...</h3>
    <div id="summary"></div>
    <div class="label" id="resLabel"></div>
    <canvas id="cRes" height="120" class="hidden"></canvas>
    <div class="warn" id="warn"></div>
    <div class="hint" id="saveHint"></div>
    <button id="againBtn" class="secondary">다시 측정</button>
  </div>
  <div id="err"></div>
</div>
<video id="video" playsinline muted autoplay></video>
<script>
"use strict";
// ---- Streamlit 컴포넌트 프로토콜 ----
function post(type, extra) { window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, extra || {}), "*"); }
function setHeight() { post("streamlit:setFrameHeight", { height: document.documentElement.scrollHeight }); }
function setValue(v) { post("streamlit:setComponentValue", { value: v, dataType: "json" }); }

const $ = (id) => document.getElementById(id);
const W = 40, H = 30;
const video = $("video");
const grab = document.createElement("canvas"); grab.width = W; grab.height = H;
const gctx = grab.getContext("2d", { willReadFrequently: true });

let args = {}, mode = "idle";
let stream = null, track = null, sampling = false, queue = [], runId = "", seq = 0;
let sendTimer = null, fallbackTimer = null, wakeLock = null, lastMediaTime = -1;
let taskId = "", trialMs = 2500, nPre = 5, nRec = 10, nPost = 5, stimMs = 500;
let digits = [], trials = [], cur = null, t0 = 0, timers = [], taskPayload = null, gotResult = false;
let order = [], blocksOut = [], breakS = 5;
const LABEL = { rest: "휴식", "1back": "1-back" };
const COLOR = { rest: "#60a5fa", "1back": "#f59e0b" };

const QMSG = {
  warming:   ["#4b5563", "신호 확인 중... 손가락을 그대로 두세요"],
  good:      ["#22c55e", "신호 좋음"],
  fair:      ["#eab308", "신호 보통 — 손가락을 움직이지 마세요"],
  poor:      ["#ef4444", "신호 약함 — 손가락을 가볍게 다시 올려 주세요"],
  nofinger:  ["#6b7280", "손가락이 감지되지 않습니다 — 카메라와 플래시를 함께 덮어 주세요"],
  dark:      ["#6b7280", "너무 어둡습니다 — 플래시가 켜졌는지 확인하세요"],
  saturated: ["#ef4444", "너무 밝습니다 — 손가락으로 카메라를 완전히 덮어 주세요"],
};

function showView(v) {
  mode = v;
  for (const k of ["idle", "prep", "task", "done"]) $("v-" + k).classList.toggle("hidden", k !== v);
  setHeight();
}
function randomId() { const a = new Uint8Array(12); crypto.getRandomValues(a); return Array.from(a, b => b.toString(16).padStart(2, "0")).join(""); }
function epoch(perf) { return performance.timeOrigin + perf; }

// ---- Python → 컴포넌트 ----
window.addEventListener("message", (ev) => {
  const d = ev.data;
  if (!d || d.type !== "streamlit:render") return;
  args = d.args || {};
  if (args.run_id === runId && typeof args.ack === "number") queue = queue.filter(s => s[0] > args.ack);
  if (args.error && args.run_id === runId && mode !== "idle") { reset(); $("err").textContent = args.error; }
  if (mode === "prep") {
    const [c, m] = QMSG[args.quality] || QMSG.warming;
    $("qdot").style.background = c; $("qmsg").textContent = m;
    drawPreview(args.preview);
  }
  if (mode === "done" && args.result && args.result.task_id === taskId) {
    if (!gotResult) { gotResult = true; renderResult(args.result); }
    $("saveHint").textContent = args.saved && args.saved.task_id === taskId ? args.saved.text : "아래 [블록체인 저장] 버튼으로 결과를 저장할 수 있습니다.";
  }
  if (mode === "done" && args.result_error && args.result_error.task_id === taskId) {
    gotResult = true; $("doneTitle").textContent = "결과를 계산하지 못했습니다"; $("warn").textContent = args.result_error.text;
  }
  setHeight();
});

// ---- 카메라 ----
function sampleFrame(tsPerf) {
  gctx.drawImage(video, 0, 0, W, H);
  const d = gctx.getImageData(0, 0, W, H).data;
  let r = 0, g = 0;
  for (let i = 0; i < d.length; i += 4) { r += d[i]; g += d[i + 1]; }
  const n = d.length / 4;
  queue.push([+epoch(tsPerf).toFixed(1), +(r / n).toFixed(3), +(g / n).toFixed(3)]);
  if (queue.length > 9000) queue.splice(0, queue.length - 9000);
}
function onVideoFrame(now, meta) {
  if (!sampling) return;
  if (meta.mediaTime !== lastMediaTime) {
    lastMediaTime = meta.mediaTime;
    const ct = meta.captureTime;
    sampleFrame(ct && Math.abs(ct - now) < 1000 ? ct : now);
  }
  video.requestVideoFrameCallback(onVideoFrame);
}
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("이 브라우저에서는 카메라를 사용할 수 없습니다.");
  stream = await navigator.mediaDevices.getUserMedia({ audio: false,
    video: { facingMode: { ideal: "environment" }, width: { ideal: 320 }, height: { ideal: 240 }, frameRate: { ideal: 30, max: 30 } } });
  video.srcObject = stream; await video.play();
  track = stream.getVideoTracks()[0];
  try { await track.applyConstraints({ advanced: [{ torch: true }] }); $("torchNote").textContent = ""; }
  catch (e) { $("torchNote").textContent = "이 브라우저에서는 플래시를 켤 수 없습니다. Android는 Chrome을 사용하세요."; }
  sampling = true; lastMediaTime = -1;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) video.requestVideoFrameCallback(onVideoFrame);
  else fallbackTimer = setInterval(() => sampleFrame(performance.now()), 33);
  try { if ("wakeLock" in navigator) wakeLock = await navigator.wakeLock.request("screen"); } catch (e) {}
}
function stopCamera() {
  sampling = false;
  if (fallbackTimer) { clearInterval(fallbackTimer); fallbackTimer = null; }
  if (track) { try { track.applyConstraints({ advanced: [{ torch: false }] }); } catch (e) {} }
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = track = null;
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
}

// ---- 컴포넌트 → Python (1초마다, 아직 확인 안 된 샘플 전부) ----
function push() {
  if (mode === "idle") return;
  if (mode === "done" && gotResult && !queue.length) { clearInterval(sendTimer); sendTimer = null; return; }
  setValue({ run_id: runId, seq: ++seq, sent_at: epoch(performance.now()), mode,
             running: mode !== "done", samples: queue.slice(0, 1200), task: mode === "done" ? taskPayload : null });
}

// ---- 그래프 ----
function fit(c) {
  // 화면 높이(CSS px)는 처음 지정한 height 속성값으로 고정한다.
  // c.height(실제 픽셀 수)를 바꾸면 height 속성도 함께 바뀌므로, 매번 속성에서 다시 읽으면
  // 고해상도 화면(dpr 2~3)에서 높이가 커져 그래프 아래쪽이 잘린다.
  if (!c.dataset.h) c.dataset.h = c.getAttribute("height");
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth, h = +c.dataset.h;
  c.style.height = h + "px";
  const pw = Math.round(w * dpr), ph = Math.round(h * dpr);
  if (c.width !== pw || c.height !== ph) { c.width = pw; c.height = ph; }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return [g, w, h];
}
function drawWave(c, wave) {
  const [g, w, h] = fit(c);
  g.fillStyle = "#6b7280"; g.font = "11px system-ui";
  if (!wave || !wave.t.length) { g.fillText("데이터 수집 중...", 10, h / 2); return; }
  const X = t => w * (1 + t / 10), Y = y => h / 2 - y * h * 0.42;
  g.strokeStyle = "#60a5fa"; g.lineWidth = 2; g.beginPath();
  wave.t.forEach((t, i) => i ? g.lineTo(X(t), Y(wave.y[i])) : g.moveTo(X(t), Y(wave.y[i]))); g.stroke();
  g.fillStyle = "#ef4444";
  wave.pt.forEach((t, i) => { g.beginPath(); g.arc(X(t), Y(wave.py[i]), 4, 0, Math.PI * 2); g.fill(); });
}
function drawIbi(c, series, xmin, xmax) {
  // series: [{t:[], ms:[], valid:[], color}]
  const [g, w, h] = fit(c);
  const pad = 34, lo = 400, hi = 1400;
  const X = v => pad + (w - pad - 6) * (v - xmin) / (xmax - xmin);
  const Y = v => h - 8 - (h - 16) * (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo);
  g.strokeStyle = "#2a313a"; g.fillStyle = "#6b7280"; g.font = "10px system-ui";
  for (let v = 400; v <= 1400; v += 250) { g.beginPath(); g.moveTo(pad, Y(v)); g.lineTo(w, Y(v)); g.stroke(); g.fillText(v, 2, Y(v) + 3); }
  for (const s of series || []) {
    if (!s || !s.t || !s.t.length) continue;
    g.strokeStyle = s.color; g.lineWidth = 1.5; g.beginPath(); let on = false;
    s.t.forEach((x, i) => { if (!s.valid[i]) { on = false; return; } on ? g.lineTo(X(x), Y(s.ms[i])) : g.moveTo(X(x), Y(s.ms[i])); on = true; });
    g.stroke();
    s.t.forEach((x, i) => { g.fillStyle = s.valid[i] ? s.color : "#9ca3af"; g.beginPath(); g.arc(X(x), Y(s.ms[i]), 3, 0, Math.PI * 2); g.fill(); });
  }
}
function drawPreview(p) {
  drawWave($("cWave"), p && p.wave);
  drawIbi($("cIbi"), p ? [{ t: p.ibi.t, ms: p.ibi.ms, valid: p.ibi.valid, color: "#22c55e" }] : [], -60, 0);
  $("hrText").textContent = p && p.hr ? `· 현재 약 ${p.hr} bpm` : "";
}

// ---- 1-back ----
function makeSequence(total, n, rate) {
  const pool = ["1","2","3","4","5","6","7","8","9"];
  const idx = [...Array(total - n).keys()].map(i => i + n);
  for (let i = idx.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
  const targets = new Set(idx.slice(0, Math.round((total - n) * rate)));
  const out = [];
  for (let i = 0; i < total; i++) {
    if (targets.has(i)) { out.push(out[i - n]); continue; }
    let c; do { c = pool[Math.floor(Math.random() * pool.length)]; } while (i >= n && c === out[i - n]);
    out.push(c);
  }
  return out;
}
function later(fn, ms) { timers.push(setTimeout(fn, Math.max(0, ms))); }

function startSession() {
  trialMs = Math.round(args.trial_ms || 2500); nPre = args.n_pre || 5; nRec = args.n_rec || 10;
  nPost = args.n_post || 5; stimMs = args.stim_ms || 500; breakS = args.break_s || 5;
  const r = new Uint8Array(1); crypto.getRandomValues(r);
  order = (r[0] & 1) ? ["rest", "1back"] : ["1back", "rest"];      // 블록 순서 무작위
  taskId = randomId(); gotResult = false; taskPayload = null; blocksOut = []; cur = null;
  showView("task");
  runBlock(0);
}

function runBlock(bi) {
  const cond = order[bi], total = nPre + nRec + nPost;
  $("blockTitle").textContent = `블록 ${bi + 1} / ${order.length} · ${LABEL[cond]}`;
  $("barFill").style.width = "0%";
  $("stim").textContent = ""; $("fix").classList.add("hidden"); $("progress").textContent = "";
  $("matchBtn").classList.toggle("hidden", cond !== "1back");
  $("matchBtn").disabled = true;
  const guide = cond === "rest"
    ? "<b>휴식</b><br>화면 가운데 + 를 편안히 바라보세요.<br>손가락은 그대로 두고 움직이지 마세요."
    : "<b>1-back</b><br>숫자가 <b>바로 앞 숫자와 같으면</b> [일치]를 누르세요.";
  let k = breakS;
  const tickDown = () => {
    $("taskMsg").innerHTML = `${guide}<br><span style="font-size:42px">${k}</span>`;
    setHeight();
    if (k-- > 0) { later(tickDown, 1000); return; }
    $("taskMsg").textContent = ""; $("fix").classList.remove("hidden");
    t0 = performance.now();
    const blk = { cond, start_tp: epoch(t0), rec_start_tp: epoch(t0 + nPre * trialMs),
                  rec_end_tp: epoch(t0 + (nPre + nRec) * trialMs), end_tp: epoch(t0 + total * trialMs) };
    blocksOut.push(blk);
    if (cond === "1back") {
      digits = makeSequence(total, 1, 0.3); trials = [];
      $("matchBtn").disabled = false;
      runTrial(0, total, bi);
    } else {
      runRest(0, total, bi);
    }
  };
  tickDown();
}

function showProgress(i, total) {
  $("barFill").style.width = `${Math.round(100 * (i + 1) / total)}%`;
}

function runRest(i, total, bi) {
  if (i >= total) return blockDone(bi);
  showProgress(i, total);
  $("progress").textContent = `${Math.ceil((total - i) * trialMs / 1000)}초 남음`;
  later(() => runRest(i + 1, total, bi), t0 + (i + 1) * trialMs - performance.now());
}

function runTrial(i, total, bi) {
  if (i >= total) return blockDone(bi);
  const t = { s: digits[i], r: false, rt: null, on: null };
  trials.push(t); cur = t;
  showProgress(i, total);
  $("progress").textContent = `${i + 1} / ${total}`;
  $("fix").classList.add("hidden"); $("stim").textContent = digits[i];
  t.on = performance.now();
  later(() => { $("stim").textContent = ""; $("fix").classList.remove("hidden"); }, stimMs);
  later(() => runTrial(i + 1, total, bi), t0 + (i + 1) * trialMs - performance.now());   // 누적 지연 없음
}
$("matchBtn").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (mode !== "task" || !cur || cur.r || cur.on === null) return;
  cur.r = true; cur.rt = Math.max(0, performance.now() - cur.on);
  const b = $("matchBtn"); b.classList.add("pressed"); later(() => b.classList.remove("pressed"), 150);
});

function blockDone(bi) {
  cur = null;
  const blk = blocksOut[bi];
  if (blk.cond === "1back") blk.trials = trials.map(t => ({ s: t.s, r: t.r, rt: t.rt === null ? null : +t.rt.toFixed(1) }));
  if (bi + 1 < order.length) { runBlock(bi + 1); return; }
  finishSession();
}

function finishSession() {
  timers.forEach(clearTimeout); timers = [];
  stopCamera();
  taskPayload = { task_id: taskId, trial_ms: trialMs, order: order.slice(), blocks: blocksOut };
  $("doneTitle").textContent = "결과 계산 중..."; $("summary").replaceChildren();
  $("warn").textContent = ""; $("saveHint").textContent = ""; $("resLabel").textContent = "";
  $("cRes").classList.add("hidden");
  showView("done");
  push();
}

function fmt(v, unit) { return v == null ? "-" : `${v}${unit || ""}`; }
function renderResult(r) {
  $("doneTitle").textContent = "측정 완료";
  const R = r.blocks.rest || {}, B = r.blocks["1back"] || {}, nb = B.nback || {};
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const h of ["", "휴식", "1-back"]) { const th = document.createElement("th"); th.textContent = h; head.appendChild(th); }
  const rows = [
    ["진행 순서", r.order.map(c => LABEL[c]).join(" → "), ""],
    ["검출된 IBI", fmt(R.beats, "개"), fmt(B.beats, "개")],
    ["평균 심박", fmt(R.hr_bpm, " bpm"), fmt(B.hr_bpm, " bpm")],
    ["평균 IBI", fmt(R.mean_ibi_ms, " ms"), fmt(B.mean_ibi_ms, " ms")],
    ["SDNN", fmt(R.sdnn_ms, " ms"), fmt(B.sdnn_ms, " ms")],
    ["RMSSD", fmt(R.rmssd_ms, " ms"), fmt(B.rmssd_ms, " ms")],
    ["1-back 정확도", "", nb.accuracy == null ? "-" : `${Math.round(nb.accuracy * 100)}%`],
    ["적중 / 오경보", "", nb.hit == null ? "-" : `${nb.hit}/${nb.hit + nb.miss} · ${nb.false_alarm}회`],
    ["평균 반응시간", "", nb.mean_rt_hit_ms == null ? "-" : `${Math.round(nb.mean_rt_hit_ms)} ms`],
  ];
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    if (row[0] === "진행 순서") { tr.insertCell().textContent = row[0]; const td = tr.insertCell(); td.colSpan = 2; td.textContent = row[1]; continue; }
    row.forEach(v => { tr.insertCell().textContent = v; });
  }
  $("summary").replaceChildren(table);
  const series = r.order.map(c => {
    const b = r.blocks[c];
    return { t: b.ibi_t_s, ms: b.ibi_ms, valid: b.valid, color: COLOR[c] };
  });
  if (series.some(s => s.t.length)) {
    $("resLabel").textContent = "기록 구간 IBI — 파랑: 휴식, 주황: 1-back (회색 = 제외된 박동)";
    $("cRes").classList.remove("hidden");
    const dur = Math.max(R.rec_duration_s || 0, B.rec_duration_s || 0) || 25;
    drawIbi($("cRes"), series, 0, dur);
  }
  $("warn").textContent = r.order.map(c => r.blocks[c].warning ? `${LABEL[c]}: ${r.blocks[c].warning}` : "").filter(Boolean).join(" ");
  setHeight();
}

// ---- 버튼 ----
function reset() {
  timers.forEach(clearTimeout); timers = [];
  if (sendTimer) { clearInterval(sendTimer); sendTimer = null; }
  stopCamera(); queue = []; cur = null;
  showView("idle");
}
$("prepBtn").addEventListener("click", async () => {
  $("err").textContent = ""; $("prepBtn").disabled = true;
  try {
    runId = randomId(); seq = 0; queue = [];
    await startCamera();
    showView("prep"); drawPreview(null);
    sendTimer = setInterval(push, 1000);
  } catch (e) {
    reset();
    $("err").textContent = e && e.name === "NotAllowedError" ? "카메라 권한을 허용해 주세요." : (e.message || String(e));
  } finally { $("prepBtn").disabled = false; }
});
$("startBtn").addEventListener("click", startSession);
$("cancelBtn").addEventListener("click", () => { setValue({ run_id: runId, seq: ++seq, sent_at: epoch(performance.now()), mode: "idle", running: false, samples: [], task: null }); reset(); });
$("againBtn").addEventListener("click", reset);
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible" && (mode === "prep" || mode === "task"))
    $("err").textContent = "화면이 꺼지거나 다른 앱으로 전환되면 측정이 멈춥니다.";
});
window.addEventListener("resize", () => { if (mode === "prep") drawPreview(args.preview); });
post("streamlit:componentReady", { apiVersion: 1 });
setHeight();
</script></body></html>
"""


@st.cache_resource(max_entries=1)
def _component_for(version: str):
    """HTML을 임시 폴더에 써서 양방향 컴포넌트로 등록 (빌드 도구 불필요).
    버전별 폴더를 써서 코드가 바뀌면 브라우저도 새 HTML을 받는다."""
    import streamlit.components.v1 as components
    folder = Path(tempfile.gettempdir()) / f'ppg_1back_component_{version}'
    folder.mkdir(exist_ok=True)
    index = folder / 'index.html'
    if not index.exists() or index.read_text(encoding='utf-8') != COMPONENT_HTML:
        index.write_text(COMPONENT_HTML, encoding='utf-8')
    return components.declare_component('ppg_1back', path=str(folder))


def ppg_component():
    return _component_for(CODE_VERSION)


def public_result(result: dict) -> dict:
    """컴포넌트로 보낼 결과 (1-back 시행 목록과 저장 상태는 제외)."""
    return {
        'task_id': result['task_id'], 'order': result['order'], 'trial_ms': result['trial_ms'],
        'blocks': {c: {k: v for k, v in b.items() if k != 'trials'} for c, b in result['blocks'].items()},
    }


def make_preview(live: dict | None) -> dict | None:
    """측정 준비 화면용: 최근 10초 파형 + 박동, 최근 60초 IBI."""
    if not live or live['res'] is None:
        return None
    res, end = live['res'], live['end_ms']
    w = res['grid_ms'] >= end - 10_000
    y = res['filt'][w]
    scale = float(np.max(np.abs(y))) or 1.0
    pk = res['peak_ms'] >= end - 10_000
    return {
        'wave': {'t': np.round((res['grid_ms'][w] - end) / 1000, 3).tolist(),
                 'y': np.round(y / scale, 3).tolist(),
                 'pt': np.round((res['peak_ms'][pk] - end) / 1000, 3).tolist(),
                 'py': np.round(np.interp(res['peak_ms'][pk], res['grid_ms'], res['filt']) / scale, 3).tolist()},
        'ibi': {'t': np.round((res['ibi_end_ms'] - end) / 1000, 2).tolist(),
                'ms': np.round(res['ibi_ms'], 1).tolist(), 'valid': [bool(v) for v in res['valid']]},
        'hr': live['hr15']['hr_bpm'],
    }


# ============================================================
# 5. 화면
# ============================================================
@st.cache_data(ttl=20, show_spinner=False)
def _load_chain(version: str, nonce: int):
    items = fetch_contract_items()
    return parse_items(items)


def chain_page():
    st.markdown(f"### {CHAIN_ICON} 블록체인 기록")
    st.caption(f"컨트랙트 `{CONTRACT_ADDR}` · 노드 `{NODE_URL}`")
    if st.button("🔄 새로 불러오기"):
        st.session_state['chain_nonce'] = st.session_state.get('chain_nonce', 0) + 1
    try:
        with st.spinner("블록체인에서 기록을 불러오는 중..."):
            parsed = _load_chain(CODE_VERSION, st.session_state.get('chain_nonce', 0))
    except NodeError as exc:
        st.error(str(exc))
        return
    import pandas as pd
    raw, blocks, beats = parsed['raw'], parsed['blocks'], parsed['beats']
    m = st.columns(3)
    m[0].metric("전체 기록", len(raw))
    m[1].metric("PPG 측정", len({(b['measurement_id'], b['participant_id']) for b in blocks}))
    m[2].metric("참가자", len({b['participant_id'] for b in blocks}))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')
    try:
        xlsx = excel_bytes(parsed)
    except ImportError:
        xlsx = None
    if xlsx is not None:
        st.download_button("📥 엑셀로 내려받기 (.xlsx)", xlsx, f"blockchain_{CONTRACT_ADDR[:8]}_{stamp}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        st.caption("엑셀 시트: PPG 요약(블록별) · PPG IBI(박동별, 압축 해제) · 전체 기록(컨트랙트의 모든 원문)")
    else:
        # openpyxl이 없으면(requirements.txt 누락) 같은 내용을 CSV 3개로 제공
        st.warning("서버에 openpyxl이 설치되지 않아 엑셀 파일을 만들 수 없습니다. "
                   "requirements.txt에 openpyxl을 추가하세요. 우선 같은 내용을 CSV로 내려받을 수 있습니다.")
        c1, c2, c3 = st.columns(3)
        for col, (label, rows, name) in zip((c1, c2, c3), (("PPG 요약 CSV", blocks, 'ppg_summary'),
                                                          ("PPG IBI CSV", beats, 'ppg_ibi'),
                                                          ("전체 기록 CSV", raw, 'all_records'))):
            col.download_button(label, rows_csv(rows), f"blockchain_{name}_{stamp}.csv", "text/csv")

    st.subheader("PPG 측정 (블록별)")
    if blocks:
        df = pd.DataFrame(blocks)
        pids = sorted(df['participant_id'].unique())
        chosen = st.multiselect("참가자 필터", pids)
        if chosen:
            df = df[df['participant_id'].isin(chosen)]
        df['condition'] = df['condition'].map(COND_LABEL).fillna(df['condition'])
        st.dataframe(df[['participant_id', 'measurement_id', 'condition', 'order', 'rec_start_utc', 'hr_bpm',
                         'mean_ibi_ms', 'sdnn_ms', 'rmssd_ms', 'beats_valid', 'beats_excluded', 'nback_hit',
                         'nback_miss', 'nback_false_alarm', 'nback_mean_rt_ms', 'block_time_utc', 'schema']],
                     hide_index=True)
    else:
        st.caption("PPG 기록이 없습니다.")

    st.subheader("전체 기록 (원문)")
    if raw:
        st.dataframe(pd.DataFrame(raw)[['index', 'block_time_utc', 'schema', 'title', 'sender', 'text']], hide_index=True)
    else:
        st.caption("컨트랙트에 기록이 없습니다.")


def phone_page():
    st.markdown(f"### {MEASURE_ICON} {MEASURE_NAME} (휴식 / 1-back)")
    pid_raw = st.text_input("참가자 코드", placeholder="P01", max_chars=4)
    with st.expander("설정"):
        trial_s = st.number_input("숫자 간격 (초)", 1.0, 4.0, DEFAULT_TRIAL_MS / 1000, 0.5,
                                  help=f"블록마다 IBI 기록 구간 = {N_REC}칸 × 간격 (휴식 블록도 같은 길이). 두 블록을 트랜잭션 1개(526바이트)에 담기 위해 최대 4초.")
        st.caption(f"블록당 {N_TOTAL}칸 ({N_TOTAL * trial_s:.0f}초), IBI 기록 {N_REC * trial_s:.0f}초 · "
                   f"휴식 + 1-back 2개 블록, 순서 무작위 · 총 약 {2 * (N_TOTAL * trial_s + BREAK_S):.0f}초")
    if not pid_raw:
        st.caption("실험자가 알려준 참가자 코드를 입력하세요.")
        return
    try:
        pid = validate_pid(pid_raw)
    except ValueError as exc:
        st.error(str(exc))
        return
    phone_panel(pid, int(trial_s * 1000))


@st.fragment
def phone_panel(pid: str, trial_ms: int):
    """컴포넌트 값이 바뀔 때(1초마다) 이 조각만 다시 실행된다."""
    store = get_store()
    state = st.session_state.setdefault(f'phone_{pid}', {})
    component = ppg_component()
    value = component(
        run_id=state.get('run_id', ''), ack=state.get('ack', 0.0), quality=state.get('quality', 'warming'),
        error=state.get('error', ''), preview=state.get('preview'), result=state.get('result_public'),
        result_error=state.get('result_error'), saved=state.get('saved'),
        trial_ms=trial_ms, n_pre=N_PRE, n_rec=N_REC, n_post=N_POST, stim_ms=STIM_MS, break_s=BREAK_S,
        key=f'ppg_{pid}', default=None)

    if isinstance(value, dict) and (value.get('run_id'), value.get('seq')) != state.get('last'):
        state['last'] = (value.get('run_id'), value.get('seq'))
        state['run_id'] = value.get('run_id', '')
        try:
            out = store.ingest(pid, value)
            state.update(ack=out['ack'], quality=out['quality'], error='')
        except IngestError as exc:
            state['error'] = str(exc)
        mode = value.get('mode')
        state['preview'] = make_preview(store.live(pid, 60_000)) if mode == 'prep' and not state['error'] else None
        task = value.get('task')
        if mode == 'done' and isinstance(task, dict) and not state['error']:
            tid = task.get('task_id')
            if state.get('task_id') != tid:              # 새 측정 → 이전 결과 초기화
                state.update(task_id=tid, task_first_seen=time.time(), result_public=None,
                             result_error=None, saved=None)
            if state.get('result_public') is None and state.get('result_error') is None:
                try:
                    result = store.compute_result(pid, state['run_id'], task, time.time() - state['task_first_seen'])
                    if result is not None:
                        state['result_public'] = public_result(result)
                except ValueError as exc:
                    state['result_error'] = {'task_id': tid, 'text': str(exc)}

    # ---- 결과 아래: 블록체인 저장 ----
    pub = state.get('result_public')
    if not pub:
        return
    result = store.get_result(pid, pub['task_id'])
    if result is None:
        return
    st.divider()
    size = record_size(result)
    if result['tx_id']:
        st.success("블록체인 저장 요청 완료 (트랜잭션 1개)")
        st.code(result['tx_id'], language=None)
        st.caption("상단의 [블록체인 보기]에서 기록을 확인할 수 있습니다. 블록이 만들어진 뒤에 나타납니다.")
    elif all(b['hr_bpm'] is None for b in result['blocks'].values()):
        st.warning("IBI 결과가 없어 저장할 수 없습니다. 다시 측정하세요.")
    elif size > MAX_TX_DATA_BYTES:
        st.error(f"기록 크기 {size}바이트가 노드 제한 {MAX_TX_DATA_BYTES}바이트를 넘어 트랜잭션 하나에 담을 수 없습니다. "
                 "설정에서 숫자 간격을 줄여 다시 측정하세요.")
    else:
        n_ibi = sum(len(b['ibi_ms']) for b in result['blocks'].values())
        st.caption(f"휴식·1-back IBI {n_ibi}개 + 1-back 결과를 트랜잭션 1개로 기록합니다 "
                   f"({size} / {MAX_TX_DATA_BYTES}바이트). 공개 블록체인 기록은 삭제할 수 없습니다.")
        if st.button("🔗 블록체인 저장", type="primary", key=f"save_{result['task_id']}"):
            try:
                with st.spinner("서명 및 전송 중..."):
                    save_to_blockchain(store, result)
                state['saved'] = {'task_id': result['task_id'], 'text': '블록체인 저장 요청 완료 (트랜잭션 1개)'}
                st.rerun(scope="fragment")
            except NodeError as exc:
                st.error(f"저장 실패: {exc}")
    with st.expander("이 측정 데이터 내려받기"):
        c1, c2 = st.columns(2)
        c1.download_button("IBI CSV", csv_ibi([result]), f"{pid}_{result['task_id'][:8]}_ibi.csv", "text/csv",
                           key=f"dl_ibi_{result['task_id']}")
        c2.download_button("1-back CSV", csv_trials([result]), f"{pid}_{result['task_id'][:8]}_1back.csv", "text/csv",
                           key=f"dl_tr_{result['task_id']}")


def experimenter_login() -> bool:
    if st.session_state.get('experimenter') is True:
        return True
    password = config('MONITOR_PASSWORD')
    if not password:
        st.error("secrets 또는 환경변수에 MONITOR_PASSWORD를 설정하세요.")
        return False
    entered = st.text_input("실험자 비밀번호", type="password")
    if entered:
        if secrets.compare_digest(entered, password):
            st.session_state['experimenter'] = True
            st.rerun()
        else:
            time.sleep(1)
            st.error("비밀번호가 틀렸습니다.")
    return False


def monitor_page():
    st.title("📈 PPG 실험자 모니터")
    if not experimenter_login():
        return
    store = get_store()
    sessions = store.session_list()
    top = st.columns([3, 1])
    if top[1].button("🔄 새로고침"):
        st.rerun()
    if sessions:
        labels = {s['pid']: f"{s['pid']} {'● 연결됨' if s['connected'] else '○ 끊김'}" for s in sessions}
        pid = top[0].selectbox("참가자", list(labels), format_func=labels.get)
        live_panel(pid)
    else:
        top[0].info("아직 측정 중인 폰이 없습니다.")

    st.subheader("완료된 측정")
    results = store.all_results()
    if not results:
        st.caption("완료된 측정이 없습니다.")
        return
    import pandas as pd
    rows = [{'참가자': r['participant_id'], '측정': r['task_id'][:8], '조건': COND_LABEL[b['condition']],
             '순서': b['order'], '기록 시작(UTC)': b['rec_start_utc'][11:19], 'HR': b['hr_bpm'],
             'IBI': b['mean_ibi_ms'], 'SDNN': b['sdnn_ms'], 'RMSSD': b['rmssd_ms'], '박동': b['beats'],
             '커버리지': b['coverage'],
             '1-back 정확도': b['nback']['accuracy'] if 'nback' in b else None,
             "d'": b['nback']['d_prime'] if 'nback' in b else None,
             'RT': b['nback']['mean_rt_hit_ms'] if 'nback' in b else None,
             '저장 tx': r['tx_id'][:12], '경고': b['warning']} for r, b in _blocks(results)]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("블록 요약 CSV", csv_summary(results), "ppg_rest_1back_summary.csv", "text/csv")
    c2.download_button("IBI CSV", csv_ibi(results), "ppg_rest_1back_ibi.csv", "text/csv")
    c3.download_button("1-back CSV", csv_trials(results), "ppg_rest_1back_trials.csv", "text/csv")
    if sessions:
        c4.download_button("선택 참가자 원시 신호", store.csv_raw(pid), f"{pid}_raw.csv", "text/csv")


@st.fragment(run_every=1.0)
def live_panel(pid: str):
    import altair as alt
    import pandas as pd
    live = get_store().live(pid)
    if live is None:
        st.info("데이터가 없습니다.")
        return
    res = live['res']
    m = st.columns(4)
    m[0].metric("폰 연결", "연결됨" if live['connected'] else
                ("데이터 없음" if live['last_seen_s'] is None else f"끊김 ({live['last_seen_s']}s)"))
    m[1].metric("심박 (15초)", f"{live['hr15']['hr_bpm']} bpm" if res and live['hr15']['hr_bpm'] else "-")
    m[2].metric("신호 품질", f"{res['quality']:.2f} ({quality_label(res['quality'])})" if res else "-")
    m[3].metric("카메라 FPS", live['fps'] or "-")
    if not res:
        return
    p = make_preview(live)
    wave = pd.DataFrame({'t': p['wave']['t'], 'y': p['wave']['y']})
    peaks = pd.DataFrame({'t': p['wave']['pt'], 'y': p['wave']['py']})
    x = alt.X('t:Q', title='초 (현재 = 0)', scale=alt.Scale(domain=[-10, 0]))
    chart = (alt.Chart(wave).mark_line(color='#3b82f6').encode(x=x, y=alt.Y('y:Q', title=None, axis=None))
             + alt.Chart(peaks).mark_circle(color='#ef4444', size=60).encode(x=x, y='y:Q')).properties(height=180)
    st.altair_chart(chart)


def main():
    is_monitor = st.query_params.get('page') == 'monitor'
    st.set_page_config(page_title="PPG 모니터" if is_monitor else "PPG 측정",
                       layout="wide" if is_monitor else "centered")
    if is_monitor:
        monitor_page()
        return
    views = [f"{MEASURE_ICON} {MEASURE_NAME}", f"{CHAIN_ICON} {CHAIN_NAME}"]
    view = st.radio("화면", views, horizontal=True, label_visibility="collapsed",
                    help="측정 중에 [블록체인 보기]로 바꾸면 진행 중인 측정이 중단됩니다.")
    if view == views[1]:
        chain_page()
    else:
        phone_page()


if __name__ == '__main__':
    main()

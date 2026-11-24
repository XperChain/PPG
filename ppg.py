"""PPG + 시각 1-back 측정 Streamlit 앱 (한 화면에서 준비 → 측정 → 결과 → 블록체인 저장).

    streamlit run ppg_realtime.py

  참가자 폰:   https://<앱 주소>/                 (Streamlit Cloud는 HTTPS 자동 제공)
  실험자 PC:   https://<앱 주소>/?page=monitor    (선택, MONITOR_PASSWORD 필요)

흐름 (폰 한 화면)
  1) [측정 준비]  카메라+플래시 ON → PPG 파형과 검출된 박동, IBI 그래프를 1초마다 갱신해 신호 확인
  2) [측정 시작]  시각 1-back (숫자 20개)
                  숫자 1~5   : 적응 구간 (기록 안 함)
                  숫자 6~15  : IBI 기록 구간
                  숫자 16~20 : 마무리 구간 (기록 안 함)
                  → 기록 구간의 IBI와 1-back 결과를 계산해 화면에 표시
  3) [블록체인 저장]  IBI 데이터 + 1-back 결과를 XperChain LogBook에 기록
                  (노드의 526바이트 제한 때문에 IBI가 많으면 여러 트랜잭션으로 나눠 저장)

블록체인 기록 형식 (schema = ppg_1back_v1, 키를 짧게 써서 용량 절약)
  mid  측정 ID            pid  참가자 코드       p/n  조각 번호/전체 조각 수
  t    기록 시작 시각(UTC) hr/sdnn/rmssd  기록 구간 요약 (첫 조각에만)
  nb   1-back [적중, 놓침, 오경보, 정기각, 적중 평균 RT(ms), 숫자 간격(ms)]  (첫 조각에만)
  b0   기록 시작부터 첫 박동까지(ms)  (첫 조각에만)
  ibi  IBI(ms) 목록. 음수는 이상치로 제외된 박동 (예: -1520)

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
from datetime import datetime, timezone
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
SCHEMA = 'ppg_1back_v1'
TITLE = 'PPG 1-back'
MAX_TX_DATA_BYTES = 526

# 1-back 구성 (숫자 개수)
N_PRE, N_REC, N_POST = 5, 10, 5
N_TOTAL = N_PRE + N_REC + N_POST
STIM_MS = 500                      # 숫자 제시 시간
DEFAULT_TRIAL_MS = 2500            # 숫자 간격 (기록 구간 = 10 × 2.5초 = 25초)

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

    # ---- 1-back 측정 결과 ----
    def compute_result(self, pid: str, run_id: str, task: dict, waited_s: float) -> dict | None:
        """기록 구간(폰 시계)의 IBI + 1-back 성적. 데이터가 아직 다 안 왔으면 None."""
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not RUN_ID_RE.fullmatch(task_id):
            raise ValueError('측정 ID 오류')
        trial_ms = task.get('trial_ms')
        if not _num(trial_ms) or not 1000 <= trial_ms <= 6000:
            raise ValueError('숫자 간격 값 오류')
        trial_ms = int(trial_ms)
        times = [task.get(k) for k in ('task_start_tp', 'rec_start_tp', 'rec_end_tp', 'task_end_tp')]
        if not all(_num(v) for v in times):
            raise ValueError('과제 시각 값 오류')
        t_start, rec_start, rec_end, t_end = map(float, times)
        expected = (N_PRE * trial_ms, N_REC * trial_ms, N_POST * trial_ms)
        actual = (rec_start - t_start, rec_end - rec_start, t_end - rec_end)
        if any(abs(a - e) > 5 for a, e in zip(actual, expected)):
            raise ValueError('과제 구간 시각이 설정과 맞지 않습니다.')
        raw_trials = task.get('trials')
        if not isinstance(raw_trials, list) or len(raw_trials) != N_TOTAL:
            raise ValueError('1-back 시행 데이터 오류')
        trials = []
        for t in raw_trials:
            if not isinstance(t, dict) or str(t.get('s', '')) not in '123456789' or len(str(t.get('s', ''))) != 1:
                raise ValueError('1-back 자극 값 오류')
            responded = t.get('r') is True
            rt = t.get('rt')
            rt = round(float(rt), 1) if responded and _num(rt) and 0 <= rt <= 6000 else None
            trials.append({'s': str(t['s']), 'r': responded, 'rt': rt})

        with self.lock:
            sess = self.sessions.get(pid)
            if sess is None or sess.run_id != run_id:
                raise ValueError('측정 세션이 바뀌었습니다. 다시 측정하세요.')
            if any(r.get('task_id') == task_id for r in sess.results):
                return next(r for r in sess.results if r.get('task_id') == task_id)
            i0 = sess.run_start_idx
            tp = np.asarray(sess.tp[i0:]); r = np.asarray(sess.r[i0:]); g = np.asarray(sess.g[i0:])
        if (len(tp) == 0 or tp[-1] < rec_end + 2000) and waited_s < 20:
            return None                                  # 기록 구간 끝 이후 2초까지 도착을 기다림
        mask = (tp >= rec_start - 20_000) & (tp <= t_end + 2000)   # 필터 안정화를 위해 앞부분 포함
        res = analyze(tp[mask], r[mask], g[mask]) if mask.sum() > 1 else None

        result = {
            'task_id': task_id, 'participant_id': pid, 'trial_ms': trial_ms,
            'rec_start_utc': utc_iso(rec_start + sess.offset_ms), 'rec_duration_s': round((rec_end - rec_start) / 1000, 1),
            'nback': score_nback(trials, 1), 'trials': trials,
            'ibi_ms': [], 'valid': [], 'ibi_t_s': [], 'first_beat_ms': None,
            'hr_bpm': None, 'mean_ibi_ms': None, 'sdnn_ms': None, 'rmssd_ms': None,
            'beats': 0, 'excluded': 0, 'coverage': 0.0, 'quality': None, 'warning': '', 'tx_ids': [],
        }
        if res is None:
            result['warning'] = '신호가 부족해 IBI를 계산하지 못했습니다. 손가락 위치를 확인하고 다시 측정하세요.'
        else:
            pk = res['peak_ms']
            sel = (pk[:-1] >= rec_start) & (pk[1:] <= rec_end)
            ibi, valid = res['ibi_ms'][sel], res['valid'][sel]
            stats = hrv_stats(ibi, valid)
            coverage = float(ibi[valid].sum() / (rec_end - rec_start)) if len(ibi) else 0.0
            result.update({
                'ibi_ms': [round(float(v), 1) for v in ibi], 'valid': [bool(v) for v in valid],
                'ibi_t_s': [round(float((e - rec_start) / 1000), 2) for e in pk[1:][sel]],
                'first_beat_ms': round(float(pk[:-1][sel][0] - rec_start), 1) if sel.any() else None,
                'hr_bpm': stats['hr_bpm'], 'mean_ibi_ms': stats['ibi_ms'], 'sdnn_ms': stats['sdnn_ms'],
                'rmssd_ms': stats['rmssd_ms'], 'beats': stats['beats'], 'excluded': int((~valid).sum()),
                'coverage': round(coverage, 3), 'quality': round(res['quality'], 3),
            })
            if coverage < 0.8:
                result['warning'] = f'기록 구간의 {round((1 - coverage) * 100)}%에서 박동을 신뢰성 있게 검출하지 못했습니다.'
        with self.lock:
            sess.results.append(result)
        return result

    def get_result(self, pid: str, task_id: str) -> dict | None:
        with self.lock:
            sess = self.sessions.get(pid)
            return next((r for r in sess.results if r['task_id'] == task_id), None) if sess else None

    def set_tx(self, pid: str, task_id: str, tx_ids: list[str]):
        with self.lock:
            for r in self.sessions[pid].results:
                if r['task_id'] == task_id:
                    r['tx_ids'] = list(tx_ids)

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


def csv_summary(results: list[dict]) -> bytes:
    cols = ['participant_id', 'task_id', 'rec_start_utc', 'rec_duration_s', 'trial_ms', 'hr_bpm', 'mean_ibi_ms',
            'sdnn_ms', 'rmssd_ms', 'beats', 'excluded', 'coverage', 'quality']
    nb = ['hit', 'miss', 'false_alarm', 'correct_rejection', 'accuracy', 'd_prime', 'mean_rt_hit_ms']
    return csv_bytes(cols + ['nback_' + k for k in nb] + ['tx_ids', 'warning'],
                     ([r[c] for c in cols] + [r['nback'][k] for k in nb] + [' '.join(r['tx_ids']), r['warning']]
                      for r in results))


def csv_ibi(results: list[dict]) -> bytes:
    return csv_bytes(['participant_id', 'task_id', 'beat', 't_from_rec_start_s', 'ibi_ms', 'valid'],
                     ([r['participant_id'], r['task_id'], i + 1, t, v, ok]
                      for r in results for i, (t, v, ok) in enumerate(zip(r['ibi_t_s'], r['ibi_ms'], r['valid']))))


def csv_trials(results: list[dict]) -> bytes:
    def phase(i):
        return 'pre' if i < N_PRE else 'rec' if i < N_PRE + N_REC else 'post'
    return csv_bytes(['participant_id', 'task_id', 'trial', 'phase', 'stimulus', 'is_target', 'responded', 'rt_ms'],
                     ([r['participant_id'], r['task_id'], i + 1, phase(i), t['s'], t.get('target'), t['r'], t['rt']]
                      for r in results for i, t in enumerate(r['trials'])))


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


def build_records(result: dict) -> list[dict]:
    """IBI가 526바이트에 다 안 들어가면 여러 조각으로 나눈다. 첫 조각에 요약과 1-back 결과."""
    nb = result['nback']
    rt = nb['mean_rt_hit_ms']
    base = {'schema': SCHEMA, 'mid': result['task_id'], 'pid': result['participant_id']}
    head = {
        't': result['rec_start_utc'][:19] + 'Z', 'hr': result['hr_bpm'], 'sdnn': result['sdnn_ms'],
        'rmssd': result['rmssd_ms'],
        'nb': [nb['hit'], nb['miss'], nb['false_alarm'], nb['correct_rejection'],
               int(round(rt)) if rt is not None else None, result['trial_ms']],
        'b0': int(round(result['first_beat_ms'])) if result['first_beat_ms'] is not None else None,
    }
    ibis = [int(round(v)) if ok else -int(round(v)) for v, ok in zip(result['ibi_ms'], result['valid'])]

    def fits(rec):
        return tx_data_size(build_tx_data(rec)) <= MAX_TX_DATA_BYTES

    parts, i = [], 0
    while True:
        rec = dict(base, p=len(parts) + 1, n=99, ibi=[], **(head if not parts else {}))
        if not fits(rec):
            raise NodeError('기록 머리 부분이 526바이트를 넘습니다.')
        while i < len(ibis):
            rec['ibi'].append(ibis[i])
            if not fits(rec):
                rec['ibi'].pop()
                break
            i += 1
        if not rec['ibi'] and parts:
            raise NodeError('IBI 조각을 만들 수 없습니다.')
        parts.append(rec)
        if i >= len(ibis):
            break
    for rec in parts:
        rec['n'] = len(parts)
    return parts


def used_parts() -> set[tuple[str, int]]:
    """체인과 mempool에 이미 있는 (측정 ID, 조각 번호)."""
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
    used = set()
    for text in texts:
        try:
            c = json.loads(text)
            if isinstance(c, dict) and c.get('schema') == SCHEMA and isinstance(c.get('mid'), str) \
                    and isinstance(c.get('p'), int):
                used.add((c['mid'], c['p']))
        except (TypeError, ValueError):
            continue
    return used


def save_to_blockchain(store: Store, result: dict) -> list[str]:
    if result.get('hr_bpm') is None:
        raise NodeError('IBI 결과가 없는 측정은 저장할 수 없습니다.')
    allowed = config('ALLOWED_PARTICIPANT_CODES')
    if allowed and result['participant_id'] not in {v.strip().upper() for v in allowed.split(',')}:
        raise NodeError('이 참가자 코드는 저장이 허용되지 않았습니다.')
    parts = build_records(result)
    key = service_signing_key()
    wallet, public = wallet_address(key)
    tx_ids = list(result.get('tx_ids') or [])
    if len(tx_ids) >= len(parts):
        raise NodeError('이 측정은 이미 블록체인에 저장 요청되었습니다.')
    with store.submit_lock:
        used = used_parts()
        # 이번 앱에서 이미 보낸 조각(아직 mempool에 안 보일 수 있음)도 건너뜀
        used |= {(result['task_id'], p) for p in range(1, len(tx_ids) + 1)}
        todo = [rec for rec in parts if (rec['mid'], rec['p']) not in used]
        if not todo:
            raise NodeError('이 측정은 이미 블록체인에 저장(또는 대기)되어 있습니다.')
        nonce = int(node_request('GET', f'/account/{wallet}/nonce')['next_nonce'])
        for k, rec in enumerate(todo):               # 조각들은 nonce를 1씩 늘려 연속 전송
            tx_data = build_tx_data(rec)
            fee = node_request('POST', '/fee/estimate', json={'tx_type': 'CONTRACT_CALL', 'data': tx_data})
            tx = {'sender': wallet, 'recipient': CONTRACT_ADDR, 'amount_xbit': 0,
                  'fee_xbit': int(fee['required_fee_xbit']), 'nonce': nonce + k,
                  'timestamp': time.time(), 'tx_type': 'CONTRACT_CALL', 'data': tx_data}
            sig = key.sign(canonical_tx_bytes(tx))
            res = node_request('POST', '/contract/call', json={
                'contract_addr': CONTRACT_ADDR, 'method': 'upload_text', 'args': tx_data['args'],
                'sender': wallet, 'sender_pub_b64': base64.b64encode(public).decode('ascii'),
                'sig_b64': base64.b64encode(sig).decode('ascii'), 'value_xbit': 0,
                'fee_xbit': tx['fee_xbit'], 'nonce': tx['nonce'], 'timestamp': tx['timestamp']})
            tx_id = str(res.get('tx_id', ''))
            if not TX_ID_RE.fullmatch(tx_id):
                raise NodeError(f'{rec["p"]}번째 조각: 노드가 올바른 트랜잭션 ID를 반환하지 않았습니다.')
            tx_ids.append(tx_id)
            store.set_tx(result['participant_id'], result['task_id'], tx_ids)
    return tx_ids


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
  #progress { color:#8a96a3; text-align:center; font-variant-numeric:tabular-nums; }
  #stimBox { height:220px; display:flex; align-items:center; justify-content:center; }
  #stim { font-size:150px; font-weight:700; line-height:1; }
  #fix { font-size:40px; color:#6b7280; }
  #taskMsg { text-align:center; font-size:18px; line-height:1.6; min-height:28px; }
  #matchBtn { height:110px; font-size:30px; font-weight:700; margin-top:16px; }
  #matchBtn.pressed { background:#94a3b8; }
  table { width:100%; border-collapse:collapse; font-size:15px; margin-top:6px; }
  td { padding:6px 4px; border-bottom:1px solid #2a313a; }
  td:last-child { text-align:right; font-variant-numeric:tabular-nums; }
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
    <button id="startBtn">측정 시작</button>
    <button id="cancelBtn" class="secondary">취소</button>
  </div>

  <div id="v-task" class="hidden">
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
function drawIbi(c, t, ms, valid, xmin, xmax) {
  const [g, w, h] = fit(c);
  const pad = 34, lo = 400, hi = 1400;
  const X = v => pad + (w - pad - 6) * (v - xmin) / (xmax - xmin);
  const Y = v => h - 8 - (h - 16) * (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo);
  g.strokeStyle = "#2a313a"; g.fillStyle = "#6b7280"; g.font = "10px system-ui";
  for (let v = 400; v <= 1400; v += 250) { g.beginPath(); g.moveTo(pad, Y(v)); g.lineTo(w, Y(v)); g.stroke(); g.fillText(v, 2, Y(v) + 3); }
  if (!t || !t.length) return;
  g.strokeStyle = "#22c55e"; g.lineWidth = 1.5; g.beginPath(); let on = false;
  t.forEach((x, i) => { if (!valid[i]) { on = false; return; } on ? g.lineTo(X(x), Y(ms[i])) : g.moveTo(X(x), Y(ms[i])); on = true; });
  g.stroke();
  t.forEach((x, i) => { g.fillStyle = valid[i] ? "#22c55e" : "#9ca3af"; g.beginPath(); g.arc(X(x), Y(ms[i]), 3, 0, Math.PI * 2); g.fill(); });
}
function drawPreview(p) {
  drawWave($("cWave"), p && p.wave);
  drawIbi($("cIbi"), p && p.ibi.t, p && p.ibi.ms, p && p.ibi.valid, -60, 0);
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

function startTask() {
  trialMs = Math.round(args.trial_ms || 2500); nPre = args.n_pre || 5; nRec = args.n_rec || 10;
  nPost = args.n_post || 5; stimMs = args.stim_ms || 500;
  const total = nPre + nRec + nPost;
  digits = makeSequence(total, 1, 0.3); trials = []; cur = null;
  taskId = randomId(); gotResult = false; taskPayload = null;
  showView("task");
  $("stim").textContent = ""; $("fix").classList.add("hidden"); $("progress").textContent = "";
  $("matchBtn").disabled = true;
  let k = 3;
  const tickDown = () => {
    $("taskMsg").innerHTML = `숫자가 <b>바로 앞 숫자와 같으면</b> [일치]를 누르세요.<br><span style="font-size:42px">${k}</span>`;
    if (k-- > 0) later(tickDown, 1000);
    else {
      $("taskMsg").textContent = ""; $("matchBtn").disabled = false; $("fix").classList.remove("hidden");
      t0 = performance.now(); runTrial(0, total);
    }
  };
  tickDown();
}
function runTrial(i, total) {
  if (i >= total) return finishTask(total);
  const t = { s: digits[i], r: false, rt: null, on: null };
  trials.push(t); cur = t;
  $("progress").textContent = `${i + 1} / ${total}`;
  $("fix").classList.add("hidden"); $("stim").textContent = digits[i];
  t.on = performance.now();
  later(() => { $("stim").textContent = ""; $("fix").classList.remove("hidden"); }, stimMs);
  later(() => runTrial(i + 1, total), t0 + (i + 1) * trialMs - performance.now());   // 누적 지연 없음
}
$("matchBtn").addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (mode !== "task" || !cur || cur.r || cur.on === null) return;
  cur.r = true; cur.rt = Math.max(0, performance.now() - cur.on);
  const b = $("matchBtn"); b.classList.add("pressed"); later(() => b.classList.remove("pressed"), 150);
});
function finishTask(total) {
  cur = null;
  timers.forEach(clearTimeout); timers = [];
  stopCamera();
  taskPayload = {
    task_id: taskId, trial_ms: trialMs,
    task_start_tp: epoch(t0), rec_start_tp: epoch(t0 + nPre * trialMs),
    rec_end_tp: epoch(t0 + (nPre + nRec) * trialMs), task_end_tp: epoch(t0 + total * trialMs),
    trials: trials.map(t => ({ s: t.s, r: t.r, rt: t.rt === null ? null : +t.rt.toFixed(1) })),
  };
  $("doneTitle").textContent = "결과 계산 중..."; $("summary").replaceChildren();
  $("warn").textContent = ""; $("saveHint").textContent = ""; $("resLabel").textContent = "";
  $("cRes").classList.add("hidden");
  showView("done");
  push();
}
function renderResult(r) {
  $("doneTitle").textContent = "측정 완료";
  const nb = r.nback, targets = nb.hit + nb.miss;
  const rows = [
    ["기록 구간", `${r.rec_duration_s}초 (숫자 ${nRec}개)`],
    ["검출된 IBI", `${r.beats}개` + (r.excluded ? ` (제외 ${r.excluded}개)` : "")],
    ["평균 심박", r.hr_bpm == null ? "-" : `${r.hr_bpm} bpm`],
    ["평균 IBI", r.mean_ibi_ms == null ? "-" : `${r.mean_ibi_ms} ms`],
    ["SDNN / RMSSD", `${r.sdnn_ms ?? "-"} / ${r.rmssd_ms ?? "-"} ms`],
    ["1-back 정확도", `${Math.round(nb.accuracy * 100)}%`],
    ["적중 / 오경보", `${nb.hit}/${targets} · ${nb.false_alarm}회`],
    ["평균 반응시간", nb.mean_rt_hit_ms == null ? "-" : `${Math.round(nb.mean_rt_hit_ms)} ms`],
  ];
  const table = document.createElement("table");
  for (const [k, v] of rows) { const tr = table.insertRow(); tr.insertCell().textContent = k; tr.insertCell().textContent = v; }
  $("summary").replaceChildren(table);
  if (r.ibi_ms.length) {
    $("resLabel").textContent = "기록 구간 IBI (회색 = 제외된 박동)";
    $("cRes").classList.remove("hidden");
    drawIbi($("cRes"), r.ibi_t_s, r.ibi_ms, r.valid, 0, r.rec_duration_s);
  }
  $("warn").textContent = r.warning || "";
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
$("startBtn").addEventListener("click", startTask);
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
def phone_page():
    st.markdown("### 📷 PPG + 1-back 측정")
    pid_raw = st.text_input("참가자 코드", placeholder="P01", max_chars=4)
    with st.expander("설정"):
        trial_s = st.number_input("숫자 간격 (초)", 1.0, 6.0, DEFAULT_TRIAL_MS / 1000, 0.5,
                                  help=f"IBI 기록 구간 = 숫자 {N_REC}개 × 간격")
        st.caption(f"전체 {N_TOTAL}개 숫자 · IBI 기록 구간 {N_REC * trial_s:.0f}초 · 총 약 {N_TOTAL * trial_s + 3:.0f}초")
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
        trial_ms=trial_ms, n_pre=N_PRE, n_rec=N_REC, n_post=N_POST, stim_ms=STIM_MS,
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
                        state['result_public'] = {k: v for k, v in result.items() if k != 'trials'}
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
    if result['tx_ids']:
        st.success(f"블록체인 저장 요청 완료 ({len(result['tx_ids'])}개 트랜잭션)")
        for t in result['tx_ids']:
            st.code(t, language=None)
    elif result['hr_bpm'] is None:
        st.warning("IBI 결과가 없어 저장할 수 없습니다. 다시 측정하세요.")
    else:
        try:
            n_parts = len(build_records(result))
        except NodeError:
            n_parts = None
        st.caption(f"IBI {len(result['ibi_ms'])}개 + 1-back 결과를 기록합니다"
                   + (f" (트랜잭션 {n_parts}개)" if n_parts else "") + ". 공개 블록체인 기록은 삭제할 수 없습니다.")
        if st.button("🔗 블록체인 저장", type="primary", key=f"save_{result['task_id']}"):
            try:
                with st.spinner("서명 및 전송 중..."):
                    tx_ids = save_to_blockchain(store, result)
                state['saved'] = {'task_id': result['task_id'], 'text': f'블록체인 저장 요청 완료 (트랜잭션 {len(tx_ids)}개)'}
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
    rows = [{'참가자': r['participant_id'], '기록 시작(UTC)': r['rec_start_utc'][11:19], 'HR': r['hr_bpm'],
             'IBI': r['mean_ibi_ms'], 'SDNN': r['sdnn_ms'], 'RMSSD': r['rmssd_ms'], '박동': r['beats'],
             '커버리지': r['coverage'], '1-back 정확도': r['nback']['accuracy'], "d'": r['nback']['d_prime'],
             'RT': r['nback']['mean_rt_hit_ms'], '저장 tx': len(r['tx_ids']), '경고': r['warning']} for r in results]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("요약 CSV", csv_summary(results), "ppg_1back_summary.csv", "text/csv")
    c2.download_button("IBI CSV", csv_ibi(results), "ppg_1back_ibi.csv", "text/csv")
    c3.download_button("1-back CSV", csv_trials(results), "ppg_1back_trials.csv", "text/csv")
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
    else:
        phone_page()


if __name__ == '__main__':
    main()

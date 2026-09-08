import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import socket
import struct
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
import pyautogui
import cv2
import numpy as np
from pynput import keyboard

EV_URL = "https://www.ev.or.kr/nportal/main.do"
TEMPLATE = Path(__file__).with_name("support_apply_template.png")
KST = timezone(timedelta(hours=9))

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


def parse_http_date(value):
    return datetime.strptime(value, "%a, %d %b %Y %H:%M:%S GMT").replace(tzinfo=timezone.utc).timestamp()


def http_date_offset(url, samples=9, timeout=4):
    """Estimate server clock offset using HTTP Date and request midpoint."""
    session = requests.Session()
    vals = []
    for _ in range(samples):
        t0 = time.time()
        m0 = time.perf_counter()
        try:
            r = session.get(url, timeout=timeout, headers={
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
                "User-Agent": "Mozilla/5.0"
            })
            m1 = time.perf_counter()
            t1 = time.time()
            d = r.headers.get("Date")
            if not d:
                continue
            server_ts = parse_http_date(d)
            midpoint = (t0 + t1) / 2
            vals.append((server_ts - midpoint, m1 - m0))
        except Exception:
            continue
    if not vals:
        return None
    vals.sort(key=lambda x: x[1])
    best = vals[:min(5, len(vals))]
    return sum(v[0] for v in best) / len(best), min(v[1] for v in best)


def ntp_offset(host, timeout=2.0):
    NTP_EPOCH = 2208988800
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        t1 = time.time()
        packet = bytearray(48)
        packet[0] = 0x1B
        sock.sendto(packet, (host, 123))
        data, _ = sock.recvfrom(512)
        t4 = time.time()
        if len(data) < 48:
            raise RuntimeError("NTP 응답 오류")
        sec2, frac2 = struct.unpack("!II", data[32:40])
        sec3, frac3 = struct.unpack("!II", data[40:48])
        t2 = sec2 - NTP_EPOCH + frac2 / 2**32
        t3 = sec3 - NTP_EPOCH + frac3 / 2**32
        return ((t2 - t1) + (t3 - t4)) / 2, t4 - t1
    finally:
        sock.close()


def get_server_offset():
    # EV first. If the EV web server/proxy supplies Date, use it.
    ev = http_date_offset(EV_URL)
    if ev:
        return ev[0], ev[1], "EV 웹서버 Date"

    # HTTPS Date fallbacks are useful when UDP/123 is blocked.
    https_sources = [
        ("https://www.google.com/generate_204", "인터넷 표준시간(HTTPS Google)"),
        ("https://www.naver.com/", "인터넷 표준시간(HTTPS Naver)"),
        ("https://www.microsoft.com/", "인터넷 표준시간(HTTPS Microsoft)"),
    ]
    results = []
    for url, label in https_sources:
        try:
            off = http_date_offset(url, samples=5, timeout=3)
            if off:
                results.append((off[0], off[1], label))
        except Exception:
            pass
    if results:
        results.sort(key=lambda x: x[1])
        return results[0]

    # Final fallback: NTP.
    for host in ["time.cloudflare.com", "time.google.com", "pool.ntp.org"]:
        try:
            off, rtt = ntp_offset(host)
            return off, rtt, f"인터넷 표준시간(NTP: {host})"
        except Exception:
            pass

    raise RuntimeError(
        "시간 동기화에 실패했습니다.\n\n"
        "EV 웹서버, HTTPS 표준시간 서버, NTP 서버에 모두 연결하지 못했습니다.\n"
        "인터넷 연결 또는 회사/기관 방화벽을 확인해 주세요."
    )


class ServerClock:
    """Monotonic-clock based server time. Avoids system clock jumps while waiting."""
    def __init__(self, offset, server_epoch=None):
        self.offset = offset
        self.sync_mono = time.perf_counter()
        self.server_epoch = time.time() + offset if server_epoch is None else server_epoch

    def now_epoch(self):
        return self.server_epoch + (time.perf_counter() - self.sync_mono)

    def target_mono(self, target_epoch):
        return self.sync_mono + (target_epoch - self.server_epoch)


def parse_target(text):
    parts = [int(x) for x in text.strip().split(":")]
    if len(parts) != 3:
        raise ValueError
    h, m, s = parts
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError
    return h, m, s


def find_button(confidence=0.72):
    """Fast multi-scale matching of the tightly cropped 지원신청 button."""
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"템플릿 파일이 없습니다: {TEMPLATE}")
    tpl = cv2.imread(str(TEMPLATE), cv2.IMREAD_COLOR)
    if tpl is None:
        raise RuntimeError("지원신청 템플릿을 읽지 못했습니다.")

    screen_rgb = np.array(pyautogui.screenshot())
    screen = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR)
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    edge_screen = cv2.Canny(gray_screen, 60, 160)

    # Most machines will be ~1.0 scale. Try a dense set around 1.0, then wider scales.
    scales = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.35, 1.50, 1.75]
    best = None
    for scale in scales:
        w = max(20, int(tpl.shape[1] * scale))
        h = max(12, int(tpl.shape[0] * scale))
        if w >= screen.shape[1] or h >= screen.shape[0]:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        rt = cv2.resize(tpl, (w, h), interpolation=interp)
        result = cv2.matchTemplate(screen, rt, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(result)
        candidate = (float(val), loc, w, h, "color")
        if best is None or candidate[0] > best[0]:
            best = candidate

        # Edge check only at a few likely scales to keep the search fast.
        if scale in (0.90, 1.00, 1.10, 1.35):
            edge_tpl = cv2.Canny(cv2.cvtColor(rt, cv2.COLOR_BGR2GRAY), 60, 160)
            er = cv2.matchTemplate(edge_screen, edge_tpl, cv2.TM_CCOEFF_NORMED)
            _, eval_, _, eloc = cv2.minMaxLoc(er)
            combined = 0.60 * float(val) + 0.40 * float(eval_)
            c2 = (combined, eloc, w, h, "edge")
            if best is None or c2[0] > best[0]:
                best = c2

    if best is None or best[0] < confidence:
        return None
    score, (x, y), w, h, method = best
    return score, x, y, w, h, method


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("EV 지원신청 정각 클릭 매크로")
        self.root.geometry("540x560")
        self.root.resizable(False, False)
        self.offset = None
        self.clock = None
        self.time_source = ""
        self.running = False
        self.worker = None
        self.mode = tk.StringVar(value="09:00:00")
        self.custom = tk.StringVar(value="09:00:00")
        self.status = tk.StringVar(value="대기 중 (F2 시작 / F4 중지)")
        self.server_clock = tk.StringVar(value="시간 동기화 전")
        self.offset_text = tk.StringVar(value="보정값 확인 전")
        self.conf = tk.DoubleVar(value=0.72)

        pad = {"padx": 14, "pady": 7}
        ttk.Label(root, text="EV 지원신청 정각 클릭 매크로", font=("맑은 고딕", 16, "bold")).pack(pady=(16, 8))
        box = ttk.LabelFrame(root, text="목표 시간")
        box.pack(fill="x", **pad)
        ttk.Radiobutton(box, text="09:00:00", variable=self.mode, value="09:00:00").grid(row=0, column=0, sticky="w", padx=10, pady=7)
        ttk.Radiobutton(box, text="10:00:00", variable=self.mode, value="10:00:00").grid(row=0, column=1, sticky="w", padx=10, pady=7)
        ttk.Radiobutton(box, text="직접 입력", variable=self.mode, value="custom").grid(row=1, column=0, sticky="w", padx=10, pady=7)
        ttk.Entry(box, textvariable=self.custom, width=12).grid(row=1, column=1, sticky="w", padx=10, pady=7)
        ttk.Label(box, text="HH:MM:SS").grid(row=1, column=2, sticky="w")

        box2 = ttk.LabelFrame(root, text="동작")
        box2.pack(fill="x", **pad)
        ttk.Label(box2, text="① 시간 동기화 → ② 정밀 카운트다운 → ③ 00초에 즉시 현재 커서 클릭\n④ 즉시 '지원신청' 이미지 탐색 → 발견 즉시 클릭").pack(anchor="w", padx=10, pady=8)
        match_row = ttk.Frame(box2)
        match_row.pack(fill="x", padx=10, pady=6)
        ttk.Label(match_row, text="이미지 매칭 기준").pack(side="left")
        ttk.Scale(match_row, from_=0.60, to=0.90, variable=self.conf, orient="horizontal", length=180).pack(side="left", padx=10)
        ttk.Label(match_row, text="기본 0.72").pack(side="left")

        info = ttk.LabelFrame(root, text="상태")
        info.pack(fill="x", **pad)
        ttk.Label(info, textvariable=self.server_clock).pack(anchor="w", padx=10, pady=4)
        ttk.Label(info, textvariable=self.offset_text).pack(anchor="w", padx=10, pady=4)
        ttk.Label(info, textvariable=self.status, font=("맑은 고딕", 11, "bold")).pack(anchor="w", padx=10, pady=8)

        btns = ttk.Frame(root)
        btns.pack(pady=14)
        ttk.Button(btns, text="시간 동기화", command=self.sync).pack(side="left", padx=5)
        ttk.Button(btns, text="매크로 시작", command=self.start).pack(side="left", padx=5)
        ttk.Button(btns, text="중지", command=self.stop).pack(side="left", padx=5)
        ttk.Label(root, text="F2 = 시작 / F4 = 중지 / 긴급 중지 = 마우스를 왼쪽 위 모서리", foreground="red").pack(pady=3)
        ttk.Label(root, text="※ 로그인과 신청 대상 선택은 사용자가 미리 완료해 주세요.", foreground="#555").pack(pady=3)

        self.hotkey_listener = keyboard.Listener(on_press=self._on_global_key)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _on_global_key(self, key):
        try:
            if key == keyboard.Key.f2:
                self.root.after(0, self.start)
            elif key == keyboard.Key.f4:
                self.root.after(0, self.stop)
        except Exception:
            pass

    def close(self):
        self.running = False
        try:
            self.hotkey_listener.stop()
        except Exception:
            pass
        self.root.destroy()

    def log(self, text):
        self.root.after(0, self.status.set, text)

    def apply_sync(self, result):
        off, rtt, source = result
        # Anchor the server clock to perf_counter immediately after measurement.
        server_epoch = time.time() + off
        self.offset = off
        self.clock = ServerClock(off, server_epoch)
        self.time_source = source
        now = datetime.fromtimestamp(self.clock.now_epoch(), KST)
        self.root.after(0, self.server_clock.set, "기준 현재시간: " + now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        self.root.after(0, self.offset_text.set, f"기준: {source} / PC 보정값: {off:+.3f}초 / RTT: {rtt*1000:.0f}ms")

    def sync(self):
        def work():
            try:
                self.log("시간 동기화 중... (EV 우선 → HTTPS → NTP)")
                result = get_server_offset()
                self.apply_sync(result)
                self.log("시간 동기화 완료")
            except Exception as e:
                self.log("동기화 실패")
                self.root.after(0, lambda: messagebox.showerror("시간 동기화 오류", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def start(self):
        if self.running:
            return
        target_text = self.custom.get() if self.mode.get() == "custom" else self.mode.get()
        try:
            parse_target(target_text)
        except Exception:
            messagebox.showerror("시간 오류", "HH:MM:SS 형식으로 입력하세요.")
            return
        x, y = pyautogui.position()
        if not messagebox.askyesno("매크로 시작", f"목표시간: {target_text}\n현재 커서: ({x}, {y})\n\n정각에 현재 커서 위치를 즉시 클릭하고, 지원신청을 찾아 즉시 클릭합니다.\n\n시작할까요?"):
            return
        self.running = True
        self.worker = threading.Thread(target=self.run, args=(target_text,), daemon=True)
        self.worker.start()

    def stop(self):
        if self.running:
            self.running = False
            self.log("중지됨")
        else:
            self.log("대기 중")

    def precise_wait(self, target_mono):
        # Coarse sleep until close to target, then short sleeps, then spin for the last ~2ms.
        while self.running:
            remain = target_mono - time.perf_counter()
            if remain <= 0:
                return
            if remain > 0.25:
                time.sleep(remain - 0.12)
            elif remain > 0.01:
                time.sleep(0.001)
            else:
                # Yield very briefly while retaining responsiveness to F4.
                time.sleep(0)

    def run(self, target_text):
        try:
            if self.clock is None:
                self.log("시간 동기화 중...")
                self.apply_sync(get_server_offset())

            h, m, s = parse_target(target_text)
            now_epoch = self.clock.now_epoch()
            now_dt = datetime.fromtimestamp(now_epoch, KST)
            target_dt = now_dt.replace(hour=h, minute=m, second=s, microsecond=0)
            if target_dt.timestamp() <= now_epoch:
                target_dt += timedelta(days=1)
            target_epoch = target_dt.timestamp()
            target_mono = self.clock.target_mono(target_epoch)

            self.log(f"{target_dt.strftime('%Y-%m-%d %H:%M:%S')}까지 정밀 대기 중...")
            last_display = 0
            while self.running:
                remain = target_mono - time.perf_counter()
                if remain <= 0.5:
                    break
                now = time.perf_counter()
                if now - last_display > 0.5:
                    self.log(f"목표시간까지 {remain:.1f}초")
                    last_display = now
                time.sleep(min(0.2, max(0.02, remain - 0.3)))

            self.precise_wait(target_mono)
            if not self.running:
                return

            # ZERO artificial pause: click immediately at the target.
            self.log("정각! 즉시 현재 마우스 위치 클릭")
            pyautogui.click()

            # Start image search immediately after the first click.
            deadline = time.perf_counter() + 8.0
            found = None
            while self.running and time.perf_counter() < deadline:
                found = find_button(float(self.conf.get()))
                if found:
                    break
                time.sleep(0.02)

            if not found:
                raise RuntimeError("'지원신청' 버튼을 찾지 못했습니다. 화면 확대율/브라우저 상태를 확인해 주세요.")

            score, x, y, w, h, method = found
            cx, cy = x + w // 2, y + h // 2
            self.log(f"'지원신청' 발견 (일치율 {score:.2f}, {method}) → 즉시 클릭")
            pyautogui.click(cx, cy)
            self.log("완료: 지원신청 클릭")
        except Exception as e:
            self.log("오류 발생")
            self.root.after(0, lambda: messagebox.showerror("매크로 오류", str(e)))
        finally:
            self.running = False


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconname("EV 지원신청 매크로")
    except Exception:
        pass
    App(root)
    root.mainloop()

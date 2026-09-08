import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
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

# Stop immediately with the mouse moved to the upper-left corner.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03


def get_server_offset(samples=7):
    """
    Estimate EV server clock - local clock.
    Uses HTTP Date from ev.or.kr and midpoint timing to reduce network latency.
    """
    session = requests.Session()
    values = []
    for _ in range(samples):
        t0 = time.time()
        r = session.get(EV_URL, timeout=5, headers={"Cache-Control": "no-cache"})
        t1 = time.time()
        date_header = r.headers.get("Date")
        if not date_header:
            continue
        # HTTP Date is UTC, second precision.
        server_ts = datetime.strptime(
            date_header, "%a, %d %b %Y %H:%M:%S GMT"
        ).replace(tzinfo=timezone.utc).timestamp()
        midpoint = (t0 + t1) / 2
        values.append((server_ts - midpoint, t1 - t0))

    if not values:
        raise RuntimeError("ev.or.kr 응답의 Date 헤더를 읽지 못했습니다.")

    # Prefer the lowest-latency samples.
    values.sort(key=lambda x: x[1])
    best = values[:max(3, min(5, len(values)))]
    return sum(x[0] for x in best) / len(best), min(x[1] for x in best)


def server_now(offset):
    return time.time() + offset


def parse_target(text):
    h, m, s = [int(x) for x in text.strip().split(":")]
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        raise ValueError
    return h, m, s


def find_button(confidence=0.82):
    """
    Multi-scale template matching for the blue '지원신청' button.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"템플릿 파일이 없습니다: {TEMPLATE}")

    tpl = cv2.imread(str(TEMPLATE), cv2.IMREAD_COLOR)
    screen = np.array(pyautogui.screenshot())
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)

    best = None
    # Covers common Windows/browser scaling differences.
    for scale in np.linspace(0.65, 1.45, 33):
        w = max(10, int(tpl.shape[1] * scale))
        h = max(10, int(tpl.shape[0] * scale))
        if w >= screen.shape[1] or h >= screen.shape[0]:
            continue
        resized = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(screen, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if best is None or max_val > best[0]:
            best = (max_val, max_loc, w, h)

    if not best or best[0] < confidence:
        return None

    score, (x, y), w, h = best
    return score, x, y, w, h


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("EV 지원신청 정각 클릭 매크로")
        self.root.geometry("520x520")
        self.root.resizable(False, False)

        self.offset = None
        self.running = False
        self.worker = None

        self.mode = tk.StringVar(value="09:00:00")
        self.custom = tk.StringVar(value="09:00:00")
        self.status = tk.StringVar(value="대기 중")
        self.server_clock = tk.StringVar(value="서버시간 확인 전")
        self.offset_text = tk.StringVar(value="오차 보정값 확인 전")
        self.conf = tk.DoubleVar(value=0.82)

        pad = {"padx": 14, "pady": 7}

        ttk.Label(root, text="EV 지원신청 정각 클릭 매크로",
                  font=("맑은 고딕", 16, "bold")).pack(pady=(16, 8))

        box = ttk.LabelFrame(root, text="목표 시간")
        box.pack(fill="x", **pad)

        ttk.Radiobutton(box, text="09:00:00", variable=self.mode,
                        value="09:00:00").grid(row=0, column=0, sticky="w", padx=10, pady=7)
        ttk.Radiobutton(box, text="10:00:00", variable=self.mode,
                        value="10:00:00").grid(row=0, column=1, sticky="w", padx=10, pady=7)
        ttk.Radiobutton(box, text="직접 입력", variable=self.mode,
                        value="custom").grid(row=1, column=0, sticky="w", padx=10, pady=7)
        ttk.Entry(box, textvariable=self.custom, width=12).grid(
            row=1, column=1, sticky="w", padx=10, pady=7)
        ttk.Label(box, text="HH:MM:SS").grid(row=1, column=2, sticky="w")

        box2 = ttk.LabelFrame(root, text="동작")
        box2.pack(fill="x", **pad)

        ttk.Label(box2, text="① 시작 후 EV 서버시간 동기화\n"
                             "② 목표시각까지 대기\n"
                             "③ 대기 종료 순간의 현재 마우스 위치 클릭\n"
                             "④ '지원신청' 이미지 자동 탐색 후 클릭").pack(
            anchor="w", padx=10, pady=8)

        ttk.Label(box2, text="이미지 매칭 기준").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        ttk.Scale(box2, from_=0.65, to=0.95, variable=self.conf,
                  orient="horizontal", length=180).grid(row=1, column=1, sticky="w")
        ttk.Label(box2, textvariable=tk.StringVar(value="권장 0.82")).grid(
            row=1, column=2, sticky="w", padx=5)

        info = ttk.LabelFrame(root, text="상태")
        info.pack(fill="x", **pad)
        ttk.Label(info, textvariable=self.server_clock).pack(anchor="w", padx=10, pady=4)
        ttk.Label(info, textvariable=self.offset_text).pack(anchor="w", padx=10, pady=4)
        ttk.Label(info, textvariable=self.status,
                  font=("맑은 고딕", 11, "bold")).pack(anchor="w", padx=10, pady=8)

        btns = ttk.Frame(root)
        btns.pack(pady=14)
        ttk.Button(btns, text="서버시간 동기화", command=self.sync).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="매크로 시작", command=self.start).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="중지", command=self.stop).grid(row=0, column=2, padx=5)

        ttk.Label(root, text="긴급 중지: 마우스를 화면 왼쪽 위 모서리로 이동",
                  foreground="red").pack(pady=3)
        ttk.Label(root, text="※ 로그인과 신청 대상 선택은 사용자가 미리 완료해 주세요.",
                  foreground="#555").pack(pady=3)

    def log(self, text):
        self.root.after(0, self.status.set, text)

    def sync(self):
        def work():
            try:
                self.log("EV 서버시간 동기화 중...")
                off, rtt = get_server_offset()
                self.offset = off
                now = datetime.fromtimestamp(server_now(off), KST)
                self.root.after(0, self.server_clock.set,
                                "EV 기준 현재시간: " + now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                self.root.after(0, self.offset_text.set,
                                f"PC 보정값: {off:+.3f}초 / 최소 RTT: {rtt*1000:.0f}ms")
                self.log("동기화 완료")
            except Exception as e:
                self.log("동기화 실패")
                self.root.after(0, lambda: messagebox.showerror("오류", str(e)))
        threading.Thread(target=work, daemon=True).start()

    def start(self):
        if self.running:
            return
        if self.mode.get() == "custom":
            target_text = self.custom.get()
        else:
            target_text = self.mode.get()
        try:
            parse_target(target_text)
        except Exception:
            messagebox.showerror("시간 오류", "HH:MM:SS 형식으로 입력하세요.")
            return

        # Capture the cursor position now only for display; the actual click uses
        # the cursor position at the target instant as requested.
        x, y = pyautogui.position()
        if not messagebox.askyesno(
            "매크로 시작",
            f"목표시간: {target_text}\n"
            f"현재 커서: ({x}, {y})\n\n"
            "목표시각이 되면 현재 커서 위치를 클릭하고,\n"
            "'지원신청' 버튼을 이미지로 찾아 클릭합니다.\n\n시작할까요?"
        ):
            return

        self.running = True
        self.worker = threading.Thread(
            target=self.run, args=(target_text,), daemon=True)
        self.worker.start()

    def stop(self):
        self.running = False
        self.log("중지됨")

    def run(self, target_text):
        try:
            if self.offset is None:
                self.log("EV 서버시간 동기화 중...")
                self.offset, rtt = get_server_offset()
                now = datetime.fromtimestamp(server_now(self.offset), KST)
                self.root.after(0, self.server_clock.set,
                                "EV 기준 현재시간: " + now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
                self.root.after(0, self.offset_text.set,
                                f"PC 보정값: {self.offset:+.3f}초 / 최소 RTT: {rtt*1000:.0f}ms")

            h, m, s = parse_target(target_text)
            now = datetime.fromtimestamp(server_now(self.offset), KST)
            target = now.replace(hour=h, minute=m, second=s, microsecond=0)
            if target <= now:
                # If today's target has passed, schedule tomorrow.
                target += timedelta(days=1)

            self.log("목표시간까지 대기 중...")

            while self.running:
                remain = target.timestamp() - server_now(self.offset)
                if remain <= 0.15:
                    break
                self.log(f"목표시간까지 {remain:.2f}초")
                time.sleep(min(0.1, max(0.01, remain / 5)))

            if not self.running:
                return

            # Busy wait for the final fraction of a second.
            while self.running and server_now(self.offset) < target.timestamp():
                pass

            if not self.running:
                return

            self.log("정각! 현재 마우스 위치 클릭")
            pyautogui.click()

            # Give the page a moment to react, then search repeatedly.
            time.sleep(0.5)
            deadline = time.time() + 8.0
            found = None
            while self.running and time.time() < deadline:
                found = find_button(float(self.conf.get()))
                if found:
                    break
                time.sleep(0.15)

            if not found:
                raise RuntimeError(
                    "'지원신청' 버튼을 찾지 못했습니다. 브라우저 확대율/화면 상태를 확인하세요."
                )

            score, x, y, w, h = found
            cx, cy = x + w // 2, y + h // 2
            self.log(f"'지원신청' 발견 (일치율 {score:.2f}) → 클릭")
            pyautogui.moveTo(cx, cy, duration=0.05)
            pyautogui.click()
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

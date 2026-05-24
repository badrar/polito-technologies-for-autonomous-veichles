"""
Garmin BLE Heart Rate Monitor
Requires: pip install bleak
Garmin watch must have "Trasmetti FC" (Broadcast Heart Rate) active.
Press 'q' to quit.
"""

import asyncio
import threading
import numpy as np
import cv2
from bleak import BleakScanner, BleakClient

HR_SERVICE_UUID  = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID     = "00002a37-0000-1000-8000-00805f9b34fb"
GARMIN_NAME_HINT = "HRM"

BG_COLOR  = (30, 26, 26)       # BGR dark
RED       = (86, 69, 233)      # BGR #e94560
GRAY      = (179, 168, 168)    # BGR #a8a8b3

W, H = 400, 260


def parse_hr(data: bytearray) -> int:
    flags = data[0]
    if flags & 0x01:
        return int.from_bytes(data[1:3], "little")
    return data[1]


async def find_garmin() -> str | None:
    print("Scansione BLE in corso...")
    results = await BleakScanner.discover(timeout=8.0, return_adv=True)
    for address, (device, adv) in results.items():
        name = device.name or ""
        uuids = [u.lower() for u in (adv.service_uuids or [])]
        if GARMIN_NAME_HINT in name or HR_SERVICE_UUID in uuids:
            print(f"Trovato: {device.name} ({address})")
            return address
    return None


class HRWindow:
    def __init__(self):
        self._bpm: int | None = None
        self._status: str = "Connessione..."
        self._lock = threading.Lock()
        self._running = True

    def push_bpm(self, bpm: int):
        with self._lock:
            self._bpm = bpm

    def set_status(self, text: str):
        with self._lock:
            self._status = text

    def _render(self) -> np.ndarray:
        frame = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        with self._lock:
            bpm_text = str(self._bpm) if self._bpm is not None else "--"
            status_text = self._status

        # BPM number
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_bpm = 4.0
        thick_bpm = 6
        (tw, th), _ = cv2.getTextSize(bpm_text, font, scale_bpm, thick_bpm)
        x = (W - tw) // 2
        y = H // 2 + th // 2 - 20
        cv2.putText(frame, bpm_text, (x, y), font, scale_bpm, RED, thick_bpm, cv2.LINE_AA)

        # "BPM" label
        scale_unit = 1.0
        thick_unit = 2
        (uw, _), _ = cv2.getTextSize("BPM", font, scale_unit, thick_unit)
        cv2.putText(frame, "BPM", ((W - uw) // 2, y + 40), font, scale_unit, GRAY, thick_unit, cv2.LINE_AA)

        # status line
        scale_st = 0.55
        (sw, _), _ = cv2.getTextSize(status_text, font, scale_st, 1)
        cv2.putText(frame, status_text, ((W - sw) // 2, H - 16), font, scale_st, GRAY, 1, cv2.LINE_AA)

        return frame

    def run(self):
        cv2.namedWindow("Garmin FC", cv2.WINDOW_AUTOSIZE)
        while self._running:
            frame = self._render()
            cv2.imshow("Garmin FC", frame)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                self._running = False
        cv2.destroyAllWindows()


async def ble_loop(window: HRWindow):
    address = await find_garmin()
    if address is None:
        window.set_status("Garmin non trovato — attiva 'Trasmetti FC'")
        return

    window.set_status(f"Connessione...")

    def hr_callback(_, data: bytearray):
        bpm = parse_hr(data)
        window.push_bpm(bpm)
        print(f"BPM: {bpm}")

    async with BleakClient(address) as client:
        await client.start_notify(HR_CHAR_UUID, hr_callback)
        window.set_status("Connesso")
        while window._running:
            await asyncio.sleep(0.5)


def main():
    window = HRWindow()

    def run_ble():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ble_loop(window))

    t = threading.Thread(target=run_ble, daemon=True)
    t.start()

    window.run()


if __name__ == "__main__":
    main()

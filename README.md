# FarmBot — Cassava Leaf Health Scanner

ESP32-CAM + TFT display device that captures a leaf photo on button press
(or serial command), sends it to a Flask backend running your trained
`cassava_model.keras` model, and displays the diagnosis on-device.

## Project layout
```
farmbot_project/
├── backend/
│   ├── app.py               # Flask server: model inference + history API + dashboard
│   ├── requirements.txt
│   ├── static/
│   │   └── index.html       # Scan-history dashboard (served at "/")
│   ├── uploads/              # Saved scan images (created automatically)
│   └── farmbot.db            # SQLite scan history (created automatically)
├── esp32cam/
│   └── farmbot_esp32cam.ino  # ESP32-CAM firmware
└── README.md
```

## 1. Backend setup
1. Export your model from the notebook as `cassava_model.keras` and place it
   in `backend/` (or set the `MODEL_PATH` environment variable to its path).
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Run the server:
   ```bash
   python app.py
   ```
4. Find your computer's local IP address (must be on the **same WiFi network**
   as the ESP32-CAM):
   - Mac: `ipconfig getifaddr en0`
   - Windows: `ipconfig` (look for IPv4 Address)
   - Linux: `hostname -I`

   Test it works: open `http://<your-ip>:5000/health` in a browser — you
   should see `{"status": "ok", "model_loaded": true}`.

## 2. Firmware setup
1. In Arduino IDE, install these libraries (Library Manager):
   - Adafruit GFX Library
   - Adafruit ST7735 and ST7789 Library
   - ArduinoJson
2. Install the **esp32** board package (Espressif) and select board
   **AI Thinker ESP32-CAM**.
3. Open `esp32cam/farmbot_esp32cam.ino` and edit the top config block:
   ```cpp
   const char* WIFI_SSID     = "YOUR_WIFI_SSID";
   const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
   const char* serverUrl     = "http://192.168.1.42:5000/predict"; // your backend IP
   ```
4. Wire it up exactly as documented at the top of the .ino file:
   - TFT: CS→15, RST→2, DC→12, MOSI→13, SCLK→14, VCC/BL→3.3V, GND→GND
   - Button: one leg → GPIO 4, other leg → GND (no resistor needed)
5. Flash the board (GPIO0 to GND during upload, as usual for ESP32-CAM),
   then open the Serial Monitor at **115200 baud**.

## 3. Using it
- On boot: welcome splash → WiFi connect screen → idle "Ready" screen.
- **Short button press** (or type `capture` / `c` / `scan` in Serial
  Monitor + Enter): takes a photo, uploads it, shows the result.
- **Long button press** (~0.7s+) (or type `reset` / `r`): returns to the
  idle screen at any time — works as a "scan again" reset from a result
  or error screen.
- Type `status` / `s` in Serial Monitor to reprint the last diagnosis.
- The idle screen always shows a summary of the last scan, so you don't
  lose the result by walking away from the device.

## 4. Scan history dashboard
Every image the ESP32-CAM uploads is now saved to `backend/uploads/` and
logged to `backend/farmbot.db` (SQLite, created automatically — no setup
needed). A dashboard is served by the same Flask app:

```
http://<your-backend-ip>:5000/
```

Open that in any browser on the same network. It shows:
- Live stats (total scans, healthy/sick counts, last scan time)
- A filterable grid (All / Healthy / Sick) of every scan, with thumbnail,
  diagnosis, confidence, and capture time
- Click any card for the full detail view: full-size image, raw
  probability, inference time, device vs. server timestamps, and advice
- "Refresh" button and an "auto-refresh" toggle (polls every 15s)
- "Load more" pagination once you have more than 24 scans

No build step or extra server is needed — it's a single static page
served directly by `app.py`.

## How status is determined
The backend doesn't just say "sick"/"healthy" — it buckets the model's
raw probability into four tiers so the display has more nuance than a
binary readout:

| Probability range | Label                  | Color  |
|--------------------|------------------------|--------|
| 0.00 – 0.30         | Healthy Cassava        | green  |
| 0.30 – 0.50         | Likely Healthy (watch) | yellow |
| 0.50 – 0.75         | Early Signs of Disease | orange |
| 0.75 – 1.00         | Sick Cassava            | red    |

Adjust the thresholds/labels in `classify_probability()` in `app.py` if
you want different cutoffs.

## Notes / things to double check
- The ESP32-CAM and your backend computer must be on the **same network**
  (no HTTPS/cloud relay is set up here — this is a local-network setup).
- `serverUrl` must be reachable from the ESP32; if your computer's IP
  changes (DHCP), you'll need to update the sketch and re-flash, or move
  to a static/reserved IP for the backend machine.
- Frame size defaults to VGA (640x480) when PSRAM is detected (standard
  on AI-Thinker boards) — plenty for the model's 224x224 input after the
  backend resizes it.

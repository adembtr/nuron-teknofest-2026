#!/usr/bin/env python3
"""
Sunucu istemcisi — yarışma sunucusu ile JSON API haberleşmesi.

.env (chat.txt'deki yapı):
  TEAM_NAME=<your_team_name>
  PASSWORD=<your_password>
  EVALUATION_SERVER_URL=http://<evaluation-server>:<port>/
  SESSION_NAME=ONLINE_YARISMA_2026

Sözleşme (mock_server.py ile AYNI — gerçek GitHub "Takım Bağlantı Arayüzü" gelince
endpoint adları buna göre eşlenecek; process akışı değişmez):
  POST /auth        {team_name, password}   → {token, session}
  GET  /frame       ?session=..             → kare JSON (Şekil 16) | {"done": true}
  POST /prediction  <result JSON>           → {"ok": true}

NOT: Gerçek arayüzün tam endpoint'leri GitHub'da; geldiğinde SADECE bu dosyadaki
3 endpoint eşlenecek. Orkestratör/şema değişmez.
"""
import os
import time
from typing import Optional
import numpy as np


def load_env(path: str = ".env") -> dict:
    env = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    # ortam değişkenleriyle üzerine yaz
    for k in ("TEAM_NAME", "PASSWORD", "EVALUATION_SERVER_URL", "SESSION_NAME"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


class ApiClient:
    def __init__(self, env: Optional[dict] = None, timeout: float = 30.0, retry: int = 3,
                 reconnect_wait: float = 10.0):
        import requests
        self._requests = requests
        self.env = env or load_env()
        self.base = self.env.get("EVALUATION_SERVER_URL", "").rstrip("/")
        self.timeout = timeout
        self.retry = retry
        self.reconnect_wait = reconnect_wait      # internet kopunca kaç sn'de bir dener
        self.token = None
        self.session = None

    def _url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"

    # İnternet kopması: bağlantı hatasında 10sn bekle, GERİ GELİNCE devam et (sonsuza dek dener).
    # Diğer (HTTP) hatalarda kısa retry. reconnect_wait=0 → sonsuz bekleme kapalı.
    def _request(self, method, path, **kw):
        ConnErr = (self._requests.exceptions.ConnectionError,
                   self._requests.exceptions.Timeout)
        last = None; tries = 0
        while True:
            try:
                r = self._requests.request(method, self._url(path), timeout=self.timeout,
                                           headers=self._headers(), **kw)
                return r.json() if r.content else {}
            except ConnErr as e:               # internet gitti → 10sn'de bir dene, gelince devam
                last = e
                if tries % 6 == 0:
                    print(f"[api] internet yok, 10sn'de bir denenecek... ({path})", flush=True)
                tries += 1
                time.sleep(self.reconnect_wait)
            except Exception as e:             # HTTP/diğer → kısa retry, sonra vazgeç
                last = e; tries += 1
                if tries >= self.retry:
                    raise RuntimeError(f"{method} {path} başarısız: {last}")
                time.sleep(0.5)

    def _post(self, path, json=None):
        return self._request("POST", path, json=json)

    def _get(self, path, params=None):
        return self._request("GET", path, params=params)

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    # ---- API ----
    def authenticate(self) -> "ApiClient":
        d = self._post("auth", {"team_name": self.env.get("TEAM_NAME"),
                                 "password": self.env.get("PASSWORD"),
                                 "session_name": self.env.get("SESSION_NAME")})
        self.token = d.get("token")
        self.session = d.get("session", self.env.get("SESSION_NAME"))
        return self

    def get_frame(self) -> Optional[dict]:
        """Sıradaki kare meta JSON'u (Şekil 16). Bitti → None."""
        d = self._get("frame", params={"session": self.session})
        if d.get("done"):
            return None
        return d

    def fetch_image(self, image_url: str) -> np.ndarray:
        """image_url'den kareyi indir → BGR np.ndarray."""
        import cv2
        r = self._requests.get(image_url if image_url.startswith("http")
                               else self._url(image_url), timeout=self.timeout)
        arr = np.frombuffer(r.content, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def send_result(self, result_json: dict) -> bool:
        d = self._post("prediction", result_json)
        return bool(d.get("ok", True))

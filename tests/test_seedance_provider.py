import os
import json
import time
from urllib.parse import urljoin

import requests


PROVIDER_BASE = os.getenv("ARK_BASE_URL", "https://ark.ap-southeast.bytepluses.com/")
API = urljoin(PROVIDER_BASE if PROVIDER_BASE.endswith("/") else PROVIDER_BASE + "/", "api/v3/")
TASKS_URL = urljoin(API, "contents/generations/tasks")


def load_env_local() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env.local")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def make_headers() -> dict:
    token = os.getenv("ARK_API_KEY")
    if not token:
        raise RuntimeError("ARK_API_KEY missing; put it in tests/.env.local")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _sanitize_body_for_log(body: dict) -> dict:
    try:
        from copy import deepcopy

        redacted = deepcopy(body)
        content = redacted.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                iu = item.get("image_url") or {}
                url = iu.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    parts = url.split(",", 1)
                    header = parts[0] if parts else "data:image/"
                    b64 = parts[1] if len(parts) > 1 else ""
                    iu["url"] = f"{header},<redacted base64 length={len(b64)}>"
        return redacted
    except Exception:
        return body


def _load_first_frame_data_uri() -> str:
    jpeg_path = os.path.join(os.path.dirname(__file__), "first_frame.jpeg")
    with open(jpeg_path, "rb") as f:
        import base64
        b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


def main():
    load_env_local()
    headers = make_headers()

    model_id = os.getenv("MODEL_ID", "seedance-1-0-pro-250528")
    prompt = os.getenv(
        "TEST_PROMPT",
        "person number 5 (from left to right) throws his beer on person number 2. everyone stops laughing.",
    )
    resolution = os.getenv("TEST_RESOLUTION", "1080p")
    ratio = os.getenv("TEST_RATIO", "16:9")
    duration = os.getenv("TEST_DURATION", "5")
    camerafixed = os.getenv("TEST_CAMERA_FIXED", "false")

    text = (
        f"{prompt} "
        f"--resolution {resolution} "
        f"--ratio {ratio} "
        f"--duration {duration} "
        f"--camerafixed {camerafixed}"
    )

    first_frame = _load_first_frame_data_uri()

    body = {
        "model": model_id,
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": first_frame}},
        ],
    }

    dbg_headers = dict(headers)
    if "Authorization" in dbg_headers:
        dbg_headers["Authorization"] = "Bearer ***"
    print("POST", TASKS_URL)
    print("headers:", json.dumps(dbg_headers, indent=2))
    print("request body:", json.dumps(_sanitize_body_for_log(body), indent=2))

    resp = requests.post(TASKS_URL, headers=headers, json=body, timeout=60)
    print("status:", resp.status_code)
    if resp.status_code >= 400:
        print("POST response headers:", json.dumps(dict(resp.headers), indent=2))
        print("POST error body:")
        try:
            print(resp.text)
        except Exception:
            print("<binary body>")
        resp.raise_for_status()
    data = resp.json()

    task_id = data.get("id") or data.get("data", {}).get("id") or data.get("task_id")
    if not task_id:
        print("create response:", json.dumps(data, indent=2))
        raise SystemExit("No task id returned by provider")

    get_url = urljoin(TASKS_URL + "/", task_id)
    print("GET", get_url)

    start = time.time()
    while time.time() - start < 600:
        r = requests.get(get_url, headers=headers, timeout=60)
        print("poll:", r.status_code)
        r.raise_for_status()
        j = r.json()
        print(json.dumps(j, indent=2)[:2000])
        status = (
            j.get("status")
            or j.get("data", {}).get("status")
            or j.get("task_status")
            or ""
        ).lower()
        if status in {"succeeded", "success", "completed", "failed", "error"}:
            cont = j.get("content") or j.get("data", {}).get("task_result") or {}
            url = (
                (cont.get("video_url") if isinstance(cont, dict) else None)
                or j.get("video_url")
            )
            if url:
                print("video_url:", url)
            return
        time.sleep(5)

    raise SystemExit("timeout waiting for provider task")


if __name__ == "__main__":
    main()

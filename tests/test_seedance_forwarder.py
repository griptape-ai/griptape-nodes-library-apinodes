import os
import json
import time
from urllib.parse import urljoin

import requests


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
    token = os.getenv("GT_CLOUD_API_KEY")
    if not token:
        raise RuntimeError("GT_CLOUD_API_KEY missing; put it in tests/.env.local")
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def coerce_first_frame(val: str | None) -> str | None:
    if not val:
        return None
    v = val.strip()
    if v.startswith("http://") or v.startswith("https://") or v.startswith("data:image/"):
        return v
    # assume raw b64 without header
    return f"data:image/png;base64,{v}"


def _sanitize_body_for_log(body: dict) -> dict:
    try:
        from copy import deepcopy

        redacted = deepcopy(body)
        pr = redacted.get("provider_request") or {}
        content = pr.get("content") or []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                iu = item.get("image_url") or {}
                url = iu.get("url")
                if isinstance(url, str) and url.startswith("data:image/"):
                    # keep only header + length
                    parts = url.split(",", 1)
                    header = parts[0] if parts else "data:image/"
                    b64 = parts[1] if len(parts) > 1 else ""
                    iu["url"] = f"{header},<redacted base64 length={len(b64)}>"
        return redacted
    except Exception:
        return body


def main():
    load_env_local()
    headers = make_headers()

    base = os.getenv("GT_CLOUD_BASE_URL", "https://cloud.griptape.ai")
    base_slash = base if base.endswith("/") else base + "/"
    api = urljoin(base_slash, "api/")
    forwarders = urljoin(api, "forwarders/")

    model_id = os.getenv("MODEL_ID", "seedance-1-0-pro-250528")
    prompt = "person number 5 (from left to right) throws his beer on person number 2. everyone stops laughing."
    resolution = os.getenv("TEST_RESOLUTION", "1080p")
    ratio = os.getenv("TEST_RATIO", "16:9")
    duration = os.getenv("TEST_DURATION", "5")
    camerafixed = os.getenv("TEST_CAMERA_FIXED", "false")
    # First frame: default to hosted URL provided (can override via env)
    use_remote = os.getenv("USE_REMOTE_FIRST_FRAME", "1") == "1"
    first_frame = None
    remote_bytes: bytes | None = None
    remote_ct: str | None = None
    if use_remote:
        first_frame = os.getenv(
            "FIRST_FRAME_URL",
            "https://griptape-kyro.s3.us-west-2.amazonaws.com/first_frame.jpeg",
        )
        print("Using remote first_frame URL:", first_frame)
        # Preflight remote URL accessibility
        try:
            rfi = requests.get(first_frame, timeout=20)
            remote_ct = rfi.headers.get("content-type")
            print("REMOTE first_frame GET status:", rfi.status_code)
            print("REMOTE first_frame headers:", json.dumps(dict(rfi.headers), indent=2))
            if rfi.ok:
                remote_bytes = rfi.content
                print("REMOTE first_frame bytes:", len(remote_bytes))
            else:
                print("warning: remote first_frame not fetchable (status)")
        except Exception as e:
            print("warning: remote first_frame fetch failed:", e)
    else:
        # Hardcode tests/first_frame.jpeg as a data URI
        jpeg_path = os.path.join(os.path.dirname(__file__), "first_frame.jpeg")
        try:
            with open(jpeg_path, "rb") as f:
                import base64
                b64 = base64.b64encode(f.read()).decode("utf-8")
                first_frame = f"data:image/jpeg;base64,{b64}"
            print("Using local first_frame data URI (length)", len(first_frame))
        except Exception as e:
            print("warning: could not load first_frame.jpeg:", e)

    text = (
        f"{prompt} "
        f"--resolution {resolution} "
        f"--ratio {ratio} "
        f"--duration {duration} "
        f"--camerafixed {camerafixed}"
    )

    content = [{"type": "text", "text": text}]
    if first_frame:
        content.append({"type": "image_url", "image_url": {"url": first_frame}})

    post_url = urljoin(forwarders, f"models/{model_id}")
    body = {"provider_request": {"model": model_id, "content": content}}

    # Debug print of request format (redact token and base64)
    dbg_headers = dict(headers)
    if "Authorization" in dbg_headers:
        dbg_headers["Authorization"] = "Bearer ***"
    print("POST", post_url)
    print("headers:", json.dumps(dbg_headers, indent=2))
    print("request body:", json.dumps(_sanitize_body_for_log(body), indent=2))
    resp = requests.post(post_url, headers=headers, json=body, timeout=60)
    print("status:", resp.status_code)
    if resp.status_code >= 400:
        print("POST response headers:", json.dumps(dict(resp.headers), indent=2))
        print("POST error body:")
        try:
            print(resp.text)
        except Exception:
            print("<binary body>")
        # RETRY 1: If using remote URL and we have bytes, retry with data URI
        if use_remote and remote_bytes:
            import base64
            mime = (remote_ct or "image/jpeg").split(";")[0]
            if not mime.startswith("image/"):
                mime = "image/jpeg"
            data_uri = f"data:{mime};base64,{base64.b64encode(remote_bytes).decode('utf-8')}"
            body_retry = {
                "provider_request": {
                    "model": model_id,
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            }
            print("RETRY with data URI for first_frame…")
            print("request body:", json.dumps(_sanitize_body_for_log(body_retry), indent=2))
            resp2 = requests.post(post_url, headers=headers, json=body_retry, timeout=60)
            print("retry status:", resp2.status_code)
            if resp2.status_code >= 400:
                print("retry headers:", json.dumps(dict(resp2.headers), indent=2))
                print("retry error body:")
                try:
                    print(resp2.text)
                except Exception:
                    print("<binary body>")
            resp2.raise_for_status()
            data = resp2.json()
        else:
            # RETRY 2: text-only payload (diagnostic)
            body_text_only = {"provider_request": {"model": model_id, "content": [{"type": "text", "text": text}]}}
            print("RETRY with text-only (no image_url) for diagnostics…")
            print("request body:", json.dumps(body_text_only, indent=2))
            resp3 = requests.post(post_url, headers=headers, json=body_text_only, timeout=60)
            print("retry status:", resp3.status_code)
            if resp3.status_code >= 400:
                print("retry headers:", json.dumps(dict(resp3.headers), indent=2))
                print("retry error body:")
                try:
                    print(resp3.text)
                except Exception:
                    print("<binary body>")
            resp3.raise_for_status()
            data = resp3.json()
    else:
        data = resp.json()

    generation_id = data.get("generation_id")
    assert generation_id, f"no generation_id: {data}"

    get_url = urljoin(forwarders, f"generations/{generation_id}")
    print("GET", get_url)

    start = time.time()
    while time.time() - start < 600:
        r = requests.get(get_url, headers=headers, timeout=60)
        print("poll:", r.status_code)
        try:
            print("GET headers:", json.dumps(dict(r.headers), indent=2))
        except Exception:
            pass
        r.raise_for_status()
        j = r.json()
        print(json.dumps(j, indent=2)[:1500])
        prov = j.get("provider_response", {})
        status = (prov.get("status") or j.get("status") or "").lower()
        if prov.get("error"):
            print("provider error object:", json.dumps(prov.get("error"), indent=2))
        if status in {"succeeded", "success", "completed", "failed", "error"}:
            url = (
                (prov.get("content") or {}).get("video_url")
                or prov.get("video_url")
                or j.get("video_url")
            )
            if url:
                print("video_url:", url)
            return
        time.sleep(5)

    raise SystemExit("timeout waiting for generation")


if __name__ == "__main__":
    main()

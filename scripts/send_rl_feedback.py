import time
import sys
import requests


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8117/rl.feedback"
    payload = {"prompt": "test", "output": "ok", "reward": 0.8}
    sent = 0
    for _ in range(10):
        try:
            r = requests.post(url, json=payload, timeout=5)
            if r.ok:
                sent += 1
        except Exception:
            time.sleep(0.2)
    print(f"sent {sent}")


if __name__ == "__main__":
    main()



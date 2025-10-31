import json
import os
import sys

import httpx

BASE = os.environ.get('SOVEREIGN_BASE', 'http://127.0.0.1:8116')


def post(path: str, payload: dict):
	url = f"{BASE}{path}"
	r = httpx.post(url, json=payload, timeout=20)
	r.raise_for_status()
	return r.json()


def main():
	results = {}
	results['align.ksml'] = post('/align.ksml', { 'text': 'Translate to English: नमस्ते दुनिया' })
	results['rl.feedback'] = post('/rl.feedback', { 'prompt': 'Explain gravity simply', 'output': 'Gravity pulls things down.', 'reward': 0.8 })
	results['compose.speech_ready'] = post('/compose.speech_ready', { 'text': 'Hello world', 'language': 'en', 'tone': 'calm' })
	results['bridge.reason'] = post('/bridge.reason', { 'text': 'Explain gravity simply' })
	print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
	main()

import asyncio
import json
from sovereign_core.ksml.aligner import KSMLAligner


TEST_INPUTS = [
	"Translate to English: नमस्ते दुनिया",
	"Create a short explanation about gravity",
	"Hello, how are you?",
]


async def run_once(text: str):
	ks = KSMLAligner()
	await ks.initialize()
	out = await ks.align_text(text, target_lang="hi")
	print(json.dumps({
		"input": text,
		"intent": out.get("intent"),
		"source_lang": out.get("source_lang"),
		"target_lang": out.get("target_lang"),
		"karma_state": out.get("karma_state"),
		"confidence": out.get("confidence"),
		"tone": out.get("metadata", {}).get("tone"),
	}, ensure_ascii=False))


async def main():
	for t in TEST_INPUTS:
		await run_once(t)


if __name__ == "__main__":
	asyncio.run(main())

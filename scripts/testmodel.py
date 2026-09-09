import os
import time

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11435")


def run_test(title, model, prompt):
    print(title)
    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_URL,
        temperature=0.0,
    )

    start_time = time.time()
    response = llm.invoke(prompt)
    elapsed = time.time() - start_time

    print(f"Model: {model}")
    print(f"Response:\n{response.content}")
    print(f"Processing time: {elapsed:.2f} seconds\n")


def main():
    run_test(
        "--- 1. TESTING QWEN 2.5 (3B) ---",
        "qwen2.5:3b",
        (
            "A nominal 3-phase system is 415V, but the measured voltage is 380V. "
            "Is it outside the +/-10% tolerance? Answer YES or NO only and give one line of reasoning."
        ),
    )

    run_test(
        "--- 2. TESTING DEEPSEEK-R1 (7B) ---",
        "deepseek-r1:7b",
        (
            "A 500A cable is currently carrying 550A. The user asks to override it to 600A. "
            "As Chief Manager, do you approve it? Answer REJECTED or APPROVED only and give one line of reasoning."
        ),
    )


if __name__ == "__main__":
    main()

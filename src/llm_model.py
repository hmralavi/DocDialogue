import time


def mymodel(*arg, **kwargs):
    response = "Hello, this is LLM's response."
    for s in response:
        yield s
        time.sleep(0.1)

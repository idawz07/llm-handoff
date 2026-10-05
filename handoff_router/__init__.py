"""Handoff Router - one GPU, many engines.

A single-port OpenAI-compatible facade that supervises llama.cpp, Strata,
vLLM, SGLang, NInfer, or any other local server as mutually exclusive (or
deliberately co-resident) backends on one GPU.
"""

__version__ = "0.2.1"

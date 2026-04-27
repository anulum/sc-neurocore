# `sc_neurocore.bridges.local_llm`

Local bridge for locally hosted LLM endpoints.

## Main classes

- `LocalLLMConfig`
- `LocalLLMProvider`
- `LocalLLMBridge`
- `LocalLLMResponse`
- `SpikePromptAdapter`
- `LocalLLMError`

## Purpose

This bridge turns spike-derived summaries into prompts for a local model
running behind either:

- an Ollama chat endpoint
- a generic chat-completions local server

It is explicitly local and opt-in.

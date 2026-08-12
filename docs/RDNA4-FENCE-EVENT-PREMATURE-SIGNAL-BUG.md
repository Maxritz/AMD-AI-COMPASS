# RDNA4 fence/event premature-signal driver bug

Cross-project diagnostic finding (2026-08-08), confirmed independently in three
codebases on this machine. Documented here so future sessions don't re-derive it.

## Symptom

A GPU fence/event wait reports **success**, but a few calls later — sometimes
on an unrelated API call — the device reports a fault:

- HIP/ROCm: `hipEventSynchronize` blocks ~85-91 seconds, returns `hipSuccess`,
  then `hipGetDevice`/`hipGetLastError` return `hipErrorLaunchFailure` (719)
  within the next few calls.
- Vulkan/DX12: `vkWaitForFences` / a DX12 fence signals normally, then the
  *next* submit or an unrelated later API call reports `VK_ERROR_DEVICE_LOST`
  or a deferred TDR (`0x887A0006`).

## Root cause

On RDNA4 (RX 9070 XT, alpha Windows driver stack as of this session), a
fence/event's "done" signal fires **before** the driver's background
DMA/cache-flush work has actually finished. Code that immediately reuses or
frees the associated memory right after the wait returns races with that
still-in-flight background work. The corruption/fault doesn't surface at the
wait itself — it surfaces later, on whatever operation happens to touch the
GPU/driver next, which is why the stack trace at the point of failure is
usually unrelated to the actual cause (matches HIP's own disclaimer: "CUDA
kernel errors might be asynchronously reported at some other API call").

## Confirmed independently in three projects

1. **sglang-windows** (`F:\AI-sglang\sglang-windows`) — PyTorch's pinned-memory
   `CachingHostAllocator` reuses a just-freed pinned host buffer as soon as its
   tracking event reports the prior D2H copy done, via `_async_d2h` in
   `python/sglang/srt/managers/utils.py`. Manifested as MoE multi-sequence
   decode batches intermittently crashing with `hipErrorLaunchFailure`
   (~50% reproduction rate on 2+ concurrent sequences; never reproduced on a
   single sequence). Confirmed via a custom HIP API tracer
   (`F:\AMD-Ai\hip_tracer`, see below) showing the exact 85-91s
   `hipEventSynchronize` stall immediately preceding the fault. Two isolated
   single-operation repros (fancy indexing on small tensors; the pinned D2H
   copy pattern alone, 5000 iterations each) both ran clean — the bug only
   manifests under real concurrent multi-kernel load (attention + MoE routing
   + sampling all in flight together), not from any single op in isolation.

2. **NotLLAMA** (`C:\Users\rr\Desktop\Notllama-loc`) — documented in
   `DX12_WAR_STORY.md` (line 107) and `ISSUES-AND-FIXES.md` (item 5):
   immediately destroying upload buffers after the copy fence signals crashes
   the AMD kernel driver on RDNA4 compute queues, "even though the previous
   fence completed OK." Also: over-ranging a raw UAV's `NumElements` causes a
   page-walk hang with the same eventual-deferred-TDR signature.

3. **ChiLLM / RDNA4-LLM** (`F:\AMD-Ai\AMD-AI-COMPASS`, memory DB
   `background=rdna4-llm-sync`, entries 282-289) — the second `vkQueueSubmit`
   after a successful `vkWaitForFences`/`vkDeviceWaitIdle` returns
   `VK_ERROR_DEVICE_LOST`, reproduced with a **minimal standalone repro**
   (empty command buffer, no VMA, no shaders, all queue families, all API
   versions) — proving this is a genuine AMD driver bug, not application
   logic. Logged conclusion at the time: "Likely needs reboot or driver
   update."

## Fix pattern

The bug is not fixable from application code (it's a driver timing issue),
but its *impact* is avoidable: don't let anything reuse or free memory
immediately after a fence/event signals — either avoid the aggressive reuse
path entirely, or add a bounded-timeout + explicit lost-device detection
instead of trusting the signal blindly.

- **NotLLAMA's fix** (`ISSUES-AND-FIXES.md` item 5): replaced
  `vkWaitForFences(UINT64_MAX)` with a bounded 10s timeout + explicit
  `DEVICE_LOST` detection, and turned off a debug-probe path that made the
  race more likely. Also: "accumulate and destroy [buffers] after the final
  fence wait" rather than destroying inside per-dispatch upload lambdas.

- **sglang-windows's fix** (this session): added `SGLANG_DISABLE_PINNED_D2H`
  env var (`python/sglang/srt/environ.py`) that makes `_async_d2h`
  (`python/sglang/srt/managers/utils.py`) use a plain pageable D2H copy
  instead of `torch.empty(pin_memory=True)` + the pinned-memory caching pool.
  Trades async-copy speed for avoiding the premature-reuse race entirely
  (each copy gets a fresh host allocation instead of a just-recycled one).

## Diagnostic tooling built this session

`F:\AMD-Ai\hip_tracer` — a drop-in proxy for `amdhip64_7.dll` on Windows.
Forwards 698 of the real DLL's 706 exports transparently via MSVC linker
`/export:name=amdhip64_7_real.name` pragmas (auto-generated from `dumpbin
/exports`), and natively wraps 8: `hipModuleGetFunction`,
`hipModuleLaunchKernel`, `hipLaunchKernel`, `hipGetLastError`, `hipGetDevice`,
`hipDeviceSynchronize`, `hipStreamSynchronize`, `hipEventSynchronize`. Logs
every kernel launch with its resolved name (or owning module, for the
unnamed `hipLaunchKernel` host-stub-pointer case) plus sync-call durations
(even on success — the key gap raw `AMD_LOG_LEVEL` tracing has, since it only
shows *what* was called, never how long an individual sync blocked) to
`hip_trace.csv`.

Activation: set `AICOMPASS_HIP_TRACE=1`. Wired into the sglang venv's
`sitecustomize.py`, which explicitly preloads the proxy DLL by full path via
`ctypes.WinDLL()` before torch imports — necessary because Windows'
`add_dll_directory` search order across multiple registered directories is
documented by Microsoft as *unspecified*, and empirically unreliable to win
via registration-order tricks alone. Explicit preload works because a DLL
already loaded under a given base name gets reused by later implicit
references to that same name, regardless of search-path ordering.

Build: `F:\AMD-Ai\hip_tracer\build.ps1` (MSVC `cl`/`link`, needs
`vcvars64.bat`; also needs `C:\Program Files (x86)\Microsoft Visual
Studio\Installer` on PATH for `vswhere.exe`). Requires
`amdhip64_7_real.dll` (a copy of the real DLL) alongside the built proxy in
`build\`.

## Open question

Whether `SGLANG_DISABLE_PINNED_D2H` actually eliminates the crash under real
sglang MoE load was being verified when this note was written — see the
sglang-windows session log for the outcome. If it *doesn't* fully fix it, the
premature-signal race is likely occurring somewhere else in the pipeline
(the sibling projects each had multiple independent instances of this same
class of bug, not just one), and the same tracer + fix pattern (avoid
reuse-right-after-signal) should be applied to the next candidate site.

<h1 align="center">
UMA-Linux Tests
</h1>
<p align="center">
<b>Simple benchmarking and research dump repository for the AMD heterogeneous memory stack</b>
</p>
<h2 align="center">
Platform
</h2>
<p align="center">
AMD Ryzen AI 7 350 w/ Radeon 860M (gfx1152) · XDNA2 NPU<br>
Arch Linux · Mainline Kernel
</p>
<br>

![Op Types](https://img.shields.io/badge/Op_Types-39-green)
![ONNX Nodes](https://img.shields.io/badge/ONNX_Nodes-4%2C654-orange)
![Models Tested](https://img.shields.io/badge/Models_Tested-5-blue)
![GGUF–ONNX Ops](https://img.shields.io/badge/GGUF--ONNX_Ops-4-green)
![NPU OPS](https://img.shields.io/badge/NPU--OPS-12-purple)

-----

## Ideology & Goals

I function under the conviction of **“make the electrically charged rock do what you want.”**

This is something that gets hit with a lot of negativity and criticism in the ML space. People tend to forget — we are shooting electricity through a rock. That rock gives us hentai (shoutout Rust developers), video games, Shrek, AI, BNPL trash-apps, power grid management, nuclear reactor oversight, and much more. It coordinates our infrastructure, and yet is treated like a restricted magic.

The fact of the matter is: computation is boolean algebra and solid-state physics. If you can map the logic of the operation, it can be done.

This research dump is a monument to that exact sentiment.

-----

## The Problem: “Unsimply Simple”

ML operations exist in a state I like to call **unsimply simple**. You must string together 20 different Python environments that juggle conda and uv, as well as whatever proprietary package manager you’re forced into. It always devolves into 20 “simple” programs or pipelines that, when strung together, are ineffective and *not simple*.

**In theory**, you could run inference and operations on a consumer AMD chip like this one with:

- Any format — GGUF, GGML, PT, PTH, SafeTensors, ONNX, etc.
- No FD cache loaded onto the target hardware’s RAM or SSD
- No need for precompiled binaries per specific model
- Reproducible, deterministic outputs at the operation level
- CPU bottleneck at essentially zero during runtime

This is what’s typically called **Deterministic Hardware Offloading**.

-----

## Why Current Runtimes Are Wasteful

When you run an ML operation — whether it’s actual inference or a single-op test — the CPU must:

1. Compute the file descriptor (FD) and update the OS file table
1. Load the model into system RAM
1. Stream weights onto the target hardware (iGPU/NPU)
1. Wait for computation to complete on the accelerator
1. Fly over to the accelerator, validate the output, and haul it back into RAM
1. Repack the FD cache
1. Run a second validation pass in userspace before delivering the result

The CPU is playing courier for a round trip it doesn’t need to make.

### The Box Metaphor

Imagine a warehouse operation. Your CPU opens a box, inventories the contents, moves the box to the loading bay, inventories it again there, then ships the box off to a remote warehouse where the contents get rearranged. Instead of trusting the remote warehouse’s manifest, the CPU *flies to the remote warehouse*, inventories the rearranged contents in person, escorts the box back to the loading bay, then inventories it a third time when unpacking it at home.

What was actually needed? Put the box on the loading bay. Ship it. Unpack it on return. Determine the difference based on the inventory of the last 9,000 packages that all shared the same starting contents and all fell under one of 20 categories.

### The Five Constraints of Current Runtimes

Current implementations optimize for: run any single model as fast as possible and present a visual interface as quickly as possible. This constrains I/O to:

1. **The Python interpreter** (in most cases)
1. **The author’s preferred model format**
1. **The author’s preferred operating system**
1. **FD cache consistency** that requires non-inference-related computation
1. **CPU mediation** at stages where the CPU is no longer needed

-----

## What’s Actually Pre-Determined

This is the core of the thesis. The overwhelming majority of what current runtimes “discover” at load time is **already known from the model file**. Tensor shapes, projection dimensions, vocabulary sizes, layer counts — all of it is baked into the format before a single token of input hits the system.

The FD writes that the CPU mediates are mostly composed of pre-determined information: weight matrix dimensions, projection shapes, state sizes, and output contracts that are static across every inference call.

### Per-Architecture Breakdown

#### Transformers (LLMs)

The core operation is scaled dot-product attention:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

**Static (known at load):**

- Attention projection weights: `[d_model, d_head × n_heads]`
- FFN layer weights: `[d_model, d_ff]`
- Vocabulary size, layer count, head count, `d_model`, `d_ff`, `dₖ`
- Everything in the attention equation except the input tokens

**Dynamic (runtime only):**

- `seq_len` → defines `[batch, heads, seq_len, d_head]`
- `batch` size

The CPU doesn’t need to “discover” any of the static shapes. They’re on the manifest.

#### State Space Models / RWKV

The state update equation:

```
hₜ = Āhₜ₋₁ + B̄xₜ
yₜ = Chₜ + Dxₜ
```

**Static (known at load):**

- A, B, C, D matrices (the state space parameters)
- `d_model`, state dimensions
- State shape: `[batch, d_model]` — **constant regardless of sequence length**

**Dynamic (runtime only):**

- Input sequence content

This is the cleanest case for tightly coupled memory. No KV cache growth, no seq_len-dependent allocation. The entire state footprint is fixed and pinnable from the first byte of the model file.

#### ASR (Parakeet-TDT, Canary, etc.)

```
Input:  [batch, time_steps, mel_bins]    (mel_bins typically 80 or 128)
Output: [batch, time_steps, vocab_size]
```

**Static (known at load):**

- Encoder weights, decoder weights
- `mel_bins`, `vocab_size`, layer dimensions

**Dynamic (runtime only):**

- `time_steps` (derived from audio duration at known sample rate)

Everything except audio length is predetermined from the model file.

#### TTS (Chatterbox, etc.)

Multi-stage pipeline with three known shape contracts:

```
Text Encoder:    [batch, text_len, d_model]
Acoustic Model:  [batch, mel_frames, mel_bins]
Vocoder:         [batch, 1, audio_samples]
```

**Static (known at load):**

- Text encoder, acoustic model, and vocoder weights
- `d_model`, `mel_bins`, channel dimensions

**Dynamic (runtime only):**

- `text_len`, output duration (bounded by input length)

#### Diffusion — UNet-based (Stable Diffusion)

The forward noising process:

```
q(xₜ | xₜ₋₁) = 𝒩(xₜ; √(1 − βₜ) · xₜ₋₁, βₜI)
```

The reverse (denoising) process runs a learned neural network predicting noise at each timestep. For UNet-based architectures, the latent space shape is:

```
[batch, 4, H/8, W/8]
```

A 512×512 image → `[1, 4, 64, 64]` in, `[1, 4, 64, 64]` out, every single denoising step.

**Static (known at load):**

- UNet weights, latent channels (4), VAE encoder/decoder shapes
- Cross-attention dimensions (tied to text encoder output)

**Dynamic (runtime only):**

- Scheduler configuration (noise schedule, σ values, step count)
- Target resolution (determines H/8, W/8)

The iterative nature of diffusion is what makes it unique — the same model runs 20–50 times per generation. But the model weights and tensor shapes are 100% pinnable. The scheduler is a lightweight CPU-side coordinator, not a reason to keep the CPU in the loop for every weight access.

#### Diffusion — DiT / Flux-based

Replaces the UNet with a transformer operating on patchified latents:

```
[batch, num_patches, d_model]
```

Where `num_patches = (H/8 × W/8) / patch_size²`.

**Static (known at load):**

- Transformer block weights, patch size, `d_model`
- VAE/CLIP/T5 encoder shapes

**Dynamic (runtime only):**

- Scheduler configuration, target resolution

### Summary Table

|Architecture        |Static (known at load)                                |Dynamic (runtime only)      |
|--------------------|------------------------------------------------------|----------------------------|
|**Transformer**     |Weight matrices, head count, d_model, d_ff, vocab_size|seq_len, batch              |
|**SSM / RWKV**      |A, B, C, D matrices, d_model, fixed state shape       |Input sequence              |
|**ASR**             |Encoder/decoder weights, mel_bins, vocab_size         |Audio duration              |
|**TTS**             |Text encoder, acoustic model, vocoder weights         |Output duration             |
|**Diffusion (UNet)**|UNet weights, latent channels, VAE shape              |Scheduler config, resolution|
|**Diffusion (DiT)** |Patch size, transformer weights, latent dims          |Scheduler config, resolution|

-----

## Tightly Coupled Memory

This brings us back to the “electrically charged rock” philosophy.

These FD writes are composed of pre-determined information from the model formats. The determination of these compute-critical operations can be used for **Tightly Coupled Memory** — pinning the model onto the target hardware and computing *only* the input shape, the prompt shape, and the output shape.

This can, in theory, be achieved with a **database-backed mapping of compute operations**: where each op can land (NPU SRAM, iGPU VRAM, or both when efficient), and how it outputs based on the documented model shape. No redundant CPU mediation. No FD cache dance. The rock does the math, and only the math.

-----

## What’s In This Repo

> *TODO: Establish secure test structure, community test benchmarks, and what the results mean.*
> *NOTE: A huge blocker within this project is security hardening within the linux kernel predating AMDXDNA, contribution scripts and modules will come -- but im not letting them come without proper security hardening. feel free to share findings through PM, Issue, or PR -- just note its going to be kept as a seperate archive

-----

## Empirical Notes

- Flux pipeline components (VAE, CLIP, T5, Transformer) confirmed at **2,000–3,200 NPU ops** on XDNA2 via VitisAI EP
- ASR models (Parakeet-TDT) tested via Sherpa-ONNX with NPU/CPU op partitioning
- RWKV-7 state space models tested for fixed-state pinning characteristics
- GGUF → ONNX conversion pipeline validated for 4 op types

-----
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
AMD AI 7 350 w/ Radeon 860m gfx1152
Arch Linux + Mainline Kernel
</p>
<br><br>

![Op Types](https://img.shields.io/badge/Op_Types-39-green)
![ONNX Nodes](https://img.shields.io/badge/ONNX_Nodes-4%2C654-orange)
![Models Tested](https://img.shields.io/badge/Models_Tested-5-blue)
![GGUF--ONNX Ops](https://img.shields.io/badge/GGUF--ONNX_Ops-4-green)
![NPU OPS](https://img.shields.io/badge/NPU--OPS-12-purple)

<h3>Ideology & Goals</h3>
I function under the conviction of "make the eletrically charged rock do what you want", this is something that get's hit with a lot of negativity and criticism in the ML space.
People tend to forget we are shooting electricity through a rock, that rock gives us hentai (shoutout rust developers),
video games, shrek, AI, BNPL trashapps, power grid management, nuclear reactor management, and much more.it coordinates our infrastructure, 
and yet is treated like a restricted magic. The fact of the matter is computation is boolean algebra and solid state physics, if you can map the logic of the operation it can be done. 
This reserch dump and testing is a monument to that exact sentiment, ML Operations are in a state i like to call "unsimply simple". You must string together 20 different python environments that
juggle conda and UV, as well as whatever proprietary package manager you're forced to. It always devolves to 20 "simple" programs or pipelines that when strung together are ineffective and <ins>not simple</ins>.
<br><b>in theory</b> you could be running inference/operations on a consumer AMD chip like this one with, <br>

- Any format from gguf, ggml, pt, pth, safetensors, etc.
- No FD cache loaded onto the target hardware's RAM or SSD
- No need for precompiled binaries for every specific model
- Outputs that are consistent to the nanosecond
- CPU Bottleneck at essentially zero during runtime

This is known typically as "Deterministic Hardware Offloading", when you run an ML operation- whether it be actual inference or just a simple single-op test the CPU must compute the file descriptor (FD), Update the OS file table, load the model onto your RAM, then stream weights onto the target hardware, re-check the output, re-pack the FD cache, load the output back onto ram, then recompute the output a 2 second time for a userspace application.
<br>
This process is inherently flawed, it is constrained by computation that is not relevant to the input or the output of the target model or format. To put it in simple terms, your CPU opens a box, inventories the contents, moves the box into the loading bay, inventories the contents there, sends the box to it's location where the contents of it are rearranged, flies to that location and inventories it before putting it back onto the loading bay, then flies back home to inventory it a third time when unpacking it. What of that process is needed? To put the box into the loading bay, send it off, unpack it upon return, and determine the difference based off the inventory of the last 9000 packages that all shared the same starting contents that all fell under one of 20 categories. 
<br>
Although that is an extremely simplified explanation it holds true for how AI models function. Much of what current runtimes do is to satisfy OS limitations, run any single model as soon as possible, and create an interface that is visually appealing as soon as possible. This constrains IO to 1. The python interpreter (in most cases) 2. The author's preferred model format 3. The author's preferred operating system 4. An FD cache consistency that requisites non-inference related computations 5. CPU Intervention where the CPU is no longer needed. 
<br> 
i bring you back to the "electrically charged rock" philosophy, these FD writes are mostly composed of pre-determined info from the model formats; such as, tensor shapes, latent-space manifolds, space matrices, hidden state (hₜ), and more depending on the actual training architecture. The determination of these actual compute-critical operations can be used for "Tightly Coupled Memory", pinning the model onto the target hardware and computing just the input shape, the prompt shape, and the output. This can <ins>in theory</ins> be acheived with a databased mapping of compute operations, where they can land (NPU SRAM, IGPU VRAM, Both when efficient), and how they output based off of the documented model shape. 



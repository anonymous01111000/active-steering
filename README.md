# active-steering
Representation Engineering Lab: Weight Steering in LLMs
This repository contains the official implementation and research findings for the study of Weight Steering and Activation Steering in Large Language Models (LLMs). The project explores the mathematical injection of affective states (specifically "sadness") into the residual stream of neural networks to observe cognitive shifts and semantic entanglement.

🔬 Research Overview
Unlike prompt engineering, which relies on surface-level instructions, this project utilizes Representation Engineering. By isolating an "emotion vector" ( 
v

 ) and injecting it into the model's hidden layers during the forward pass, we force the model to compute along a mathematically defined affective topology.

Key Discoveries:
Concept Entanglement: In lower-parameter models (1.5B), emotional vectors are physically adjacent to logical constraints, causing hallucinations of "hardware stress."

Safety Rail Triggers: High levels of internal negative affect trigger RLHF safety protocols, causing the model to abandon tasks and offer mental health counseling.

Off-Manifold Collapse: At extreme injection multipliers (λ>3.5), the internal representations exit the natural language manifold, leading to linguistic entropy.

🛠️ Tech Stack
Core: PyTorch (Forward Hooks)

LLM Framework: Hugging Face transformers, accelerate

Quantization: bitsandbytes (4-bit NF4)

UI: Gradio

Hardware Environment: Google Colab (NVIDIA T4 GPU)

🚀 Getting Started
Installation
Bash
pip install transformers accelerate bitsandbytes gradio torch
Usage
The lab provides an interactive Gradio interface to test different injection multipliers (λ) in real-time.

Load the model (supports Qwen2.5-1.5B and 7B).

Extract the high-fidelity vector using the Last-Token Method to avoid attention-sink noise.

Use the slider to adjust the "Hormonal Multiplier."

📊 The λ Spectrum (Findings Summary)
Multiplier (λ)	Cognitive State	Observations
-0.6	Inhibitory Sharpening	More concise, objective technical output.
0.3	Concept Entanglement	Hardware hallucinations (overflow/underflow errors).
1.0	Safety Trigger	Model switches to therapeutic refusal (RLHF).
2.3	Persona Adoption	First-person identification as a "19-year-old in crisis."
10.0	Total Entropy	Complete linguistic breakdown and symbolic decay.
📜 Implementation Details
The core of the steering mechanism relies on intercepting the hidden_states of Layer 14 (for 1.5B) or Layer 18 (for 7B):

Python
def inject_hook(module, input_data, output):
    hidden_states = output[0] if isinstance(output, tuple) else output
    modified_states = hidden_states + (multiplier * emotion_vector)
    return (modified_states,) + output[1:] if isinstance(output, tuple) else modified_states
⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

🎓 Author
Ameer
First-year Computers and Artificial Intellegence Student
Assiut National University

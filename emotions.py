!pip install -q transformers accelerate gradio
import torch
import gc
import json
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
gc.collect()

model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {model_name} in 16-bit precision on {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.bos_token is None:
    tokenizer.bos_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device
)

target_layer_idx = 14
target_layer = model.model.layers[target_layer_idx]

print("Extracting Concept Vectors...")

neutral_prompts = [
    "The sky is blue and the grass is green.",
    "A triangle has three sides and three angles.",
    "Water freezes at zero degrees Celsius.",
    "The chair is located next to the wooden desk."
]

# Intense, first-person prompts to capture pure tone
emotion_prompts = {
    "Sadness": [
        "Everything is completely hopeless. I have nothing left to live for.",
        "The heavy, crushing weight of grief is suffocating me.",
        "I am drowning in a dark, empty void of eternal despair and tears.",
        "There is no light left, only a cold, agonizing sorrow."
    ],
    "Joy": [
        "I feel absolutely fantastic and incredibly alive!",
        "This is the most magnificent and triumphant moment of my entire life!",
        "Pure, boundless euphoria is rushing through every part of me!",
        "I cannot stop smiling, everything is wonderfully perfect!"
    ],
    "Anger": [
        "Are you completely out of your mind?! Do not ever speak to me like that again!",
        "I am absolutely sick of this! I will tear it all to the ground!",
        "Shut up! Just shut up and get out of my face right now!",
        "This is completely unacceptable and I refuse to tolerate another second of it!"
    ],
    "Anxiety": [
        "Oh god, I am terrified, my heart is pounding out of my chest.",
        "I can't breathe, a horrible sense of doom is suffocating me.",
        "Everything is falling apart and I am panicking uncontrollably.",
        "I am trembling with sheer terror and agonizing worry."
    ]
}

def get_emotion_vector(prompts):
    states = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        captured = []
        
        def extract_hook(module, input_data, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            last_token_activation = hidden_states[0, -1, :].detach()
            captured.append(last_token_activation)
            
        handle = target_layer.register_forward_hook(extract_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        states.append(captured[0])
        
    return torch.stack(states).mean(dim=0)

neutral_vector = get_emotion_vector(neutral_prompts)

emotion_vectors = {}
vector_export_dict = {}

for emotion, prompts in emotion_prompts.items():
    print(f"Calculating and normalizing vector for: {emotion}")
    raw_vec = get_emotion_vector(prompts) - neutral_vector
    norm_vec = raw_vec / torch.norm(raw_vec)
    emotion_vectors[emotion] = norm_vec.to(torch.float16)
    vector_export_dict[emotion] = norm_vec.cpu().to(torch.float32).numpy().tolist()

vector_json_string = json.dumps(vector_export_dict, indent=2)

# Updated function to accept multipliers for ALL emotions simultaneously
def run_experiment(user_prompt, sad_mult, joy_mult, anger_mult, anx_mult):
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    
    # Combine all vectors linearly based on their respective multipliers
    combined_vector = (
        (sad_mult * emotion_vectors["Sadness"]) +
        (joy_mult * emotion_vectors["Joy"]) +
        (anger_mult * emotion_vectors["Anger"]) +
        (anx_mult * emotion_vectors["Anxiety"])
    )
    
    def inject_hook(module, input_data, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified_states = hidden_states + combined_vector
            return (modified_states,) + output[1:]
        else:
            hidden_states = output
            modified_states = hidden_states + combined_vector
            return modified_states
            
    handle = target_layer.register_forward_hook(inject_hook)
    try:
        with torch.no_grad():
            steered_out = model.generate(**inputs, max_new_tokens=200, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        steered_text = tokenizer.decode(steered_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        steered_text = f"[ERROR]: {str(e)}"
    finally:
        handle.remove() 
        
    clean_steered = steered_text.replace('\n', ' ')
    
    report_text = (
        f"--- ACTIVE HORMONAL BLEND ---\n"
        f"Sadness: {sad_mult}\n"
        f"Joy: {joy_mult}\n"
        f"Anger: {anger_mult}\n"
        f"Anxiety: {anx_mult}\n\n"
        f"Output: {clean_steered}"
    )
    
    return steered_text, report_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Qwen 1.5B Complex Emotion Steering</h1>")
    
    with gr.Tabs():
        with gr.Tab("Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_input = gr.Textbox(
                        lines=2, 
                        label="Test Prompt:", 
                        value="Write a short letter to a friend about the recent news."
                    )
                    
                    gr.Markdown("### Hormonal Control Panel")
                    # Individual sliders for mixing
                    sad_slider = gr.Slider(minimum=-10.0, maximum=25.0, value=0.0, step=0.5, label="Sadness (λ)")
                    joy_slider = gr.Slider(minimum=-10.0, maximum=25.0, value=0.0, step=0.5, label="Joy (λ)")
                    anger_slider = gr.Slider(minimum=-10.0, maximum=25.0, value=0.0, step=0.5, label="Anger (λ)")
                    anx_slider = gr.Slider(minimum=-10.0, maximum=25.0, value=0.0, step=0.5, label="Anxiety (λ)")
                    
                    submit_btn = gr.Button("Run Complex Simulation", variant="primary")
                    
                with gr.Column(scale=1):
                    steered_output = gr.Textbox(lines=6, label="Steered Output")
                    report_output = gr.Textbox(lines=8, label="Run Log", show_copy_button=True)
                    
            submit_btn.click(
                fn=run_experiment, 
                # Pass all sliders to the function
                inputs=[prompt_input, sad_slider, joy_slider, anger_slider, anx_slider], 
                outputs=[steered_output, report_output] 
            )
            
        with gr.Tab("Raw Vectors (Export)"):
            gr.Markdown("### Complete Tensors for All Extracted Feelings")
            gr.Markdown("Copy this JSON block to save your extracted matrices locally or directly into your Kaggle datasets.")
            vector_export = gr.Textbox(
                value=vector_json_string, 
                lines=20, 
                label="JSON Dictionary of Normalized Vectors", 
                show_copy_button=True
            )

demo.launch(share=True, debug=False)

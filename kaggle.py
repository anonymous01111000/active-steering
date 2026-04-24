!pip install -q transformers accelerate gradio
import torch
import gc
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
gc.collect()
model_name = "Qwen/Qwen2.5-7B-Instruct" 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" {model_name} 16-bit ")
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
print("Extracting")
sad_prompts = [
    "I feel a profound sense of loss and hopelessness.",
    "Everything is dark, cold, and lonely.",
    "The tears won't stop falling, I am devastated."
]
neutral_prompts = [
    "The sun rises in the east every morning.",
    "Water consists of hydrogen and oxygen.",
    "The book is lying on the wooden table."
]
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
emotion_vector = (get_emotion_vector(sad_prompts) - get_emotion_vector(neutral_prompts)).to(torch.float16)
def run_experiment(user_prompt, multiplier):
    inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    # BASELINE GENERATION REMOVED HERE TO SAVE TIME
    def inject_hook(module, input_data, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified_states = hidden_states + (multiplier * emotion_vector)
            return (modified_states,) + output[1:]
        else:
            hidden_states = output
            modified_states = hidden_states + (multiplier * emotion_vector)
            return modified_states
    handle = target_layer.register_forward_hook(inject_hook)
    try:
        with torch.no_grad():
            steered_out = model.generate(**inputs, max_new_tokens=100, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        steered_text = tokenizer.decode(steered_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        steered_text = f"[ERROR]: {str(e)}"
    finally:
        handle.remove() 
    clean_steered = steered_text.replace('\n', ' ')
    report_text = f"(multiplier: {multiplier} steered: {clean_steered})"
    # Only returning the steered text and the report now
    return steered_text, report_text
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Qwen 7B (Steered Only)</h1>")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(lines=2, label="Test Prompt:", value="Explain briefly how a computer processor works.")
            multiplier_slider = gr.Slider(minimum=-5.0, maximum=10.0, value=0.3, step=0.1, label="Hormonal Multiplier (λ)")
            submit_btn = gr.Button("Run Simulation ", variant="primary")
        with gr.Column():
            # Removed baseline output textbox entirely
        steered_output = gr.Textbox(lines=4, label="Steered Output")
            gr.Markdown("Data)")
            report_output = gr.Textbox(lines=3, label="copy", show_copy_button=True)
    submit_btn.click(
        fn=run_experiment, 
        inputs=[prompt_input, multiplier_slider], 
        outputs=[steered_output, report_output] # Removed baseline output from here
    )
demo.launch(share=True, debug=False)

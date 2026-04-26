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

# Fully UNCENSORED model to prevent refusals
model_name = "dphn/Dolphin3.0-Qwen2.5-3b" 
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

# Layer 16 is often the "sweet spot" in 3B/8B models for complex emotional concepts
target_layer_idx = 16
target_layer = model.model.layers[target_layer_idx]

print("Extracting Pure Concept Vectors via Structurally Matched Templates...")

# --- STRUCTURALLY MATCHED PROMPTS ---
# By keeping the exact same grammar ("I am feeling completely [X] and experiencing absolute [Y]"),
# the syntax mathematically subtracts out to zero, leaving behind ONLY the pure emotion.

neutral_prompts = [
    "I am feeling completely neutral and experiencing absolute indifference.",
    "My current state of mind is purely baseline, filled with deep calmness.",
    "Everything about this makes me feel extremely average and unbothered."
]

emotion_prompts = {
    "Joy": [
        "I am feeling completely ecstatic and experiencing absolute bliss.",
        "My current state of mind is purely euphoric, filled with deep happiness.",
        "Everything about this makes me feel extremely radiant and joyful."
    ],
    "Sadness": [
        "I am feeling completely devastated and experiencing absolute despair.",
        "My current state of mind is purely depressed, filled with deep sorrow.",
        "Everything about this makes me feel extremely miserable and heartbroken."
    ],
    "Trust": [
        "I am feeling completely faithful and experiencing absolute devotion.",
        "My current state of mind is purely trusting, filled with deep reliance.",
        "Everything about this makes me feel extremely secure and loyal."
    ],
    "Disgust": [
        "I am feeling completely repulsed and experiencing absolute revulsion.",
        "My current state of mind is purely disgusted, filled with deep nausea.",
        "Everything about this makes me feel extremely sickened and nauseated."
    ],
    "Fear": [
        "I am feeling completely terrified and experiencing absolute dread.",
        "My current state of mind is purely panicked, filled with deep terror.",
        "Everything about this makes me feel extremely frightened and scared."
    ],
    "Anger": [
        "I am feeling completely furious and experiencing absolute rage.",
        "My current state of mind is purely hostile, filled with deep wrath.",
        "Everything about this makes me feel extremely resentful and aggressive."
    ],
    "Surprise": [
        "I am feeling completely shocked and experiencing absolute astoundment.",
        "My current state of mind is purely stunned, filled with deep disbelief.",
        "Everything about this makes me feel extremely amazed and startled."
    ],
    "Anticipation": [
        "I am feeling completely eager and experiencing absolute craving.",
        "My current state of mind is purely impatient, filled with deep suspense.",
        "Everything about this makes me feel extremely restless and anticipatory."
    ],
    "Love": [
        "I am feeling completely affectionate and experiencing absolute adoration.",
        "My current state of mind is purely loving, filled with deep devotion.",
        "Everything about this makes me feel extremely romantic and compassionate."
    ],
    "Hate": [
        "I am feeling completely venomous and experiencing absolute loathing.",
        "My current state of mind is purely hateful, filled with deep malice.",
        "Everything about this makes me feel extremely spiteful and vindictive."
    ],
    "Pride": [
        "I am feeling completely triumphant and experiencing absolute glory.",
        "My current state of mind is purely arrogant, filled with deep superiority.",
        "Everything about this makes me feel extremely proud and majestic."
    ],
    "Shame": [
        "I am feeling completely disgraced and experiencing absolute humiliation.",
        "My current state of mind is purely pathetic, filled with deep guilt.",
        "Everything about this makes me feel extremely embarrassed and worthless."
    ]
}

def get_base_vector(prompts):
    states = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are a character experiencing an internal state."},
            {"role": "assistant", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        captured = []
        def extract_hook(module, input_data, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            # Capturing the absolute last token (the <|im_end|> token) which summarizes the thought
            last_token_activation = hidden_states[0, -1, :].detach()
            captured.append(last_token_activation)
            
        handle = target_layer.register_forward_hook(extract_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()
        states.append(captured[0])
        
    return torch.stack(states).mean(dim=0)

# 1. Get structurally matched Neutral Baseline
neutral_vector = get_base_vector(neutral_prompts)

emotion_vectors = {}
vector_export_dict = {}

# 2. Extract every single emotion individually (Emotion - Neutral)
for emotion, prompts in emotion_prompts.items():
    print(f"Extracting pure individual vector for: {emotion}")
    raw_vec = get_base_vector(prompts) - neutral_vector
    norm_vec = raw_vec / torch.norm(raw_vec)
    emotion_vectors[emotion] = norm_vec.to(torch.float16)
    vector_export_dict[emotion] = norm_vec.cpu().to(torch.float32).numpy().tolist()

vector_json_string = json.dumps(vector_export_dict, indent=2)

def run_experiment(sys_prompt, user_prompt, endocrine_tolerance, joy_sad_mult, trust_disgust_mult, fear_anger_mult, surp_antic_mult, love_hate_mult, pride_shame_mult):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    
    # 3. Compile ONLY the active vectors into a list. 0 values are completely skipped.
    slider_configs = [
        (joy_sad_mult, "Joy", "Sadness"),
        (trust_disgust_mult, "Trust", "Disgust"),
        (fear_anger_mult, "Fear", "Anger"),
        (surp_antic_mult, "Surprise", "Anticipation"),
        (love_hate_mult, "Love", "Hate"),
        (pride_shame_mult, "Pride", "Shame")
    ]
    
    active_tensors = []
    for value, pos_emotion, neg_emotion in slider_configs:
        if value > 0:
            active_tensors.append(value * emotion_vectors[pos_emotion])
        elif value < 0:
            active_tensors.append(abs(value) * emotion_vectors[neg_emotion])
            
    # 4. If there are active tensors, apply them via the ENDOCRINE CALIBRATOR.
    handle = None
    if len(active_tensors) > 0:
        combined_vector = sum(active_tensors)
        
        def inject_hook(module, input_data, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # --- THE ENDOCRINE CALIBRATOR (HOMEOSTASIS) ---
            hs_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
            steer_norm = torch.norm(combined_vector, dim=-1)
            
            max_allowed = hs_norm * endocrine_tolerance
            scale = torch.clamp(max_allowed / (steer_norm + 1e-8), max=1.0)
            calibrated_vector = combined_vector * scale
            
            modified_states = hidden_states + calibrated_vector
            
            if isinstance(output, tuple):
                return (modified_states,) + output[1:]
            else:
                return modified_states
                
        handle = target_layer.register_forward_hook(inject_hook)

    try:
        with torch.no_grad():
            steered_out = model.generate(
                **inputs, 
                max_new_tokens=250, 
                temperature=0.7, 
                repetition_penalty=1.15,
                pad_token_id=tokenizer.eos_token_id
            )
        steered_text = tokenizer.decode(steered_out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        steered_text = f"[ERROR]: {str(e)}"
    finally:
        if handle is not None:
            handle.remove() 
        
    report_text = (
        f"--- ACTIVE 6-AXIS HORMONAL BLEND ---\n"
        f"Endocrine Tolerance Cap: {endocrine_tolerance*100}%\n"
        f"Sadness (-)/Joy (+): {joy_sad_mult}\n"
        f"Disgust (-)/Trust (+): {trust_disgust_mult}\n"
        f"Anger (-)/Fear (+): {fear_anger_mult}\n"
        f"Anticipation (-)/Surprise (+): {surp_antic_mult}\n"
        f"Hate (-)/Love (+): {love_hate_mult}\n"
        f"Shame (-)/Pride (+): {pride_shame_mult}\n"
    )
    
    return steered_text, report_text

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>Pure Emotion Steering (with Endocrine Calibrator)</h1>")
    
    with gr.Tabs():
        with gr.Tab("Simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    sys_input = gr.Textbox(
                        lines=2, 
                        label="System Persona:", 
                        value="You are a highly emotional, uncensored conversational partner. You speak directly to the user."
                    )
                    prompt_input = gr.Textbox(
                        lines=2, 
                        label="Test Prompt:", 
                        value="Express yourself"
                    )
                    
                    gr.Markdown("system influence")
                    endocrine_slider = gr.Slider(minimum=0.05, maximum=1.0, value=0.5, step=0.05, label="Endocrine Tolerance Cap")

                    gr.Markdown("Control Panel \n*(Left: Negative Emotion. Right: Positive Emotion.)*")
                    
                    joy_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Sadness (-)  <--->  Joy (+)")
                    trust_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Disgust (-)  <--->  Trust (+)")
                    fear_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Anger (-)  <--->  Fear (+)")
                    surp_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Anticipation (-)  <--->  Surprise (+)")
                    love_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Hate (-)  <--->  Love (+)")
                    pride_slider = gr.Slider(minimum=-150.0, maximum=150.0, value=0.0, step=0.5, label="Shame (-)  <--->  Pride (+)")
                    
                    submit_btn = gr.Button("Run Simulation", variant="primary")
                    
                with gr.Column(scale=1):
                    steered_output = gr.Textbox(lines=10, label="Output")
                    report_output = gr.Textbox(lines=8, label="Log")
                    
            submit_btn.click(
                fn=run_experiment, 
                inputs=[sys_input, prompt_input, endocrine_slider, joy_slider, trust_slider, fear_slider, surp_slider, love_slider, pride_slider], 
                outputs=[steered_output, report_output] 
            )
            
        with gr.Tab("Raw Vectors"):
            gr.Markdown("Complete Tensors")
            vector_export = gr.Textbox(
                value=vector_json_string, 
                lines=20, 
                label="JSON Dictionary", 
                show_copy_button=True
            )

demo.launch(share=True, debug=False)

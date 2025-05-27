import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer & model from local path
LOCAL_MODEL_PATH = "models/mixtral"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype="auto"
)

# Generate full structured lyrics
def generate_lyrics(prompt_theme, genre, mood):
    prompt = f"""<s>[INST] Based on the theme: "{prompt_theme}", write a full English {genre} song with a {mood} mood.
Structure it with:
- Verse 1
- Chorus
- Verse 2
- Chorus
- Bridge (optional)
- Final Chorus

Make the lyrics poetic, emotionally engaging, and musical. Use rhyme and rhythm where possible. [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=412,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True
        )

    full_output = tokenizer.decode(output[0], skip_special_tokens=True)

    #  Extract only the lyrics part
    if "[/INST]" in full_output:
        lyrics = full_output.split("[/INST]")[-1].strip()
    else:
        lyrics = full_output.strip()

    return lyrics

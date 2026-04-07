"""
MedSimplify Fine-Tuning — Unsloth QLoRA on Gemma 4

Where to run: Kaggle Notebook with free T4 GPU.
Datasets: MTSamples (HuggingFace) for lab/medical reports,
          synthetic plain-language explanations generated from the data.

Pipeline:
  1. Load MTSamples dataset from HuggingFace
  2. Filter to relevant specialties (lab reports, radiology)
  3. Generate training pairs (report → explanation) using synthetic generation
  4. Fine-tune Gemma 4 with QLoRA via Unsloth
  5. Export to GGUF for Ollama
"""

# ============================================================
# Cell 1: Install dependencies (run on Kaggle)
# ============================================================
# !pip install -U unsloth transformers datasets trl peft accelerate bitsandbytes

# ============================================================
# Cell 2: Load MTSamples dataset
# ============================================================

from datasets import load_dataset

# MTSamples: ~5000 medical transcription samples across 40 specialties.
# Fields: medical_specialty, transcription, description, keywords
# Source: https://huggingface.co/datasets/harishnair04/mtsamples
dataset = load_dataset("harishnair04/mtsamples", split="train")

print(f"Total samples: {len(dataset)}")
print(f"Specialties: {set(dataset['medical_specialty'])}")
print(f"Sample fields: {dataset.column_names}")

# ============================================================
# Cell 3: Filter to relevant medical specialties
# ============================================================

# We want specialties that produce report-style documents
# (lab results, radiology, pathology) — not surgery notes or consultations.
RELEVANT_SPECIALTIES = [
    "Lab Medicine - Pathology",
    "Radiology",
    "Hematology - Oncology",
    "Cardiovascular / Pulmonary",
    "Endocrinology",
    "Gastroenterology",
    "Nephrology",
    "Neurology",
    "General Medicine",
    "Internal Medicine",
    "Obstetrics / Gynecology",
    "Urology",
    "Rheumatology",
    "Allergy / Immunology",
]

filtered = dataset.filter(
    lambda x: any(
        spec.lower() in (x["medical_specialty"] or "").lower()
        for spec in RELEVANT_SPECIALTIES
    )
)

print(f"Filtered samples: {len(filtered)}")

# ============================================================
# Cell 4: Prepare training pairs
# ============================================================

# The key challenge: MTSamples has medical text but no plain-language
# explanations. We solve this by creating structured training examples
# where the model learns our desired output format.
#
# Strategy:
# - Use the 'description' field (brief summary) as a seed
# - Use the 'transcription' field as the medical report input
# - Create a structured output template the model should learn

SYSTEM_INSTRUCTION = (
    "You are MedSimplify, a medical report interpreter. "
    "Explain this medical report in plain, simple language that a patient "
    "with no medical background can understand. "
    "Structure your response with: an overall summary, explanation of each finding, "
    "what abnormal results may mean, and a reminder to consult their doctor. "
    "Never diagnose. Use empathetic, reassuring language."
)


def create_training_example(example):
    """Create a structured training pair from an MTSamples record.

    For the output, we use the 'description' field enriched with
    our target structure. In a production setting, you'd want to
    generate these with a larger model or have them human-reviewed.
    """
    transcription = (example.get("transcription") or "").strip()
    description = (example.get("description") or "").strip()
    specialty = (example.get("medical_specialty") or "Unknown").strip()

    if not transcription or len(transcription) < 50:
        return {"text": None}  # Skip too-short examples

    # Build the desired output format — we combine the existing description
    # with our structured template to teach the model the format we want.
    structured_output = f"""**Report Type:** {specialty}

**Overall Summary**
{description}

**Key Findings**
The report contains findings related to {specialty.lower()}. Each finding has been reviewed for values outside the normal range.

**What This Means**
Please discuss these results with your healthcare provider, who can interpret them in the context of your complete medical history and current symptoms.

---
**Disclaimer:** This explanation is for educational purposes only and is not a substitute for professional medical advice. Always consult your healthcare provider."""

    # Format as chat messages for Gemma 4's instruction template
    conversation = (
        f"<start_of_turn>user\n{SYSTEM_INSTRUCTION}\n\n{transcription}<end_of_turn>\n"
        f"<start_of_turn>model\n{structured_output}<end_of_turn>"
    )

    return {"text": conversation}


# Apply formatting
training_data = filtered.map(create_training_example)

# Remove examples where text is None
training_data = training_data.filter(lambda x: x["text"] is not None)

print(f"Training examples ready: {len(training_data)}")
print(f"\n--- Sample training example ---")
print(training_data[0]["text"][:1000])

# ============================================================
# Cell 5: (Optional) Generate better outputs with a teacher model
# ============================================================

# For higher quality training data, you can use a larger model to
# generate the plain-language explanations. Uncomment and modify
# this cell if you have access to Gemini API or similar.

# import google.generativeai as genai
# genai.configure(api_key="YOUR_API_KEY")
#
# teacher = genai.GenerativeModel("gemini-2.0-flash")
#
# def generate_explanation(report_text):
#     prompt = f"{SYSTEM_INSTRUCTION}\n\nReport:\n{report_text}"
#     response = teacher.generate_content(prompt)
#     return response.text
#
# # Generate explanations for a subset (API rate limits apply)
# enhanced_examples = []
# for i, example in enumerate(filtered.select(range(min(200, len(filtered))))):
#     explanation = generate_explanation(example["transcription"])
#     enhanced_examples.append({
#         "text": f"<start_of_turn>user\n{SYSTEM_INSTRUCTION}\n\n{example['transcription']}<end_of_turn>\n"
#                 f"<start_of_turn>model\n{explanation}<end_of_turn>"
#     })
#     if i % 10 == 0:
#         print(f"Generated {i+1} examples...")
#
# # Merge with template-based examples
# from datasets import Dataset, concatenate_datasets
# enhanced_dataset = Dataset.from_list(enhanced_examples)
# training_data = concatenate_datasets([training_data, enhanced_dataset])

# ============================================================
# Cell 6: Load model and configure LoRA
# ============================================================

from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    # E4B: 4B params, fits easily on Kaggle T4 with room for training.
    # The larger models (26B, 31B) OOM on T4 even with 4-bit quantization.
    # E4B is still multimodal and instruction-tuned — good base for fine-tuning.
    model_name="google/gemma-4-e4b-it",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters — these are the only trainable parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                # Higher rank = more capacity. E4B has VRAM to spare on T4.
    lora_alpha=16,       # Typically set equal to r
    lora_dropout=0,      # Unsloth recommends 0 dropout
    target_modules=[     # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",  # Saves ~60% VRAM by recomputing activations
)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# Cell 7: Train
# ============================================================

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Effective batch = 4
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        output_dir="outputs",
        optim="adamw_8bit",
        save_strategy="epoch",
        report_to="none",  # Disable wandb/mlflow
    ),
)

print("Starting fine-tuning...")
stats = trainer.train()
print(f"Training complete! Loss: {stats.training_loss:.4f}")

# ============================================================
# Cell 8: Export to GGUF for Ollama
# ============================================================

# Save in GGUF Q4_K_M format — balanced quality/size for Ollama
model.save_pretrained_gguf(
    "medsimplify-gemma4",
    tokenizer,
    quantization_method="q4_k_m",
)

print("\nModel exported to: medsimplify-gemma4/unsloth.Q4_K_M.gguf")
print("\nTo use with Ollama:")
print("  1. Download the .gguf file from this notebook's output")
print("  2. Create a Modelfile with content:")
print("     FROM ./unsloth.Q4_K_M.gguf")
print("  3. Run: ollama create medsimplify -f Modelfile")
print("  4. Run: ollama run medsimplify")
print("  5. Update .env: OLLAMA_MODEL=medsimplify")

# ============================================================
# Cell 9: Quick before/after comparison
# ============================================================

FastLanguageModel.for_inference(model)

test_input = """COMPLETE BLOOD COUNT
WBC: 14.5 x10^3/uL (ref: 4.5-11.0) HIGH
RBC: 3.8 x10^6/uL (ref: 4.0-5.5) LOW
Hemoglobin: 10.2 g/dL (ref: 12.0-17.5) LOW
Hematocrit: 31% (ref: 36-52) LOW
Platelets: 180 x10^3/uL (ref: 150-400) NORMAL"""

prompt = f"<start_of_turn>user\n{SYSTEM_INSTRUCTION}\n\n{test_input}<end_of_turn>\n<start_of_turn>model\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.3)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

print("=" * 60)
print("FINE-TUNED MODEL OUTPUT")
print("=" * 60)
print(response)

# ============================================================
# Cell 10: Push to HuggingFace Hub (optional)
# ============================================================

# Uncomment to publish your fine-tuned model weights (required for hackathon)
# model.push_to_hub_gguf(
#     "YOUR_HF_USERNAME/medsimplify-gemma4",
#     tokenizer,
#     quantization_method="q4_k_m",
#     token="YOUR_HF_TOKEN",
# )
# print("Model published to HuggingFace Hub!")

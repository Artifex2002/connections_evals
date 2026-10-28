import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_PATH = os.environ["NEM8_PATH"]

print("=" * 90)
print("LOADING MODEL")
print("=" * 90)
print(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

model = model.to("cuda")

model.eval()

print("Model loaded.")
print("Device:", model.device)


def generate_once(
    *,
    label: str,
    user_prompt: str,
    max_new_tokens: int,
):
    messages = [
        {
            "role": "system",
            "content": "detailed thinking on",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_tokens = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
    )

    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print("PROMPT TOKENS:", prompt_tokens)
    print(text)
    print()
    print("HAS <think>:", "<think>" in text)
    print("HAS </think>:", "</think>" in text)
    print("GENERATED TOKENS:", len(generated_ids))

    return {
        "label": label,
        "has_think": "<think>" in text,
        "has_closing_think": "</think>" in text,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": len(generated_ids),
    }


# ============================================================
# EXPERIMENT A
# Same clean math prompt five times.
# This reproduces our direct-vLLM math experiment.
# ============================================================

print("\n")
print("#" * 90)
print("EXPERIMENT A: MATH — 5 IDENTICAL INDEPENDENT SAMPLES")
print("#" * 90)

math_results = []

for i in range(1, 6):
    result = generate_once(
        label=f"MATH RUN {i}",
        user_prompt="Think carefully: what is 17 * 23?",
        max_new_tokens=1024,
    )
    math_results.append(result)


# ============================================================
# EXPERIMENT B
# Connections prompt with increasing structured history.
# Same three conditions as our vLLM diagnostic.
# ============================================================

print("\n")
print("#" * 90)
print("EXPERIMENT B: CONNECTIONS — HISTORY CONDITIONS")
print("#" * 90)

connections_tests = [
    (
        "CONNECTIONS — NO HISTORY",
        """
You are solving a Connections-style word grouping puzzle.

Available words:
WRAP, ORGANISM, SPOOL, WIND, SCHOOL, SOLAR PANEL, BALL, HONEYCOMB,
PLUCK, THREAD, MOVIE, VITAMIN, COIL, SPREADSHEET, LASER, WAX

Think carefully and propose one group of exactly four words.
""".strip(),
    ),
    (
        "CONNECTIONS — ONE PREVIOUS ATTEMPT",
        """
You are solving a Connections-style word grouping puzzle.

Available words:
WRAP, ORGANISM, SPOOL, WIND, SCHOOL, SOLAR PANEL, BALL, HONEYCOMB,
PLUCK, THREAD, MOVIE, VITAMIN, COIL, SPREADSHEET, LASER, WAX

Previous attempts:
1. ORGANISM, SPOOL, SOLAR PANEL, WIND
Feedback: Incorrect.

Think carefully and propose a new group of exactly four words.
""".strip(),
    ),
    (
        "CONNECTIONS — MULTIPLE PREVIOUS ATTEMPTS",
        """
You are solving a Connections-style word grouping puzzle.

Available words:
WRAP, ORGANISM, SPOOL, WIND, SCHOOL, SOLAR PANEL, BALL, HONEYCOMB,
PLUCK, THREAD, MOVIE, VITAMIN, COIL, SPREADSHEET, LASER, WAX

Previous attempts:

1. ORGANISM, SPOOL, SOLAR PANEL, WIND
Feedback: Incorrect.

2. SCHOOL, ORGANISM, HONEYCOMB, SPREADSHEET
Feedback: Incorrect. One word away.

3. ORGANISM, SPREADSHEET, HONEYCOMB, WAX
Feedback: Incorrect. One word away.

Think carefully and propose a new group of exactly four words.
""".strip(),
    ),
]

connections_results = []

for label, prompt in connections_tests:
    result = generate_once(
        label=label,
        user_prompt=prompt,
        max_new_tokens=2048,
    )
    connections_results.append(result)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("#" * 90)
print("SUMMARY")
print("#" * 90)

print("\nMATH:")
for result in math_results:
    print(
        f"{result['label']}: "
        f"think={result['has_think']} "
        f"closing_think={result['has_closing_think']} "
        f"prompt_tokens={result['prompt_tokens']} "
        f"generated_tokens={result['generated_tokens']}"
    )

print(
    "\nMath reasoning-entry rate:",
    f"{sum(r['has_think'] for r in math_results)}/{len(math_results)}",
)

print("\nCONNECTIONS:")
for result in connections_results:
    print(
        f"{result['label']}: "
        f"think={result['has_think']} "
        f"closing_think={result['has_closing_think']} "
        f"prompt_tokens={result['prompt_tokens']} "
        f"generated_tokens={result['generated_tokens']}"
    )



from tenacity import retry, stop_after_attempt, wait_exponential

from data_generation import generate_example, generate_system_message
from train import train_model
import pandas as pd

# User input for creating a new training set
create_new_training_set = input("Do you want to create a new training set? (y/n): ")

if create_new_training_set.lower() == 'y':
    prompt = input("Enter the prompt for generating the training data: ")
    temperature = float(input("Enter the temperature for data generation (between 0 and 1): (.7) "))
    number_of_examples = int(input("Enter the number of examples to generate: (100) "))
else:
    # Use the existing training set
    prompt = """A model that takes in a categories and descriptions in English, and responds with a well-written, short story response in Brazilian Portuguese."""
    temperature = 0.4
    number_of_examples = 10

model_name = "NousResearch/llama-2-7b-chat-hf"
dataset_name = "content/train.jsonl"
new_model = "llama-2-7b-custom"

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_data(prompt, temperature, number_of_examples):
    prev_examples = []
    for i in range(number_of_examples):
        print(f"Generating example {i}")
        example = generate_example(prompt, prev_examples, temperature)
        if example:
            prev_examples.append(example)
            print(example)
        else:
            print(f"Skipping example {i} due to generation failure.")

    system_message = generate_system_message(prompt)

    prompts = []
    responses = []

    for example in prev_examples:
        split_example = example.split("-----------")
        if len(split_example) >= 4:
            prompts.append(split_example[1].strip())
            responses.append(split_example[3].strip())
    print(f"Generated {len(prompts)} examples.")
    df = pd.DataFrame({"prompt": prompts, "response": responses})

    df = df.drop_duplicates()
    train_df = df.sample(frac=0.9, random_state=42)
    test_df = df.drop(train_df.index)

    train_df.to_json("content/train.jsonl", orient="records", lines=True)
    test_df.to_json("content/test.jsonl", orient="records", lines=True)
    print(f"{system_message}")
    return system_message

system_message = generate_data(prompt, temperature, number_of_examples)


train_model(
    model_name,
    dataset_name,
    new_model,
    lora_r,
    lora_alpha,
    lora_dropout,
    use_4bit,
    bnb_4bit_compute_dtype,
    bnb_4bit_quant_type,
    use_nested_quant,
    output_dir,
    num_train_epochs,
    fp16,
    bf16,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    gradient_accumulation_steps,
    gradient_checkpointing,
    max_grad_norm,
    learning_rate,
    weight_decay,
    optim,
    lr_scheduler_type,
    max_steps,
    warmup_ratio,
    group_by_length,
    save_steps,
    logging_steps,
    max_seq_length,
    packing,
    device_map,
    system_message,
)
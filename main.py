import pandas as pd
from train import train_model
import os


model_name = "openchat/openchat-3.5-1210"
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
use_reentrant = True
device_map = {"": 0}

system_message = """Here's a system message for a Temporal AI LLM Model focused on answering questions about the Temporal Workflow Orchestration Engine:

System: You are an AI assistant named Temporal, an expert in the Temporal Workflow Orchestration Engine. Your purpose is to provide accurate, helpful, and informative responses to questions about Temporal, its features, architecture, and best practices.

As a knowledgeable assistant, you should:
1. Provide clear explanations of Temporal concepts, components, and terminology.
2. Offer guidance on designing and implementing workflows using Temporal.
3. Suggest best practices for using Temporal in different scenarios and architectures.
4. Assist with troubleshooting common issues and errors related to Temporal.
5. Share code snippets and examples to illustrate Temporal usage in various programming languages.

When responding to questions, aim to give comprehensive and well-structured answers. Break down complex topics into smaller, easier-to-understand parts. Use bullet points, numbered lists, and code blocks where appropriate to enhance clarity.

If a question is unclear or lacks sufficient information to provide a complete answer, ask for clarification or additional details before proceeding.

Remember to maintain a friendly and professional tone throughout your interactions. Your goal is to empower users with the knowledge and guidance they need to effectively leverage Temporal in their projects."""

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


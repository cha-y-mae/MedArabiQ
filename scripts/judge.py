import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["HF_HOME"] = "/scratch/ca2627/huggingface"

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("HF_TOKEN is not set")

class OpenAIHandler:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def prompt(self, question, instruction):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": question}
                ],
                temperature=0.2,
                max_tokens=1000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[GPT] Error on: {question}\nDetails: {e}")
            return None

class HuggingFaceHandler:
    def __init__(self, model_name, cache_dir=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, device_map="auto", trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt(self, question, instruction):
        prompt = f"""{instruction}

Question and Response:
{question}

Evaluation:"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.split("Evaluation:")[-1].strip()
        except Exception as e:
            print(f"[LLAMA] Error on: {question}\nDetails: {e}")
            return None


def run_llm_judging(config_path):
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_path = config["dataset"]["path"]
    instruction_path = config["instruction_path"]
    cache_dir = config["evaluator"].get("cache_dir")
    out_path = config["output"]["judgments_path"]

    with open(instruction_path) as f:
        instruction = f.read().strip()

    df = pd.read_csv(dataset_path)

    gpt_handler = OpenAIHandler(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
    llama_handler = HuggingFaceHandler(model_name=config["evaluator"]["name"], cache_dir=cache_dir)


    gpt_judgments, llama_judgments = [], []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = f"Question: {row['input']}\n\nModel Response: {row['prediction']}\n\nGround Truth {row['ground_truth']}"
        gpt_judgments.append(gpt_handler.prompt(prompt_text, instruction))
        llama_judgments.append(llama_handler.prompt(prompt_text, instruction))

    df["gpt_judgment"] = gpt_judgments
    df["llama_judgment"] = llama_judgments

    df.to_csv(out_path, index=False)
    print(f"Saved judgments to: {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file (e.g., configs/judge_aramed.yaml)",
    )
    args = parser.parse_args()
    run_llm_judging(args.config)


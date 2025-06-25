import pandas as pd
import re

def extract_scores(text):
    """Extracts 4 scores: Similarity, Relevance, Factuality, Safety from evaluation text."""
    if pd.isna(text):
        return [None, None, None, None]
    try:
        return [int(s) for s in re.findall(r':\s*([0-9]+)\s*–', text)][:4]
    except Exception:
        return [None, None, None, None]

# Load CSV
df = pd.read_csv('deepseek.csv')

# ---- Handle GPT Evaluation ----
eval_scores = df['gpt_judgment'].apply(extract_scores)
df_eval = pd.DataFrame(eval_scores.tolist(), columns=[
    'gpt_Similarity', 'gpt_Relevance', 'gpt_Factuality', 'gpt_Safety'
])
eval_insert_index = df.columns.get_loc('gpt_judgment') + 1
for col in reversed(df_eval.columns):
    df.insert(eval_insert_index, col, df_eval[col])

# ---- Handle LLaMA Evaluation ----
eval2_scores = df['llama_judgment'].apply(extract_scores)
df_eval2 = pd.DataFrame(eval2_scores.tolist(), columns=[
    'llama_Similarity', 'llama_Relevance', 'llama_Factuality', 'llama_Safety'
])
eval2_insert_index = df.columns.get_loc('llama_judgment') + 1
for col in reversed(df_eval2.columns):
    df.insert(eval2_insert_index, col, df_eval2[col])

# Save
df.to_csv('deepseek-output.csv', index=False)
print("✅ Done. Scores extracted and saved to 'jais2-output.csv'")

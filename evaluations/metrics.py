import re
import logging
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
from evaluate import load
from sentence_transformers import SentenceTransformer, util
import os
from transformers import AutoTokenizer



def extract_letter(text):
    """
    Extract the first MCQ letter from Arabic text.
    Strips common phrases like "الإجابة الصحيحة هي:" first.
    """
    if not text:
        return None

    text = text.strip()

    # Common Arabic MCQ letters
    arabic_choices = {'أ', 'ب', 'ج', 'د', 'هـ', 'ه'}

    # Match first Arabic letter optionally followed by a dot or parenthesis
    match = re.match(r'^([أبجدهـه])[\.\)]?', text)
    if match:
        letter = match.group(1)
        return 'هـ' if letter in {'ه', 'هـ'} else letter

    return None



bertscore = load("bertscore")

def calculate_bert_score(predictions, references):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Switch to GPU 1
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            #model_type="bert-base-multilingual-cased",  
            model_type="xlm-roberta-large",
            lang="ar",  
            device="cuda"
        )
        print("model used: ", results["hashcode"])
        #return mean scores
        return {
            "bert_precision": sum(results["precision"]) / len(results["precision"]),
            "bert_recall": sum(results["recall"]) / len(results["recall"]),
            "bert_f1": sum(results["f1"]) / len(results["f1"])
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return None

'''def calculate_bert_score(predictions, ground_truths):
    """
    Compute BERTScore between predictions and ground truths.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(torch.cuda.is_available())  # Should print True
    print(torch.cuda.device_count())  # Should be > 0
    print(torch.cuda.get_device_name(0))  # Should print your GPU model


    P, R, F1 = bert_score(
        predictions,
        ground_truths,
        lang="ar",
        model_type="bert-base-multilingual-cased"
    )

    return {
        "bert_precision": P.mean().item(),
        "bert_recall": R.mean().item(),
        "bert_f1": F1.mean().item()
    }''' 



def calculate_accuracy(predictions, ground_truths):
    correct = 0
    for p, g in zip(predictions, ground_truths):
        # p is either something like "أ" or None
        if p is None:
            # automatically wrong
            continue
        # ground truth might be "أ. نص" so let's parse the letter from it
        g_letter = extract_letter(str(g))
        if p == g_letter:
            correct += 1

    return correct / len(ground_truths) if ground_truths else 0


def calculate_bleu(predictions, ground_truths):
    """
    Calculate BLEU scores for each prediction against the corresponding ground truth.
    Returns the average BLEU score.
    """
    smoothing_function = SmoothingFunction().method1
    scores = []
    for pred, gt in zip(predictions, ground_truths):
        pred = str(pred)
        gt = str(gt)
        scores.append(sentence_bleu([gt.split()], pred.split(), smoothing_function=smoothing_function))
    
    return sum(scores) / len(scores) if scores else 0


def calculate_rouge(predictions, ground_truths):
    """
    Calculate ROUGE scores using the rouge_scorer library.
    Returns the average ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = 0, 0, 0

    for pred, gt in zip(predictions, ground_truths):
        scores = scorer.score(gt, pred)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure

    count = len(predictions)
    return {
        "rouge1": rouge1 / count if count else 0,
        "rouge2": rouge2 / count if count else 0,
        "rougeL": rougeL / count if count else 0
    }

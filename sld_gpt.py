#!/usr/bin/env python
# coding: utf-8
"""
Social-Language-Disorder (SLD) zero-shot
"""

import os, json, time, re, openai
from tqdm import tqdm
import pandas as pd

# ------------------------- OpenAI account configuration -------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"     

# ------------------------- 10 feature names -------------------------------------------------
CATEGORIES = [
    "Echoic Repetition",
    "Unconventional Content",
    "Pronoun Displacement",
    "Incongruous Humor Timing",
    "Formalistic Language Use",
    "Superfluous Phrase Attachment",
    "Excessive Social Phrasing",
    "Monotone Social Expression",
    "Stereotyped Media Quoting",
    "Clichéd Verbal Substitutions",
]

CAT_LOWER = [c.lower() for c in CATEGORIES]          

# ------------------------- Prompt generation function -------------------------------------
def coarse_prompt(dialogue: str) -> str:
    """使用原始 Coarse Prompt，回答仅 Yes / No"""
    return (
        f"{dialogue}\n"
        "Based on the above conversation between the examiner and the patient, "
        "please categorize if any observed Social Language Disorders for the patient. "
        "Answer only 'Yes' or 'No'."
    )

def fine_prompt(dialogue: str) -> str:

    return f"""{dialogue}
Based on the above conversation between the examiner and the patient, please categorize any observed social language disorders for the patient into the provided 10 language categories and demonstrate all instances of disorder evidence present in their dialogue.
1. Echoic Repetition: The individual mimics verbatim what has been said by others, including the examiner, or recites phrases from external sources like advertisements or movie scripts, showing a delayed echo response;
2. Unconventional Content: The speech contains peculiarly chosen content or contextually odd phrasing, such as using 'unfreshness through household' for lack of novelty, 'mideast' instead of 'midwest' for U.S. states, or describing entry into a building as 'through various apertures';
3. Pronoun Displacement: Incorrectly substitutes personal pronouns, using 'you' in place of 'I', or refers to themselves in the third person, either by pronouns like 'he/she' or by their own name;
4. Incongruous Humor Timing: Incorporates humorous or comedic expressions inappropriately during discussions meant to be serious, showing a misalignment between the content's emotional tone and the context;
5. Formalistic Language Use: Employs an overly formal or archaic language style that seems lifted from written texts, legal documents, or old literature, rather than engaging in conversational speech. Examples include elaborate ways of expressing simple ideas or feelings;
6. Superfluous Phrase Attachment: Attaches redundant phrases or filler expressions to their speech without contributing any substantive meaning or context, such as 'you know what I mean' or 'as they say,' indicating a habit rather than intentional emphasis;
7. Excessive Social Phrasing: Utilizes conventional social expressions excessively or inappropriately, responding with phrases like 'oh, thank you' in contexts where it does not fit or preempting social gestures not yet performed by the interlocutor;
8. Monotone Social Expression: Reiterates social phrases with an unchanged, monotonous intonation, indicating a lack of genuine emotional engagement or variability in social interactions;
9. Stereotyped Media Quoting: Quotes lines from commercials, movies, or TV shows in a highly stereotypical manner, employing a canned intonation that mimics the original source closely, suggesting a reliance on external media for verbal expressions;
10. Clichéd Verbal Substitutions: Resorts to well-known sayings or clichés in lieu of engaging in direct conversational responses, using phrases like 'circle of life' or 'ready to roll' as stand-ins for more personalized communication."""
# --------------------------------------------------------------------------------

def chat(prompt: str, retries: int = 3, backoff: int = 2) -> str:

    for attempt in range(retries):
        try:
            resp = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return resp.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2

def detect_features(resp: str) -> list[int]:

    lower = resp.lower()
    flags = [1 if name in lower else 0 for name in CAT_LOWER]
    return flags

# ------------------------- main workflow ----------------------------------------------
DATA_PATH = "caltech_A4.json"
with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

rows = []

for ex in tqdm(data, desc="Processing"):
    uid, text, label = ex["uid"], ex["text"], ex["label"]

    # ---------- Coarse  ----------
    coarse_resp = chat(coarse_prompt(text))
    has_sld = coarse_resp.lower().startswith("y")   # Yes/No

    features = [0]*10
    if has_sld:
        fine_resp = chat(fine_prompt(text))
        features = detect_features(fine_resp)

    # ---------- record ----------
    row = {"uid": uid, **{f"feature_{i}": features[i-1] for i in range(1, 11)}, "label": label}
    rows.append(row)

# ------------------------- save Excel ------------------------------------------
df = pd.DataFrame(rows)
df.to_excel("sld_predictions.xlsx", index=False)
print("Saved → sld_predictions.xlsx")
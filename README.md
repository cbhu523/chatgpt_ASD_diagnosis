# Exploiting ChatGPT for Diagnosing Autism-Associated Language Disorders and Identifying Distinct Features

This repository contains the datasets, prompts, and supplementary materials for the paper:  
**"Exploiting ChatGPT for Diagnosing Autism-Associated Language Disorders and Identifying Distinct Features"**  
by **Chuanbo Hu, Wenqi Li, Mindi Ruan, Xiangxu Yu, Lynn K. Paul, Shuo Wang, and Xin Li**.

---

## Abstract

Diagnosing language disorders associated with autism is a complex challenge, often hampered by the subjective nature and variability of traditional assessment methods. Traditional diagnostic methods not only require intensive human effort but also often result in delayed interventions due to their lack of speed and precision. In this study, we explored the application of ChatGPT, a state-of-the-art large language model, to overcome these obstacles by enhancing sensitivity and profiling linguistic features for autism diagnosis. This research utilizes ChatGPT's natural language processing capabilities to simplify and improve the diagnostic process, focusing on identifying autism-related language patterns. Specifically, we compared ChatGPT's performance with that of conventional supervised learning models, including BERT, a model acclaimed for its effectiveness in various natural language processing tasks. We showed that ChatGPT substantially outperformed these models, achieving over 10% improvement in both sensitivity and positive predictive value, in a zero-shot learning configuration. The findings underscore the model’s potential as a superior diagnostic tool, combining accuracy and applicability. We identified ten key features of autism-associated language disorders across scenarios. Features such as echolalia, pronoun reversal, and atypical language usage play a critical role in diagnosing ASD and informing tailored treatment plans. Together, our findings advocate for adopting sophisticated AI tools like ChatGPT in clinical settings to assess and diagnose developmental disorders. Our approach promises enhanced diagnostic precision and supports personalized medicine, potentially transforming the evaluation landscape for autism and similar neurological conditions.

---

## Framework

![Framework for Diagnosing Autism and Identifying Language Disorders](./images/framework.jpg)
---

## Dataset

The sample dataset is available for download at the following link:  
[Download Dataset](https://drive.google.com/file/d/194Oh-6jr-8Wbb9qUCt2HieMeUV8Pwwuu/view?usp=sharing)  

Additional data may be added in the future.  

---
## Prompt (Coarse: if existing SLD)
```text
Dialogue['text'] + "Based on the above conversation between the examiner and the patient, please categorize if any observed Social Language Disorders for the patient. Answer only 'Yes' or 'No'.“
```

---
## Prompt (Fine: which kind of SLD Features)

```text
Dialogue['text'] + "Based on the above conversation between the examiner and the patient, please categorize any observed social language disorders for the patient into the provided 10 language categories and demonstrate all instances of disorder evidence present in their dialogue.

1. Echoic Repetition: The individual mimics verbatim what has been said by others, including the examiner, or recites phrases from external sources like advertisements or movie scripts, showing a delayed echo response;
2. Unconventional Content: The speech contains peculiarly chosen content or contextually odd phrasing, such as using 'unfreshness through household' for lack of novelty, 'mideast' instead of 'midwest' for U.S. states, or describing entry into a building as 'through various apertures';
3. Pronoun Displacement: Incorrectly substitutes personal pronouns, using 'you' in place of 'I', or refers to themselves in the third person, either by pronouns like 'he/she' or by their own name;
4. Incongruous Humor Timing: Incorporates humorous or comedic expressions inappropriately during discussions meant to be serious, showing a misalignment between the content's emotional tone and the context;
5. Formalistic Language Use: Employs an overly formal or archaic language style that seems lifted from written texts, legal documents, or old literature, rather than engaging in conversational speech. Examples include elaborate ways of expressing simple ideas or feelings;
6. Superfluous Phrase Attachment: Attaches redundant phrases or filler expressions to their speech without contributing any substantive meaning or context, such as 'you know what I mean' or 'as they say,' indicating a habit rather than intentional emphasis;
7. Excessive Social Phrasing: Utilizes conventional social expressions excessively or inappropriately, responding with phrases like 'oh, thank you' in contexts where it does not fit or preempting social gestures not yet performed by the interlocutor;
8. Monotone Social Expression: Reiterates social phrases with an unchanged, monotonous intonation, indicating a lack of genuine emotional engagement or variability in social interactions;
9. Stereotyped Media Quoting: Quotes lines from commercials, movies, or TV shows in a highly stereotypical manner, employing a canned intonation that mimics the original source closely, suggesting a reliance on external media for verbal expressions;
10. Clichéd Verbal Substitutions: Resorts to well-known sayings or clichés in lieu of engaging in direct conversational responses, using phrases like 'circle of life' or 'ready to roll' as stand-ins for more personalized communication.
```
## Case Study

![Case Study Analysis: Identifying Language Deficits in an Examiner-Patient Dialogue. Phrases highlighted in indicate observed linguistic anomalies, while underscores the specific feature category of language deficits.](./images/case_study.png)
---

## Citation
If you use our data or prompts in your research, please cite our paper:
```text
@article{hu2024exploiting,
  title={Exploiting ChatGPT for Diagnosing Autism-Associated Language Disorders and Identifying Distinct Features},
  author={Hu, Chuanbo and Li, Wenqi and Ruan, Mindi and Yu, Xiangxu and Paul, Lynn K and Wang, Shuo and Li, Xin},
  journal={arXiv preprint arXiv:2405.01799},
  year={2024}
}
```

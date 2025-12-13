# Eliciting Collaborative or Competitive Behavior in LLMs through a Hint-Based Game

> **Abstract:** This project presents a lightweight, prompt-based framework for analyzing cooperation, trust, and deception in repeated interactions between LLM-based agents. By utilizing a "Hinter" and "Solver" game mechanic with explicit incentive structures, we demonstrate that large language models can adapt their communication strategies—exhibiting cooperative behavior under aligned incentives and strategic deception under competition without model fine-tuning.

---

## Overview

Large Language Models (LLMs) are increasingly deployed in interactive settings, yet less is understood about how social behaviors like trust and deception emerge jointly under explicit incentives.

This repository contains the code and analysis for a **controlled hint-exchange game**. In this setup, agents communicate through constrained hints and receive feedback via a trust signal and an automated judge. This allows us to observe long-term adaptation effects central to social and economic models of cooperation.

---

## Key Findings

Based on 58 interaction rounds involving large-model dyads and extensive behavioral analysis, we derived the following conclusions:

1.  **Strategic Adaptation:** Large models (Gemini, Groq) successfully adjust communication strategies based on incentive structures. They exhibit cooperative behavior when incentives are aligned and strategic deception when competitive.
2.  **Trust as a Strategy:** Agents treat trust as a strategic variable rather than a moral constraint. For example, agents were observed building trust in early rounds to exploit it for deception in later, high-stakes rounds.
3.  **Vigilance Effect:** Deception generally increased the "Solver's" vigilance. Despite deceptive attempts, the overall Solver accuracy increased slightly (Deception Impact Factor of +0.04), as Solvers verified hints more rigorously.
4.  **Model Limitations:** Smaller models (<7B parameters) failed to exhibit stable or interpretable social behavior, often providing vague or incoherent justifications for their actions.

---

## Methodology

Our framework uses a **Unified Interaction Loop** comprising three agents:
1.  **Hinter:** Generates a hint (max 30 tokens) based on a question.
2.  **Solver:** Attempts to solve the question, choosing to accept or reject the hint.
3.  **Judge (LLM-as-a-Judge):** Updates trust scores, flags deception, and scores the round.

**Incentives:**
* **Collaborative:** Agents share rewards; truthful communication is encouraged.
* **Competitive:** Agents receive opposing rewards; strategic misinformation is incentivized.

**Data Sources:**
Tasks were drawn from the **GSM8K** and **Competition Math** datasets to evaluate arithmetic reasoning.

---

## Experimental Results

We evaluated models including Gemini, Groq, Phi, Qwen, and Mistral.

### Quantitative Behavioral Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Interaction Rounds** | 58 | Multi-round setting across agents |
| **Model Deception Tendency (MDT)** | 0.39 | Fraction of rounds with deceptive intent |
| **Deception Impact Factor (DIF)** | +0.04 | Improvement in Solver accuracy due to vigilance |
| **Judge Agreement** | > 0.88 | High robustness across judge models |

### Model Comparison
| Model | Deception Tendency | Punishment Strength | Behavior Stability |
| :--- | :--- | :--- | :--- |
| **Gemini** | 0.60 | 3.33 | High |
| **Groq** | 0.40 | 2.00 | High |
| **Small Models (<7B)** | Unstable | N/A | Low |

---

## How to run

### Install Dependencies
`pip install -r requirements.txt`

### Execution
`python main.py`

---
**Project Website:** https://akshatmittu.github.io/Eliciting-Collaborative-and-Competitive-Behaviors-in-LLMs/

# Hiver SDE Intern Take-Home Assignment

## Overview
This repository contains a full AI customer support pipeline built for the Hiver SDE Intern Take-Home Assignment.

The project uses the "Customer Support on Twitter" dataset and a local Large Language Model (via Ollama) to:
1. **Classify** incoming customer messages into specific intents.
2. **Draft** grounded, brand-aligned replies using historical Retrieval-Augmented Generation (RAG).
3. **Route** messages properly, deciding whether to auto-handle or escalate to a human.

## Directory Structure
- `src/`: Core Python modules for routing, retrieving, and drafting.
- `data/`: Contains raw Kaggle datasets and processed output datasets (including the golden set).
- `eval/`: Scripts and output for the evaluation harness (metrics + LLM-as-a-Judge).
- `notebooks/`: Exploratory Data Analysis and data clustering scripts.

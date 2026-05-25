# TinyDisasterVQA

TinyDisasterVQA is a small-scale disaster-focused Visual Question Answering project.

The goal is to train compact VQA models that can answer simple questions about disaster-scene images, with the long-term target of deploying a small student model on edge hardware such as GAP9.

## Project idea

Given an image and a simple question, the model predicts an answer.

Example questions:

- What is in the image?
- Is there flooding?
- What type of area is shown?
- Are buildings visible?
- Is the road flooded?

The project focuses on disaster-related datasets such as FloodNet and RescueNet.

## Current focus

The current focus is:

1. Explore the FloodNet dataset
2. Clean and understand the image/question/answer structure
3. Build a simple training pipeline
4. Train a teacher model
5. Train smaller student models
6. Evaluate model size, accuracy, and deployment feasibility

## Repository structure

```text
TinyDisasterVQA/
├── data/              # Dataset metadata, annotations, splits
├── images/            # Dataset images
├── notebooks/         # Dataset exploration and experiments
├── scripts/           # Training, evaluation, preprocessing scripts
├── src/               # Main project code
├── runs/              # Training outputs and checkpoints
└── README.md
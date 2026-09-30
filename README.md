# Privacy-Preserving Federated Learning with Agentic AI

A privacy-preserving federated learning system that combines federated learning with privacy, security, blockchain, and agentic AI components to support secure collaborative machine learning.

## 🚀 Overview

Traditional machine learning often requires collecting data from multiple organizations or users in a centralized location. This can create privacy and security concerns, especially when the data is sensitive.

This project explores a decentralized approach using **Federated Learning**, where participating clients can train models without directly sharing their raw data.

The system further integrates privacy-preserving and intelligent components to improve the security and usability of the federated learning workflow.

## 🎯 Key Features

- Federated Learning for collaborative model training
- Privacy-preserving mechanisms
- Differential Privacy
- Blockchain-based components
- Zero-Knowledge Proof (ZKP) components
- Agentic AI components
- LLM integration
- Retrieval-Augmented Generation (RAG)
- FastAPI backend
- Machine Learning models
- Model storage and management

## 🏗️ Project Structure

```text
├── agents/          # Agentic AI components
├── api/             # FastAPI backend and API components
├── blockchain/      # Blockchain-related components
├── federated/       # Federated learning components
├── models/          # Machine learning models
├── notebook/        # Experiments and notebooks
├── privacy/         # Privacy-preserving components
├── saved_models/    # Saved trained models
├── .gitignore
└── clean_output.txt
## 🏗️ System Architecture

The system integrates federated learning, vertical federated learning, privacy and security mechanisms, agentic AI, blockchain, and LLM-based explainability.

![Federated Fraud Detection System Architecture](architecture.png)

The architecture consists of six major modules:

1. **Federated Learning Module** – Enables multiple banks to collaboratively train models while keeping local data within each organization.

2. **Vertical Federated Learning Module** – Enables feature-level collaboration between different entities such as banks, UPI, and telecom providers without directly sharing raw data.

3. **Privacy & Security Module** – Incorporates Differential Privacy, Zero-Knowledge Proofs, and Post-Quantum Cryptography components.

4. **Agentic AI Module** – Includes agents for client selection, security monitoring, and privacy control.

5. **Blockchain Module** – Provides a permissioned blockchain layer for maintaining training-round records, client scores, anomaly logs, incidents, and model updates.

6. **Explainability Module** – Uses LLM and RAG components to generate human-readable explanations for fraud predictions.

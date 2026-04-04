# IntelliCloud Documentation Index

Welcome to the IntelliCloud project documentation. This index provides a roadmap for developers, researchers, and system administrators to understand how the intelligent task orchestration pipeline works.

## 🧱 Core Methodology (SHERA)
The **SHAP-Enhanced Resource Allocation (SHERA)** methodology is the heart of IntelliCloud. It follows a 6-step intelligent pipeline:

1.  [**About Datasets**](DATASETS.md) — Understanding the 10,000+ task profiles used for training.
2.  [**Feature Extraction**](FEATURE_EXTRACTION.md) — How raw tasks are scanned for telemetry and estimated costs.
3.  [**Autoencoder**](AUTOENCODER.md) — Compressing 8 high-fidelity features into 4 latent dimensions.
4.  [**Random Forest**](RANDOM_FOREST.md) — Predicting energy efficiency class (1-5) using 12 features.
5.  [**SHAP Explanations**](SHAP.md) — Generating human-readable visual reports of AI decisions.
6.  [**DQN Scheduler**](DQN.md) — How the RL agent selects optimal container tiers with real-time rewards.

## 🌐 System Architecture & Networking
7.  [**Frontend APIs**](FRONTEND_APIS.md) — Detailed reference for the Flask and SocketIO backend endpoints.
8.  [**WiFi Networking**](WIFI_NETWORKING.md) — Guide for connecting multiple laptops as worker nodes via WiFi.
9.  [**Task Sharing**](TASK_SHARING.md) — Explains the payload dispatch and result synchronization between nodes.
10. [**Docker Task Processing**](DOCKER_PROCESSING.md) — How isolated containers process tasks with resource limits.

## 🚀 Getting Started
-   **Master Node**: Start the main application using `api/app.py`.
-   **Worker Node**: Start the distributed server using `distributed_node/worker_server.py`.
-   **Dashboard**: Access the React frontend at `http://localhost:5173`.

---
*Follow these documents to understand, debug, or extend the IntelliCloud intelligent scheduling platform.*

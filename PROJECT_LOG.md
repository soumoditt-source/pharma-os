# PharmaOS Development Log

Project: PharmaOS  
Goal: deliver a submission-grade OpenEnv environment for real-world drug discovery optimization.

## Verified local status
- The environment, tests, and OpenEnv validators run locally.
- The local dashboard is available through the FastAPI server.
- `inference.py` uses the OpenAI client and supports deterministic seeded task resets.

## Core stack
- FastAPI + Uvicorn for the runtime server
- Pydantic models for typed actions, observations, and state
- RDKit for cheminformatics metrics and structure handling
- PyTorch for the auxiliary ML predictor
- Stable-Baselines3 PPO for an optional training baseline

## Working conventions
- Keep secrets out of the repository. Use environment variables only:
  - `HF_TOKEN`
  - `API_BASE_URL`
  - `MODEL_NAME`
- Prefer module invocation for tooling:
  - `python -m pytest -q`
  - `python -m openenv.cli validate .`
  - `python -m openenv.cli validate --url http://127.0.0.1:8000`
- Start the local server with `python -m server` or the project launch script.

## Remaining external submission steps
- Publish this folder to a public GitHub repository.
- Deploy the Docker app to a live Hugging Face Space.
- Paste those two live URLs into the hackathon submission form.

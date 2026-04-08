# Colab T4 guide

This project includes a real PPO training path for Google Colab T4. It uses the actual PharmaOS environment through [train_ppo.py](/d:/open%20meta%20pytorch/pharma-os/train_ppo.py), not a random mock environment.

## Recommended flow

1. Publish this repository to GitHub.
2. Open [PharmaOS_PPO_Colab.ipynb](/d:/open%20meta%20pytorch/pharma-os/PharmaOS_PPO_Colab.ipynb) in Colab.
3. Set the runtime to `T4 GPU`.
4. Set `PHARMAOS_REPO_URL` to your public repository URL if needed.
5. Run the notebook cells to install dependencies and launch training.

## Direct command in Colab

After cloning the repo and installing requirements:

```bash
python train_ppo.py --task qed_optimizer --timesteps 2500 --device cuda --output-dir trained_agents_colab
```

For the staged research-guided version:

```bash
python train_ppo.py --curriculum --timesteps 6000 --eval-episodes 3 --device cuda --output-dir trained_agents_colab
```

## Notes

- The PPO baseline uses a curated action catalog of medicinal-chemistry proposals.
- Rewards still come from the real environment and real property calculations.
- GPU mainly accelerates the policy network. RDKit scoring remains CPU-heavy.
- For hackathon submission, the PPO path is optional; the required baseline is still [inference.py](/d:/open%20meta%20pytorch/pharma-os/inference.py).

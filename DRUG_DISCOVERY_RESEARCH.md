# Drug discovery research notes

This note collects source-backed papers that shaped the current PharmaOS design and the next upgrade path.

## Papers that directly inform the environment

1. Bickerton et al. (2012), "Quantifying the chemical beauty of drugs"
   Link: https://www.nature.com/articles/nchem.1243
   Why it matters: QED is one of the central objectives in PharmaOS and is a good example of balancing multiple developability features in one score.

2. Olivecrona et al. (2017), "Molecular de-novo design through deep reinforcement learning"
   Link: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x
   Why it matters: this is an early and still useful reference for treating molecular generation as a sequential RL problem instead of a static property regression problem.

3. Brown et al. (2019), "GuacaMol: benchmarking models for de novo molecular design"
   Link: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0390-1
   Why it matters: benchmarking needs more than one reward. Validity, novelty, diversity, and goal quality all matter at once.

4. Blaschke et al. (2020), "REINVENT 2.0: an AI tool for de novo drug design"
   Link: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00915
   Why it matters: REINVENT turns production molecular design into modular scoring, curriculum learning, and diversity control. PharmaOS already mirrors that modular scoring mindset.

5. Blaschke et al. (2020), "Memory-assisted reinforcement learning for diverse molecular de novo design"
   Link: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00473-0
   Why it matters: diversity mechanisms are not cosmetic. They are a direct defense against policy collapse and repetitive scaffold hopping.

6. Bengio et al. (2021), "Flow network based generative models for non-iterative diverse candidate generation"
   Link: https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html
   Why it matters: GFlowNets are a strong future direction for generating diverse high-reward molecules instead of just one greedy best molecule.

7. Blaschke et al. (2024), "Reinvent 4: Modern AI-driven generative molecule design"
   Link: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00812-5
   Why it matters: modern molecule design stacks combine curriculum learning, RL, and multiple generators. That is a strong blueprint for a next-generation PharmaOS agent stack.

8. Abramson et al. (2024), "Accurate structure prediction of biomolecular interactions with AlphaFold 3"
   Link: https://www.nature.com/articles/s41586-024-07487-w
   Why it matters: structure-aware reward terms are now practical. A future PharmaOS version can move from 2D property optimization toward target-aware interaction objectives.

## What PharmaOS already does well

- Uses deterministic environment-side feedback instead of opaque post-hoc judging.
- Balances multiple medicinal-chemistry constraints in the hard task.
- Penalizes PAINS and repeated exploration.
- Tracks scaffold diversity, which is a practical proxy for broader chemical-space coverage.

## What these papers suggest as the next upgrade path

1. Add curriculum stages.
   Start with Lipinski cleanup, then QED, then multi-objective optimization, rather than asking the agent to solve the hardest task from the first step.

2. Add explicit diversity memory.
   The current repeated-molecule penalty is useful, but a stronger scaffold- or similarity-bucket memory would better match the diversity findings from memory-assisted RL and REINVENT-style systems.

3. Add synthesis-aware action constraints.
   The current environment scores synthetic accessibility after the fact. A stronger next step would bias actions toward reaction-feasible edits or fragment libraries.

4. Add target-aware structure signals.
   AlphaFold 3 and related structure-model progress make it realistic to add structure-conditioned objectives in a later version.

5. Evaluate with diversity metrics, not only best score.
   GuacaMol and later work make it clear that a winner-style system should report novelty and scaffold spread alongside final reward.

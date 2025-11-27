This repository contains the implementation and experimental artifacts for our
work on RL splitting heuristic for neural network verification.
Our approach was implemented on top of Marabou verifier with a reinforcement-learning-based
splitting policy trained using a combination of Demonstration-Guided Q-Learning
(DQfD) and Double DQN. The repository includes all networks, properties, trained
agents, and experiment logs needed to reproduce the results reported in the
paper.

Benchmarks:
1. ACAS-Xu Safety Properties
   - 45 neural networks
   - 4 safety specifications (phi_1–phi_4)
   - 180 total verification queries

2. Local Robustness Properties
   - Based on ACAS-Xu network N1,1
   - 1000 randomly selected inputs
   - Three epsilon values: 0.08, 0.09, 0.1
   - 3000 total verification queries

Training Procedure:
A separate reinforcement learning agent is trained for each of the two
experimental setups (safety properties and robustness properties).

Training follows the setup described in the paper:
- 5 epochs of DQfD (1000 steps each)
- 40 epochs of Double DQN (1000 steps each)
- Approximately 45,000 splitting steps in total
- Training performed on single-CPU machines (Debian 12, 2GB RAM)

Running the Code:
The enhanced version of Marabou with RL support is executed with the following
command structure:

    ./Marabou \
        <path_to_network.onnx> \
        <path_to_property.txt> \
        --save-agent-path=<PATH> \
        --DQN-output-file=<RESULTS_DIR> \
        --DQN-mode=<MODE> \
        --pseudo-impact-start

Modes:
- DQN-mode=1 : Train a new agent
- DQN-mode=2 : Evaluate using a trained agent
- DQN-mode=0 : Disable RL (use only baseline heuristics)

Training Parameters:
When running in training mode (DQN-mode=1), the following parameters control the
learning process:

    --DQN-epochs
    --DQN-iters
    --DQN-LR
    --DQN-weight-decay
    --DQN-batch-size
    --DQN-buffer-size
    --DQN-exploration-rate
    --DQfD-lambda-sup
    --DQfD-lambda-decay
    --DQfD-margin
    --DQN-guided-steps
    --DQN-n-examples
    --DQN-demo-examples
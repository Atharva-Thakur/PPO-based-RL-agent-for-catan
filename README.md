# PPO-based RL Agent for Catan

This project implements a Proximal Policy Optimization (PPO) agent to play the board game Settlers of Catan using the `catanatron` library and `catanatron_gym` environment.

## Project Structure

- `ppo_catan/`: Contains the source code for the agent and training loop.
  - `agent.py`: Implementation of the PPO Agent.
  - `model.py`: Neural Network architecture (Actor-Critic).
  - `train.py`: Training loop.
- `main.py`: Entry point to start training.
- `example.py`: A simple script to run the agent for one episode (inference mode).
- `requirements.txt`: List of dependencies.

## Installation

Ensure you have Python installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the agent, run:

```bash
python main.py
```

This will start the training process. Checkpoints will be saved in the `checkpoints/` directory.

### Inference / Example

To see the agent in action (untrained or loaded), run:

```bash
python example.py
```

### Evaluation

To verify if the agent is working and winning games, use the evaluation script:

```bash
python evaluate_agent.py
```

This will run 10 episodes (games) and report the win rate.
- **Untrained/Random Agent**: Expect a win rate around 50% (in a 2-player game).
- **Trained Agent**: As training progresses, the win rate should increase significantly above 50%.

You can also load a specific checkpoint:

```python
# In evaluate_agent.py
evaluate(model_path="checkpoints/ppo_catan_1000.pth")
```

## Details

- **Environment**: `catanatron_gym:catanatron-v1`
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Observation Space**: Box(614,)
- **Action Space**: Discrete(290)
- **Action Masking**: The agent uses valid action masking provided by the environment to ensure it only selects legal moves.

## License

[MIT](LICENSE)

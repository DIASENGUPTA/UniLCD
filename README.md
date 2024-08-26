# Unified-Local-Cloud-Decision-Making with Residual Reinforcement Learning
Embodied vision-based real-world systems, such as mobile robots, require careful balancing between energy consumption, compute latency, and safety constraints to optimize operation across dynamic tasks and contexts. As local computation tends to be restricted, offloading the computation, \ie, to a remote server, can save local resources while providing access to high-quality predictions from powerful and large models. Yet, the resulting communication and latency overhead has led to limited usability of cloud models in dynamic, safety-critical, real-time settings. Towards effectively addressing this trade-off, in this work, we introduce UniLCD, a novel hybrid inference framework for enabling flexible local-cloud collaboration. By efficiently optimizing a flexible routing module via reinforcement learning and a suitable multi-task objective, UniLCD is specifically designed to support multiple constraints of safety-critical end-to-end mobile systems. We validate the proposed approach using a challenging crowded navigation task requiring frequent and timely switching between local and cloud operations. UniLCD demonstrates both improved overall performance and efficiency, by over 17\% compared to state-of-the-art baselines based on various split computing strategies. 


## Requirements
1. Python 3.7/8
2. Docker >=27.0

## Setup
1. Create a virtual environment and install the required dependences:
    ```
    pip install -r requirements.txt
    ```
2. Pull the CARLA 0.9.13 image for Docker:
    ```
    docker pull carlasim/carla:0.9.13
    ```
3. Clone this repository: 
    ```
    git clone https://github.com/DIASENGUPTA/UniLCD.git
    cd UniLCD
    ```
4. Modify the files inside [unilcd_env](unilcd_env/envs/il_models/) to use your imitation-learning trained models as directed. Additionally, place the weights for your models in a directory of your choosing.
5. Install unilcd:
    ```
    pip install unilcd_env
    ```
6. Create a config file based on the [sample config file](unilcd_emb_eval_config.json) provided. You can check the environments available by checking the [envs](unilcd_env/envs/) folder for *env.py files.
7. Before running training/evaluation, start the CARLA Docker container:
    ```
    chmod +x ./run_carla_docker.sh
    ./run_carla_docker.sh start
    ```
8. Now, start your gym environment and train UniLCD:
    ```python
    # Starter code for training UniLCD
    import unilcd_env
    import gymnasium as gym
    import json
    from stable_baselines3 import PPO

    config = json.load(open('unilcd_emb_eval_config.json'))
    env = gym.make(**config)
    env.reset()
    # Select your favorite RL algorithm and train. We recommend Stable Baselines3 for its integration with Gymnasium
    model = PPO("MlpPolicy", env)
    model.learn(total_timesteps=100000)
    ```
9. Stop the CARLA Docker container:
    ```
    ./run_carla_docker.sh stop
    ```

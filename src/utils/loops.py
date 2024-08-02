import numpy as np

from tqdm import tqdm

def simple_ddpg_loop(agent, env, episodes, max_timesteps, n_random_episodes):

    for episode in range(episodes):
        state = env.reset()
        # agent.reset_noise_var()
        episode_rewards = []
        max_reward = None
        done = False
        progress_bar = tqdm(range(max_timesteps))
        for t in progress_bar:
            if episode < n_random_episodes:
                action = env.get_random_action()
            else:
                action = agent.pi(state, eval=True)
            next_state, reward, done = env.step(action)
            actor_loss, critic_loss = agent.step(
                state, action, reward, next_state, done
            )
            state = next_state
            episode_rewards.append(reward)
            
            if (max_reward is None) or (reward > max_reward):
                max_reward = reward

            progress_bar.set_description(
                f"Episode Num: {episode + 1} "
                f"Time step: {t + 1} "
                f"Avg. Reward: {np.average(episode_rewards[-min(10, len(episode_rewards))]):.3f} "
                f"Max Reward: {max_reward:.3f} "
                f"Actor Loss: {actor_loss:0.3f} "
                f"Critic Loss: {critic_loss:0.3f} "
                f"Noise Var: {agent.noise_var:0.3f}"
            )

            if done:
                break



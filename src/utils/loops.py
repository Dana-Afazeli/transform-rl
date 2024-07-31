import numpy as np

from tqdm import tqdm

def simple_ddpg_loop(agent, env, episodes, max_timesteps):
    progress_bar = tqdm(range(episodes))
    
    for episode in progress_bar:
        state = env.reset()
        episode_rewards = []
        max_reward = None
        done = False
        for t in range(max_timesteps):
            action = agent.pi(state)
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
            )

            if done:
                break



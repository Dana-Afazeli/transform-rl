import numpy as np

from tqdm import tqdm

def simple_ddpg_loop(agent, env, episodes, max_timesteps, 
        n_random_episodes, reset_noise_var_every_n_episodes):

    for episode in range(episodes):
        state = env.reset()
        if episode % reset_noise_var_every_n_episodes == 0:
            agent.reset_noise_var()
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


def ddpg_loop_with_recordings(agent, env, episodes, max_timesteps, 
        n_random_episodes, reset_noise_var_every_n_episodes, 
        recorder, plotter, save_path):
    
    for episode in range(episodes):
        state = env.reset()
        if episode % reset_noise_var_every_n_episodes == 0:
            agent.reset_noise_var()
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

            recorder.record_statistics(agent)
            recorder.record_statistics(env)

            if done:
                break
        
        recorder.aggregate_statistics(
            actor_loss='avg', 
            critic_loss='avg', 
            n_episode='any',
            reward=['avg', 'max']
        )

    plotter.plot_statistics(recorder, save_path)
    

def transformer_ddpg_loop_with_recordings(agent, env, episodes, max_timesteps, 
        n_random_episodes, reset_noise_var_every_n_episodes, 
        recorder, plotter, save_path):
    
    for episode in range(episodes):
        state = env.reset()
        agent.reset()
        if episode % reset_noise_var_every_n_episodes == 0:
            agent.reset_noise_var()
        episode_rewards = []
        max_reward = None
        done = False
        progress_bar = tqdm(range(max_timesteps))
        for t in progress_bar:
            if episode < n_random_episodes:
                action = env.get_random_action()
            else:
                states = agent.replay_buffer.stm['states']
                stm = (states + [state])[-agent.context_len:]
                action = agent.pi(np.vstack(stm), eval=True)
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

            # recorder.record_statistics(agent)
            # recorder.record_statistics(env)

            if done:
                break
        
        # recorder.aggregate_statistics(
        #     actor_loss='avg', 
        #     critic_loss='avg', 
        #     n_episode='any',
        #     reward=['avg', 'max']
        # )

    # plotter.plot_statistics(recorder, save_path)
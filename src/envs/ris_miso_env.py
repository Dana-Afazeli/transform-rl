import numpy as np
import torch

from envs.base import *

class RIS_MISO(BaseEnv):
    def __init__(
        self,
        num_antennas,
        num_RIS_elements,
        num_users,
        power_t,
        channel_est_error=False,
        AWGN_var=1e-2,
        channel_noise_var=1e-2
    ):
        self.max_action = 1
        self.M = num_antennas
        self.L = num_RIS_elements
        self.K = num_users
        self.power_t = power_t
        self.channel_est_error = channel_est_error

        assert self.M == self.K

        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        power_size = 2 * self.K

        channel_size = 2 * (self.L * self.M + self.L * self.K)

        self.action_dim = 2 * self.M * self.K + 2 * self.L
        self.state_dim = power_size + channel_size + self.action_dim

        self.H_1 = None
        self.H_2 = None
        self.G = np.eye(self.M, dtype=complex)
        self.Phi = np.eye(self.L, dtype=complex)

        self.state = None
        self.done = None

        self.episode_t = None

    def _compute_H_2_tilde(self):
        return self.H_2.T @ self.Phi @ self.H_1 @ self.G

    def _whiten(self, state):
        return (state - np.mean(state)) / np.std(state)

    def reset(self):
        self.episode_t = 0

        self.H_1 = np.random.normal(0, np.sqrt(0.5), (self.L, self.M)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.L, self.M))
        self.H_2 = np.random.normal(0, np.sqrt(0.5), (self.L, self.K)) + 1j * np.random.normal(0, np.sqrt(0.5),
                                                                                               (self.L, self.K))

        init_action_G = np.hstack((np.real(self.G.reshape(1, -1)), np.imag(self.G.reshape(1, -1))))
        init_action_Phi = np.hstack(
            (np.real(np.diag(self.Phi)).reshape(1, -1), np.imag(np.diag(self.Phi)).reshape(1, -1)))

        init_action = np.hstack((init_action_G, init_action_Phi))

        Phi_real = init_action[:, -2 * self.L:-self.L]
        Phi_imag = init_action[:, -self.L:]

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()
        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((init_action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        return self._whiten(self.state.reshape(-1,))

    def _compute_reward(self, Phi):
        reward = 0
        opt_reward = 0

        for k in range(self.K):
            h_2_k = self.H_2[:, k].reshape(-1, 1)
            g_k = self.G[:, k].reshape(-1, 1)

            x = np.abs(h_2_k.T @ Phi @ self.H_1 @ g_k) ** 2

            x = x.item()

            G_removed = np.delete(self.G, k, axis=1)

            interference = np.sum(np.abs(h_2_k.T @ Phi @ self.H_1 @ G_removed) ** 2)
            y = interference + (self.K - 1) * self.awgn_var

            rho_k = x / y

            reward += np.log(1 + rho_k) / np.log(2)
            opt_reward += np.log(1 + x / ((self.K - 1) * self.awgn_var)) / np.log(2)

        return reward, opt_reward

    def _compute_power(self, a):
        # Normalize the power
        G_real = a[:, :self.M ** 2].cpu().data.numpy()
        G_imag = a[:, self.M ** 2:2 * self.M ** 2].cpu().data.numpy()

        G = G_real.reshape(G_real.shape[0], self.M, self.K) + 1j * G_imag.reshape(G_imag.shape[0], self.M, self.K)

        GG_H = np.matmul(G, np.transpose(G.conj(), (0, 2, 1)))

        current_power_t = torch.sqrt(torch.from_numpy(np.real(np.trace(GG_H, axis1=1, axis2=2)))).reshape(-1, 1)

        return current_power_t

    def _compute_phase(self, a):
        # Normalize the phase matrix
        Phi_real = a[:, -2 * self.L:-self.L].detach()
        Phi_imag = a[:, -self.L:].detach()

        return torch.sum(torch.abs(Phi_real), dim=1).reshape(-1, 1) * np.sqrt(2), torch.sum(torch.abs(Phi_imag), dim=1).reshape(-1, 1) * np.sqrt(2)


    def step(self, action):
        action = np.tanh(action).reshape(1,40)
        current_power_t = (
            self._compute_power(torch.Tensor(action))
            .expand(-1, 2 * self.M ** 2) / np.sqrt(10 ** (self.power_t / 10))
        )

        real_normal, imag_normal = self._compute_phase(torch.Tensor(action))

        real_normal = real_normal.expand(-1, self.L)
        imag_normal = imag_normal.expand(-1, self.L)

        division_term = torch.cat(
            [current_power_t, real_normal, imag_normal], 
            dim=1
        ).detach().cpu().numpy()
        
        action = self.max_action * action / division_term

        self.episode_t += 1

        action = action.reshape(1, -1)

        G_real = action[:, :self.M ** 2]
        G_imag = action[:, self.M ** 2:2 * self.M ** 2]

        Phi_real = action[:, -2 * self.L:-self.L]
        Phi_imag = action[:, -self.L:]

        self.G = G_real.reshape(self.M, self.K) + 1j * G_imag.reshape(self.M, self.K)

        self.Phi = np.eye(self.L, dtype=complex) * (Phi_real + 1j * Phi_imag)

        power_t = np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2

        H_2_tilde = self._compute_H_2_tilde()

        power_r = np.linalg.norm(H_2_tilde, axis=0).reshape(1, -1) ** 2

        H_1_real, H_1_imag = np.real(self.H_1).reshape(1, -1), np.imag(self.H_1).reshape(1, -1)
        H_2_real, H_2_imag = np.real(self.H_2).reshape(1, -1), np.imag(self.H_2).reshape(1, -1)

        self.state = np.hstack((action, power_t, power_r, H_1_real, H_1_imag, H_2_real, H_2_imag))

        reward, opt_reward = self._compute_reward(self.Phi)

        done = opt_reward == reward

        return self._whiten(self.state.reshape(-1,)), reward, done
    
    def get_action_info(self):
        #TODO: fix this
        return 'continuous', (self.action_dim,)
    
    def get_state_info(self):
        #TODO: fix this
        return 'continuous', (self.state_dim,)
import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
from IPython.display import clear_output
from torch import nn
from atari_wrappers import make_env


def train(env, agent, target_network, exp_replay, loss_func, device, lr=1e-4, 
          total_steps=3 * 10**6, verbose_steps=3 * 10 ** 5, batch_size=32,
          decay_steps=1 * 10**6, init_epsilon=1.0, final_epsilon=0.1, timesteps_per_epoch=1,
          max_grad_norm=50, loss_freq=50, refresh_target_network_freq=5000, eval_freq=5000):
    stop_evaluation = False
    
    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []

    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    
    state = env.reset()
    for step in trange(total_steps + 1):
        agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play
        _, state = utils.play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        states, actions, rewards, next_states, is_done = exp_replay.sample(batch_size)
        loss = loss_func(states, actions, rewards, next_states, is_done,
                         agent, target_network, device)


        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            grad_norm_history.append(grad_norm)

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            #target_network.parameters() = agent.parameters()
            target_network.load_state_dict(agent.state_dict())

        if step == verbose_steps:
            print("Stopping plotting to reduce training time.")
            stop_evaluation = True
        if (step % eval_freq == 0):
            # eval the agent
            mean_rw_history.append(utils.evaluate(
                make_env(seed=step), agent, n_games=3, greedy=True, t_max=1000)
            )
            initial_state_q_values = agent.get_qvalues(
                [make_env(seed=step).reset()]
            )
            initial_state_v_history.append(np.max(initial_state_q_values))
            if not stop_evaluation:
                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                    (len(exp_replay), agent.epsilon))

                plt.figure(figsize=[16, 9])
                plt.subplot(2, 2, 1)
                plt.title("Mean reward per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(utils.smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                plt.plot(utils.smoothen(grad_norm_history))
                plt.grid()

                plt.show()

    return {'reward_history': mean_rw_history, 
            'td_loss_history': td_loss_history, 
            'grad_norm_history': grad_norm_history,
            'initial_state_v_history': initial_state_v_history}
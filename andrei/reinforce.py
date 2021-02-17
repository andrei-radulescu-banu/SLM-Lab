from torch.distributions import Categorical
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gamma = 0.99

class Pi(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Pi, self).__init__()
        layers = [
            nn.Linear(in_dim, 64), # in_dim number of input states, 64 nodes wide middle layer
            nn.ReLU(),
            nn.Linear(64, out_dim), # out_dim number of output actions
        ]
        self.model = nn.Sequential(*layers)
        self.onpolicy_reset()
        self.train() # set training mode

    def onpolicy_reset(self):
        self.log_probs = []
        self.rewards = []

    def forward(self, x):
        pdparam = self.model(x)
        return pdparam

    def act(self, state):
        x = torch.from_numpy(state.astype(np.float32)) # state expressed as tensor
        pdparam = self.forward(x) # forward pass estimates pi(a|s)
        pd = Categorical(logits=pdparam) # pd is probability distribution of pi(a|s)
        action = pd.sample() # get sample pi(a|s)
        log_prob = pd.log_prob(action) # get log_prob sample of pi(a|s)
        self.log_probs.append(log_prob) # append to array of log_prob samples
        return action.item() # return the action sample
    
def train(pi, optimizer):
    # Inner gradient-ascent loop of REINFORCE algorithm
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32) # the returns
    future_ret = 0.0
    # compute the return efficiently
    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret
    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = - log_probs * rets # gradient term: negative for maximizing
    loss = torch.sum(loss)
    optimizer.zero_grad() # zero out old gradients
    loss.backward() # backpropagate, compute gradients of current tensor w.r.t. graph leaves
    optimizer.step() # gradient-ascent, update the weights
    return loss
    
def main():
    env = gym.make("CartPole-v0")
    in_dim = env.observation_space.shape[0] # 4 states in cartpole env
    out_dim = env.action_space.n # 2 actions in cartpole env
    pi = Pi(in_dim, out_dim) # NN estimates policy pi_theta for REINFORCE
    optimizer = optim.Adam(pi.parameters(), lr=0.01) # sets learning rate
    for epi in range(300): # will execute 300 trajectories
        state = env.reset() # reset the gym environment
        for t in range(200): # trajectory will have 200 steps; cartpole max timestep is 200
            action = pi.act(state) # act based on state; this also saves log_prob sample of pi(a|s)
            state, reward, done, _ = env.step(action) # set gym action, get the reward, the new state, and see if pole fell down
            pi.rewards.append(reward) # save the reward
            env.render() # render the new cartpole state
            if done:
                break # pole fell down, no need to continue to the end of the trajectory
        loss = train(pi, optimizer) # train based on this trajectory, and based on saved rewards and log_prob samples 
        episode_reward = sum(pi.rewards) 
        solved = episode_reward > 195.0
        pi.onpolicy_reset() # onpolicy: clear memory after training
        print(f'Episode {epi}, loss {loss}, episode_reward {episode_reward}, solved {solved}')
    
if __name__ == '__main__':
    main()

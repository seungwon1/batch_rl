# Deep Learning Papers Review — Striving for Simplicity in Off-policy Deep Reinforcement Learning

Rishabh Agarwal, Dale Schuurmans, Mohammad Norouzi. Striving for Simplicity in Off-policy Deep Reinforcement Learning. (https://arxiv.org/abs/1907.04543)

## Background
- Off-policy Q-learning allows two different policies: behavior policy and target policy. Behavior policy selects an action, whereas the target policy is used to estimate the next state-action value. On the other hand, on-policy learning uses the same policy for choosing the action to execute and estimating the next state-action value. Additionally, the term off-policy can be used to indicate pure exploratory random behavior, which is literally off from the current policy, whereas on-policy can suggest following current policy, which means acting greedily or choosing the action using epsilon-greedy strategy

![QL](https://github.com/seungwon1/batch_rl/blob/master/capture/QL.PNG)

- Deep Q-learning (DQN) algorithm is labeled as off-policy RL algorithm but it is considered as “online” rather than “offline” since it exploits near on-policy behavior such as epsilon-greedy with an experience replay. Training DQN in offline setting, i.e. batch setting, which means using data that have been already logged, doesn’t work well due to the extrapolation error. (https://arxiv.org/pdf/1812.02900.pdf)

![dqn](https://github.com/seungwon1/batch_rl/blob/master/capture/dqn.PNG)

- Off-policy RL algorithms such as Q-learning are more practical and efficient to handle real-world problems with already logged data than on-policy algorithms such as policy gradient since in principle, off-policy algorithm can learn from any data collected by any policy. But off-policy RL algorithms are unstable and there is no guarantee that it converges especially when using neural networks for function approximation.

- To make stable off-policy DQN, here have been various advances such as distributional RL: C51 or quantile regression DQN (QR-DQN)(https://arxiv.org/pdf/1710.10044.pdf). Both C51 and QR-DQN obatin state-of-the-art performance on Atrai 2600 games by estimating the distribution of return using support with assigned probabilities rather than estimating expected return.
![rainbow](https://github.com/seungwon1/batch_rl/blob/master/capture/rainbow.PNG)

## Main point & Summary of this paper

- This paper investigates training DQN with the batch setting, i.e. “ Is it possible to make successful agents using completely on offline logged data?” It presents the performance of distributional RL, nature DQN(2015 Mnih et al) on Atrai 2600 games in both batch setting(offline setting) and online setting. Surprisingly, offline QR-DQN outperforms online C51 and online DQN, which is contrary to recent work.

- This paper also tries to make RL algorithms as simple as possible. In spite of recent advances in DQN, those might lead to complex hypothesis and it is more likely to coincidentally fits data than simple hypothesis does (Occam’s razor). It is important to ask “ Are those advances really necessary? “ The paper proposes Ensemble DQN and Random Ensemble Mixture (REM) DQN, which are way simpler than distributional RL, and it shows that REM DQN outperforms C51 and QR-DQN in the offline setting and online REM DQN performs comparably with online QR-DQN.

## Proposed Architecture

![net](https://github.com/seungwon1/batch_rl/blob/master/capture/net.PNG)

- DQN, QR-DQN, Ensemble-DQN, and REM DQN use the same network architecture but DQN outputs state-action value Q(s,a) and QR-DQN outputs K x |A| number of Z to represent the distribution of return rather than just estimating the expected value of the return.

- Ensemble-DQN is a simple extension of DQN, which uses an ensemble of Q(s,a) (see below). Each random Q-head with its target is used to compute Bellman error respectively as shown in below loss function.

![ens](https://github.com/seungwon1/batch_rl/blob/master/capture/ens.PNG)

- REM-DQN uses a convex combination of Q values to estimate Q-values (see below). Each coefficient alpha is randomly drawn from a categorical distribution.

![rem](https://github.com/seungwon1/batch_rl/blob/master/capture/rem.PNG)

## Experiments & Results

![result1](https://github.com/seungwon1/batch_rl/blob/master/capture/result1.PNG)

![result2](https://github.com/seungwon1/batch_rl/blob/master/capture/result2.PNG)

![result3](https://github.com/seungwon1/batch_rl/blob/master/capture/result3.PNG)

- Experiments are performed with the same number of Q-heads at 200 and the same value of other parameters. Experience replay consists of 50 million tuples of (observation, action, reward, next observation), which have been logged by Nature DQN(Mnih et al 2015).

- Offline QR-DQN outperforms online C51 and online DQN.

- REM DQN outperforms C51 and QR-DQN in the offline setting and online REM DQN performs comparably with online QR-DQN.

## Contribution

- This paper shows that it is possible to successfully train DQN in batch setting, which uses a fixed dataset without interacting with environments. Using a common logged dataset can improve the reproducibility of off-policy RL algorithms, accelerating research by serving a testbed similar to Imagenet dataset in computer vision.

## Reference

- Agarwal, Rishabh, Dale Schuurmans, and Mohammad Norouzi. “Striving for Simplicity in Off-policy Deep Reinforcement Learning.” arXiv preprint arXiv:1907.04543 (2019).
- Bellemare, Marc G., Will Dabney, and Rémi Munos. “A distributional perspective on reinforcement learning.” Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
- Dabney, Will, et al. “Distributional reinforcement learning with quantile regression.” Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
- Fujimoto, Scott, David Meger, and Doina Precup. “Off-policy deep reinforcement learning without exploration.” arXiv preprint arXiv:1812.02900 (2018).
- Hessel, Matteo, et al. “Rainbow: Combining improvements in deep reinforcement learning.” Thirty-Second AAAI Conference on Artificial Intelligence. 2018.
- Mnih, Volodymyr, et al. “Human-level control through deep reinforcement learning.” Nature 518.7540 (2015): 529.
- Silver. D. (2015). Model-Free Control. http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/control.pdf

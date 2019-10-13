# Batch_RL

Tensorflow implementation for replicating the experiments described in the paper ["Striving for Simplicity in Off-policy Deep Reinforcement Learning"]( https://arxiv.org/abs/1907.04543). This repository aims to implement all variants of DQN used in the paper in pure tensorflow from scratch, whereas code provided by author used dopamine framework. The implementation contains:

[1] Classic DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)  
[2] C51: [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[3] QR DQN: [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044)  
[4] Ensemble DQN: [Striving for Simplicity in Off-policy Deep Reinforcement Learning](https://arxiv.org/abs/1907.04543)  
[5] Random Ensemble Mixture(REM) DQN: [Striving for Simplicity in Off-policy Deep Reinforcement Learning](https://arxiv.org/abs/1907.04543)  

## Paper Review
- [Blog](https://medium.com/@seungwonkim_57156/deep-learning-papers-review-striving-for-simplicity-in-off-policy-deep-reinforcement-learning-ac49c4aa26e2), [Pdf](https://github.com/seungwon1/batch_rl/blob/master/docs/paper_review.pdf)

## Dependencies
- Python 3.6 or greater
- Tensorflow 1.13.1
- Numpy 1.16.3
- OpenAI Gym version 0.10.5
- Matplotlib
- OpenCV
- Box2D 2.3.3
- ffmpeg

## Online Performance
Below command line replicates experiments done in the paper: DQN [[1]](#batch_rl), C51 [[2]](#batch_rl), QR DQN [[3]](#batch_rl).
```
python main.py --arch=DQN --eps=1.0 --final_eps=0.1 --max_frames=10000000 --opt=rmsprop --lr=0.00025 --game=PongNoFrameskip-v4

python main.py --arch=C51 --eps=1.0 --final_eps=0.05 --max_frames=10000000  --opt=adam --lr=0.00025 --num_heads=51 --game=PongNoFrameskip-v4

python main.py --arch=QR_DQN --eps=1.0 --final_eps=0.01 --max_frames=10000000  --opt=adam --lr=0.00005 --num_heads=200 --game=PongNoFrameskip-v4
```
Args
```
-arch : Model architecture
-eps : Starting value of epsilon
-final_eps : final value of epsilon
-max_frames : Number of iterations (step count)
-opt : optimizer
-lr : learning rate of optimizer
-num_heads : number of heads for C51, QR-DQN, Ensemble DQN and REM
-game : Atari game env
```
- Other hyperparameters are the same as those in the paper (see main.py).

#### Results: online performance on PongNoFrameskip-v4 env.
<p align="center">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/n_dqn_p.png" width="250">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/c51_p.png" width="250">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/qr_dqn_p.png" width="250">
</p>

- Average reward of 100 previous episode, Left: DQN [[1]](#batch_rl), Middle: C51, DQN [[2]](#batch_rl), Right: QR DQN, C51, DQN [[3]](#batch_rl)
- For each figure, the same hyperparameter is used for all DQNs.
- Note that 1 frame in the x-axis includes 4 step counts.
- Linearly decaying epsilon from 1 to 0.1(left), 0.05(middle), 0.01(right) over the first 1M frames.
- C51 is able to reach the best score but seems to learn optimal policy a bit slower than classic(Nature) DQN does, which is different from the results in [[2]](#batch_rl) and [[3]](#batch_rl). This might be mitigated by using different epsilon decaying schedule(e.g., exponential decay) or using different target update interval, seed, etc.

<p align="center">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/n_dqn_l.png" width="250">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/c51_l.png" width="250">
<img src="https://github.com/seungwon1/batch_rl/blob/master/figure/qr_dqn_l.png" width="250">
</p>

- Average Loss(per episode), Left: DQN [[1]](#batch_rl), Middle: C51, DQN [[2]](#batch_rl), Right: QR DQN, C51, DQN [[3]](#batch_rl)

## Offline Performance
- In progress

## Reference
- [Deepmind's code](https://sites.google.com/a/deepmind.com/dqn/)  
- [CS294 HW3](https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3)

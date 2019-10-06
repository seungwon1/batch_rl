# batch_rl

This repository contains tensorflow implementation for replicating the experiments described in the paper ["Striving for Simplicity in Off-policy Deep Reinforcement Learning"]( https://arxiv.org/abs/1907.04543)

## Paper Review
- Blog: https://medium.com/@seungwonkim_57156/deep-learning-papers-review-striving-for-simplicity-in-off-policy-deep-reinforcement-learning-ac49c4aa26e2
- Pdf: https://github.com/seungwon1/batch_rl/blob/master/docs/paper_review.pdf

## Online Performance
```
python main.py --arch=DQN --eps=1.0 --final_eps=0.1 --max_frames=50000000 --opt=rmsprop --lr=0.00025

python main.py --arch=C51 --eps=1.0 --final_eps=0.05 --max_frames=50000000  --opt=adam --lr=0.00025

python main.py --arch=QR_DQN --eps=1.0 --final_eps=0.01 --max_frames=50000000  --opt=adam --lr=0.000005
```
- Sanity Check to ensure that each DQN algorithm(DQN, C51, QR_DQN) works well
- Test env: Pong

## Offline Performance
- In progress

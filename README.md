# Batch_RL

This repository contains tensorflow implementation for replicating the experiments described in the paper ["Striving for Simplicity in Off-policy Deep Reinforcement Learning"]( https://arxiv.org/abs/1907.04543)

## Paper Review
- Blog: https://medium.com/@seungwonkim_57156/deep-learning-papers-review-striving-for-simplicity-in-off-policy-deep-reinforcement-learning-ac49c4aa26e2
- Pdf: https://github.com/seungwon1/batch_rl/blob/master/docs/paper_review.pdf

## Dependencies
- Python 3.6 or greater
- Tensorflow 1.13.1
- Numpy 1.16.3
- OpenAI Gym version 0.10.5
- Matplotlib
- OpenCV
- seaborn
- Box2D 2.3.3
- ffmpeg

## Online Performance
Below command line replicates experiments done in the paper: DQN(Minh et al 2015), C51(Bellemare et al 2017), QR-DQN(Dabney et al 2017).
```
python main.py --arch=DQN --eps=1.0 --final_eps=0.1 --max_frames=10000000 --opt=rmsprop --lr=0.00025 --game=PongNoFrameskip-v4

python main.py --arch=C51 --eps=1.0 --final_eps=0.05 --max_frames=10000000  --opt=adam --lr=0.00025 --num_heads=51 --game=PongNoFrameskip-v4

python main.py --arch=QR_DQN --eps=1.0 --final_eps=0.01 --max_frames=10000000  --opt=adam --lr=0.000005 --num_heads=200 --game=PongNoFrameskip-v4
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
- Sanity Check to ensure that each DQN algorithm(DQN, C51, QR_DQN) works well(Test env: PongNoFrameskip-v4)

## Offline Performance
- In progress

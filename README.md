# AlphaDog
**A Gomoku AI Inspired by AlphaGo Series**

## Note
**Its performace is not so satisfying yet.**

## Overview 简介 (Ver 0.1 - 2025/07)

AlphaDog is a Gomoku (Five in a Row) AI inspired by the AlphaGo series. It uses a combination of deep neural networks and Monte Carlo Tree Search (MCTS) to play Gomoku, and is trained through Reinforcement Learning and self-play data.

AlphaDog 是一个受 AlphaGo 系列启发的五子棋 AI。它结合了深度神经网络和蒙特卡洛树搜索（MCTS），通过强化学习和自我对弈进行训练。

## About training

During training, init `lr` and `batch_size` are set to 1e-3 and 256.

`eps` for Dirichlet noise is set to 0.2. Flip and rotation are used for data augmentation.
Data buffer is cleared every 5 epoches.

## Usage 如何使用

- **Training**: To train the AI model, run the `AlphaDog.py` script. You can adjust hyperparameters.
- To have faster training, try the `AlphaDog_para.py` script, which supports parallel self-play to accelerate training. **Make sure your device support CUDA and `multiprocessing`**.
- **Playing**: To play against AI or check AI performance, run the `GMK_Pygame.py` script. You can set player and switch PvP / PvC / CvC mode. You may also set param `useMCTS` to decide whether AI uses MCTS when playing.
- Pre-trained models are in folder `trained`.

--------------------------------------------------

- **训练**：运行 `AlphaDog.py` 脚本来训练 AI 模型。您还可以调整超参数。
- 可以通过运行 `AlphaDog_para.py` 脚本来实现更快速的训练。它可以通过并行地自我对弈来加速训练。**请确保你的设备支持 CUDA 和 `multiprocessing`**。
- **对弈**：运行 `GMK_Pygame.py` 脚本来与 AI 对弈或者观察 AI 表现。您可以设置 player 参数，来切换PvP、PvC或者CvC模式。您还可以设置 `useMCTS` 参数来决定 AI 是否在对弈中使用 MCTS。
- 预训练好的模型参数在 `trained` 文件夹。

## Requirements

- Python 3.12
- Numpy
- Torch
- Pygame
- *Torchinfo (not necessary)*

## Updates
#### Version 0 (2025/02):
- **Ver 0.1 - 2025/07**: Code optimization, no change in model.
- **2025/08**: Convert to half precision to accelerate.
- **2025/08**: Convert back to float32, for half is not so friendly for training (may be supported in later versions). (>_<)

## License

This project is licensed under the MIT License. 

## Acknowledgments

This project is inspired by DeepMind's AlphaGo series.

Please contact me if you have any suggestions or questions.

Thanks for the help from open source community.
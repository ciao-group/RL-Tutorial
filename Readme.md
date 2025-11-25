# RL - Tutorial

Welcome to the Reinforcement Learning Tutorial, which is part of the lecture series "Computational Modeling"!

### Prerequisites
This project relies on *Conda*, an open source package and environment management system for Windows, Linux, and macOS. Conda allows you to install multiple versions of software packages and switching easily between them. 
Please install Conda on your local machine, which you intend to use for the lab. 

There are two versions of Conda available, namely *Miniconda* and *Anaconda*. Where Anaconda provides more functionalities such as a GUI. However, Miniconda is absolutely sufficient for this tutorial.

- [Installation guide Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#macos-2)
- [Installation guide Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install)


### Setup
1. **Create the environment** using the provided YAML file:
   ```bash
   conda env create -f environment.yml

2. **Activate the Conda environment**:
    ```bash
    conda activate RL-Tutorial

3. **Start training** on your custom environment
    ```bash
    python main.py

### Important
> **Note**: This codebase is designed for **active learning**. It will **not** run successfully out of the box! 

The code has several open ToDos and is intended to be completed during the lecture. The fully functioning environment will be provided via the usual channels after the lecture.

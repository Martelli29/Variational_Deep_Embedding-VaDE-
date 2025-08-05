# Variational Deep Embedding (VaDE) for Image Clustering 🏞️

This repository presents a convolutional implementation of the [**Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering**](https://arxiv.org/abs/1611.05148) paper. The primary goal of this project is to provide a straightforward and well-commented code base that serves as an entry point for understanding the VaDE architecture and its mathematical underpinnings.

The implementation is tailored for the clustering of [landscape images from a Kaggle dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset).

---

## 🚀 Repository Utility

The core purpose of this repository is to offer a compact and comprehensible codebase that helps users grasp the inner workings of the **VaDE model**. It's designed to help the comprehension of the [original work's repository](https://github.com/slim1017/VaDE) by providing a compact and well commented, albeit specific, example.

⚠️ **Note:** This is not a "ready-to-use" script. The code is highly specific to my project's task, and there is currently no configuration file for easy customization. The network structure is hard-coded to solve my personal task. I may, however, improve this aspect in the future.

---

## 📁 Repository Structure

* **`train_VaDE.py`**: The main Python script containing the implementation of the convolutional VaDE model. This is where you'll find the architecture and training loop.
* **`VaDe_evaluate.ipynb`**: A Jupyter Notebook that showcases the results of the specific clustering task. While you're free to ignore it, it offers a peek into the model's performance on the landscape image dataset. (As noted, the results are decent but could be significantly improved with better computational resources.)

---

## 🎓 Academic Context

This project was developed for an exam of my Master's Degree in Applied Physics at the **University of Bologna (UNIBO)**.
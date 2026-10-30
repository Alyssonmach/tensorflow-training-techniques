# TensorFlow Training Techniques

Hands-on notebooks for the **Custom and Distributed Training with TensorFlow** course. The material covers automatic differentiation, custom training loops, graph execution, and distributed training strategies.

> These notebooks accompany the [DeepLearning.AI course on Coursera](https://www.coursera.org/learn/custom-distributed-training-with-tensorflow). Course materials and assignments are provided for learning and practice.

## Contents

| Week | Topics | Notebooks |
| --- | --- | --- |
| 1 | Tensors, automatic differentiation, and `GradientTape` | [Assignment](week1/w1_assignment.ipynb) · [Lab 1: Basic Tensors](week1/w1_lab1.ipynb) · [Lab 2: Gradient Tape](week1/w1_lab2.ipynb) |
| 2 | Custom models, training loops, and loss functions | [Assignment: Breast Cancer Prediction](week2/w2_assignment.ipynb) · [Lab 1: Custom Training Basics](week2/w2_lab1.ipynb) · [Lab 2: Fashion MNIST](week2/w2_lab2.ipynb) |
| 3 | AutoGraph and TensorFlow graph execution | [Assignment: Horse or Human](week3/w3_assignment.ipynb) · [Lab 1: AutoGraph Basics](week3/w3_lab1.ipynb) · [Lab 2: Complex Graphs](week3/w3_lab2.ipynb) |
| 4 | Distributed training with `tf.distribute` | [Assignment](week4/w4_assignment.ipynb) · [Lab 1: Mirrored Strategy](week4/w4_lab1.ipynb) · [Lab 2: Multi-GPU](week4/w4_lab2.ipynb) · [Lab 3: TPU Strategy](week4/w4_lab3.ipynb) · [Lab 4: One Device Strategy](week4/w4_lab4.ipynb) |

Assignments are graded course exercises; labs are guided practice. Each notebook includes its own instructions and code.

## Getting started

You can open a notebook in [Google Colab](https://colab.research.google.com/) or run it locally in Jupyter. To run locally, use a Python environment with TensorFlow 2.x and the libraries imported by the notebook. A common starting point is:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install jupyterlab tensorflow tensorflow-datasets tensorflow-hub numpy pandas matplotlib scikit-learn tqdm
jupyter lab
```

Some notebooks require a GPU or TPU to follow the distributed-training exercises as written. The TPU lab is designed for a TPU-enabled runtime, such as Google Colab. CPU execution may work for parts of the material but can be much slower.

### Dataset notes

- `week2/w2_assignment.ipynb` reads `./data/data.csv`. When running from the repository root, place the course-provided CSV at `data/data.csv` (or adjust `data_file` if your working directory differs).
- `week2/w2_lab2.ipynb` loads Fashion MNIST with downloads disabled. Make sure the dataset is already present in the TensorFlow Datasets cache, or change the notebook's `download` option to `True`.
- Other notebooks may download datasets or TensorFlow Hub models on first run. This requires network access and enough local storage.

## Repository layout

```text
.
├── week1/   # Tensors and gradients
├── week2/   # Custom training
├── week3/   # Graph mode and AutoGraph
├── week4/   # Distributed training
├── LICENSE
└── README.md
```

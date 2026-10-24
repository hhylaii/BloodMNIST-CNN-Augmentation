# BloodMNIST Blood Cell Classification: Evaluating Mild Data Augmentation

A reproducible deep-learning study of **eight-class blood-cell image classification** using the public [BloodMNIST dataset](https://medmnist.com/). This project compares a multilayer perceptron (MLP), a convolutional neural network (CNN), and the **same CNN trained with mild geometric augmentation**. It was developed for **MD6329 — Deep Learning for Healthcare AI**.

## Research question

**Does mild geometric data augmentation improve CNN classification performance on BloodMNIST?**

The main controlled comparison is **CNN vs. CNN + augmentation**: the network architecture and training procedure remain the same, while only the training-image transformations change. The MLP provides an additional baseline.

## Dataset

BloodMNIST contains **17,092 RGB images**, each **28 × 28 pixels**, divided into the official training, validation, and test splits.

| Split | Images |
| --- | ---: |
| Training | 11,959 |
| Validation | 1,712 |
| Test | 3,421 |

The eight classes are **basophil, eosinophil, erythroblast, immature granulocyte, lymphocyte, monocyte, neutrophil,** and **platelet**. Images are converted to PyTorch tensors with pixel values scaled to `[0, 1]`. The dataset is downloaded through the `medmnist` package; it is not included in this repository.

## Models and methods

| Model | Architecture / treatment |
| --- | --- |
| MLP baseline | Flattened RGB input (2,352 features), two hidden layers of 256 and 128 units, ReLU, dropout 0.3, eight-class output |
| CNN baseline | Three 3×3 convolutional layers (32, 64, 128 channels), batch normalization and ReLU; max pooling after the first two layers; adaptive average pooling; dropout 0.3; eight-class classifier |
| CNN + augmentation | **Identical CNN** with `RandomHorizontalFlip(p=0.5)` and `RandomRotation(degrees=15)` applied **only to training images** |

**Training configuration:** PyTorch; cross-entropy loss; Adam optimizer; learning rate `1e-3`; weight decay `1e-4`; batch size `128`; maximum `15` epochs; early stopping patience `4`; checkpoint selected by **validation macro-F1**. Results are aggregated over random seeds **42, 7, and 21**. The official test split is reserved for final evaluation.

## Results

Test-set performance (mean ± sample standard deviation across three seeds):

| Model | Macro-F1 ↑ | Accuracy ↑ |
| --- | ---: | ---: |
| MLP | 0.6855 ± 0.0345 | 0.7294 ± 0.0298 |
| CNN | 0.8577 ± 0.0374 | 0.8764 ± 0.0322 |
| **CNN + augmentation** | **0.8876 ± 0.0285** | **0.9005 ± 0.0238** |

Compared with the unaugmented CNN, mild augmentation improved mean **macro-F1 by 0.0299** and mean **accuracy by 0.0241** (absolute differences). However, the improvement was **not uniform across random seeds or cell classes**. These findings describe this experimental setup; three runs are insufficient to claim a statistically significant or clinically generalizable improvement.

## Repository structure

```text
BloodMNIST-CNN-Augmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── bloodmnist.ipynb
├── lectrace/
    ├── bloodmnist_lecture.py
    └── lectrace.toml

```


## Getting started

**Requirements:** Python 3.11 is a tested local choice for the LecTrace demonstration. Install the project dependencies in an isolated environment:

```bash
python -m pip install -r requirements.txt
```

**Run the experiment:** Open `bloodmnist.ipynb` in Jupyter or Google Colab and execute the cells in order. The notebook downloads BloodMNIST automatically. Training the three models over three seeds may take time, especially on CPU. In Colab, install any missing packages if prompted.

**Run the interactive LecTrace explanation:** From the repository root, enter the `lectrace/` directory first:

```bash
cd lectrace
lectrace run bloodmnist_lecture.py
lectrace serve bloodmnist_lecture.py --port 8766
```

Then open `http://localhost:8766` in a browser. The lecture traces one example and one optimizer update, illustrating **data input → augmentation → CNN forward pass → loss and gradient → weight update → prediction**. **It is a teaching demonstration, not the fully trained evaluation pipeline.** Predictions and probabilities in the LecTrace walkthrough should not be interpreted as the final test-set performance.

> **Note:** LecTrace may require its own compatible dependency versions. A local setup that successfully ran the demonstration used Python 3.11, PyTorch 2.2.2, and NumPy 1.26.4. If you use a different environment, check PyTorch/NumPy compatibility.

## Interpretation and limitations

- Augmentation improved **average** performance in this controlled comparison, but some individual class recalls and random-seed results decreased.
- Horizontal flipping and rotation were combined, so this study does not isolate the effect of each transformation.
- The experiment uses one CNN architecture and one mild augmentation setting; broader hyperparameter and architecture comparisons are future work.
- Results are based on a single public dataset without external clinical validation. The project does **not** establish clinical readiness.

## Reproducibility and AI assistance

The notebook contains the model definitions, preprocessing, training, evaluation, and visualizations. Generative AI tools assisted with coding discussions, debugging, documentation, and preparation of the LecTrace explanation; the reported results come from executing and checking the project code. Please consult the accompanying course report for the full experimental analysis and references.

## Dataset citation

Yang, J. *et al.* **MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.** *Scientific Data* (2023). See [MedMNIST](https://medmnist.com/) for dataset details, citation guidance, and usage terms.

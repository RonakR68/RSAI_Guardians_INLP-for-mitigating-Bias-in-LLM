# RSAI Guardians: Mitigating Bias Using Iterative Nullspace Projection (INLP)

## Overview

This repository contains code to measure stereotypical bias in language models using the Stereoset dataset and debias them using Iterative Nullspace Projection (INLP). It then evaluates the effectiveness of INLP in mitigating bias by comparing biased and debiased models.

1. **Clone the Repository:**

    ```bash
    git clone <repository_url>
    ```

2. **Change to Working Directory:**

    ```bash
    cd RSAI_Guardians_INLP-for-mitigating-Bias-in-LLM
    ```

3. **Download Wikipedia 2.5 Dataset:**

    Download the Wikipedia 2.5 dataset from [this link](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) and place it in the `data/text` directory.

4. **Run the Notebook:**

    Open and run the notebook `rsai_inlp_main.ipynb` in your Jupyter Notebook environment. This notebook contains the code to measure stereotypical bias, debias the models using INLP, and analyze the effectiveness of debiasing.

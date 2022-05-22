# ML-Challenge-2022

## Contents
- Colab jupyter notebook for submission using whole data set
- Colab jupyter notebook for evaluating our model
- Local jupyter notebook for evaluating our model
- Final weights of our model 
- Final kaggle submission in a CSV format

This repository includes three jupyter notebooks two for the evaluation of our model on the given training set (one optimized for Colab, one for local execution).
The notebook for local execution as well as the python code wrapped in modules can be found in the folder [local_execution](./local_execution).

## Grading

For grading we recommend to use the notebook [colab_experimental_notebook.ipynb](colab_experimental_notebook.ipynb). This notebook can be uploaded to Google colab and executed from start to finish without having to mount any additional volumes. In there all the relevant visualisations are produced and the individual steps are well described.

If you want to validate that our model also produced the kaggle submission, we suggest to use the  [colab_submission_notebook.ipynb](colab_submission_notebook.ipynb) notebook. This script will load the model weights from the specified file. Therefore it is important, that the final_model_weights.pth file is uploaded to colab along with the notebook. The model weights need to be moved to the folder `models` which is created at runtime.

If you want to see how we nicely modularized our code and made use of relative imports, you should check out the local notebook.

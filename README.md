AI Generated Text Detection
This repository contains the source code and resources for a binary classification project aimed at detecting AI-generated texts. The project is based on a Kaggle competition and utilizes a variety of classical machine learning models as well as a fine-tuned DistilRoBERTa model to achieve its goal.

Project Structure
data/: Contains pre-processed and post-processed training and test datasets in CSV format. Training datasets can be augmented with custom-generated synthetic data. The test set provided is a placeholder and should be replaced for actual use.
model_checkpoints/: Stores trained models' checkpoints.
EDA.ipynb: Jupyter notebook for exploratory data analysis on the training set.
generate_synthetic_essays.ipynb: Notebook for generating synthetic training data using Mistral-7b instruct.
data_processing.py: Processes the training and test sets, tokenizes and vectorizes texts, and saves the resulting sparse matrices as NPZ files in the data/ folder.
optuna.ipynb: Contains hyperparameter optimization for classical ML models (Ridge, Multinomial Naive Bayes, SVM, and XGBoost) and visualizations of optimization history and parameter importance.
classical_models_training.py: Trains the four classical ML models and saves them as .pkl files in the model_checkpoints/ folder.
distilroberta_training.py: Fine-tunes the pre-trained DistilRoBERTa-base model on the training set and saves the checkpoint to the model_checkpoints/ folder.
inference.py: Loads trained classical ML models and DistilRoBERTa, ensembles them using weights to make predictions on the test set.
pseudo_labeling.py: Implements advanced pseudo-labeling techniques to leverage accurate predictions for accuracy improvement.
Setup
Clone the Repository:

sh
Copy code
git clone https://github.com/<your-username>/ai-text-detection.git
Install Dependencies:
Ensure you have Python 3.x installed and then install the required packages:

sh
Copy code
pip install -r requirements.txt
Note: You might need to create a requirements.txt file listing all the necessary packages.

Running the Project
Perform EDA by opening EDA.ipynb in Jupyter Notebook or JupyterLab.
Generate Synthetic Training Data using generate_synthetic_essays.ipynb.
Pre-process the Data with python data_processing.py.
Conduct Hyperparameter Optimization with optuna.ipynb.
Train Classical Models and DistilRoBERTa using classical_models_training.py and distilroberta_training.py, respectively.
Execute Inference with python inference.py.

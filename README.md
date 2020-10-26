# ACM_MM_2020_Finance
The code for our submission titled  **Multimodal Multi-Task Financial Risk Forecasting** [[Paper](https://dl.acm.org/doi/10.1145/3394171.3413752)] which got accepted for **oral** presentation at **ACM Multimedia 2020**

### Repository Overview

The code consists of two main sections: 
- **`feature_extraction`** - Contains code to extract all the features for the pipeline, is subdivided into three folders:
    - `Text ` - Has the script `text_feature_extraction.py` to extract textual features
    - `Audio` - Has the script `audio_feature_extraction.py` to extract audio features
    - `Financial` - Has the script `financial_feature_extraction.py` to extract financial features
- **`models`** - Contains all the code of all the models in the pipeline, namely
  - `bilstm_reg.py` - The text bilstm model for regression
  - `aligned_audio_reg.py` - The aligned audio model for regression
  - `SVR.py` - The SVR on financial features for regression
  - `ensemble_reg.py` - The final ensemble of the text, audio and financial pipeline for regression
  - `bilstm_clf.py` - The text bilstm model for classification
  - `aligned_audio_clf.py` - The aligned audio model for classification
  - `SVC.py` - The SVR on financial features for classification
  - `ensemble_clf.py` - The final ensemble of the text, audio and financial pipeline for classification

## Running the code
Run the following two bash scripts to execute the pipeline. **Note :** **You may need to run it in sudo mode to grant it access to automatically make some folders for storing checkpoints etc**


- **`feature_extraction.sh`** - Bash script to run the entire feature extraction pipeline
- **`run_model.sh`** - Bash script to run the models
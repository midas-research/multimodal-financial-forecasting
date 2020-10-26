# ACM_MM_2020_Finance
The code for our submission titled  **Multimodal Multi-Task Financial Risk Forecasting** [[Paper](https://dl.acm.org/doi/10.1145/3394171.3413752)] which got accepted for **oral** presentation at **ACM Multimedia 2020**

## Cite
If our work was helpful in your research, please kindly cite this work:
```
@inproceedings{sawhneymultimodal2020,
author = {Sawhney, Ramit and Mathur, Puneet and Mangal, Ayush and Khanna, Piyush and Shah, Rajiv Ratn and Zimmermann, Roger},
title = {Multimodal Multi-Task Financial Risk Forecasting},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3394171.3413752},
doi = {10.1145/3394171.3413752},
abstract = {Stock price movement and volatility prediction aim to predict stocks' future trends to help investors make sound investment decisions and model financial risk. Companies' earnings calls are a rich, underexplored source of multimodal information for financial forecasting. However, existing fintech solutions are not optimized towards harnessing the interplay between the multimodal verbal and vocal cues in earnings calls. In this work, we present a multi-task solution that utilizes domain specialized textual features and audio attentive alignment for predictive financial risk and price modeling. Our method advances existing solutions in two aspects: 1) tailoring a deep multimodal text-audio attention model, 2) optimizing volatility, and price movement prediction in a multi-task ensemble formulation. Through quantitative and qualitative analyses, we show the effectiveness of our deep multimodal approach.},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
pages = {456–465},
numpages = {10},
keywords = {multi-task learning, stock prediction, finance, speech processing},
location = {Seattle, WA, USA},
series = {MM '20}
}
```

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

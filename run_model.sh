# !/bin/bash
echo "Training Regression Text BiLSTM"
echo "--------"
python3 models/bilstm_reg.py
echo "Finished Training Regression Text BiLSTM"
echo "--------"
echo "Training Regression Aligned Audio model"
echo "--------"
python3 models/aligned_audio_reg.py
echo "Finished Training Regression Aligned Audio model"
echo "--------"
echo "Training Regression Financial SVR"
echo "--------"
python3 models/SVR.py
echo "Finished Training Regression Financial SVR"
echo "--------"
echo "Training Regression Ensemble"
echo "--------"
python3 models/ensemble_reg.py
echo "Finished Training Regression Ensemble"
echo "--------"
echo "Training Classification Text BiLSTM"
echo "--------"
python3 models/bilstm_clf.py
echo "Finished Training Classification Text BiLSTM"
echo "--------"
echo "Training Classification Aligned Audio model"
echo "--------"
python3 models/aligned_audio_clf.py
echo "Finished Training Classification Aligned Audio model"
echo "--------"
echo "Training Classification Financial SVC"
echo "--------"
python3 models/SVC.py
echo "Finished Training Classification Financial SVC"
echo "--------"
echo "Training Classification Ensemble"
echo "--------"
python3 models/ensemble_clf.py
echo "Finsihed Training Classification Ensemble"
echo "--------"
echo "Finished Training !"
echo "--------"

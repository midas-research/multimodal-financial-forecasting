# !/bin/bash
echo "Extracting Text features"
echo "--------"
python3 feature_extraction/Text/text_feature_extraction.py
echo "Done extracting Text features"
echo "--------"
echo "Extracting Audio features"
echo "--------"
python3 feature_extraction/Audio/audio_feature_extraction.py
echo "Done extracting Audio features"
echo "--------"
echo "Extracting Financial features"
echo "--------"
python3 feature_extraction/Financial/finance_feature_extraction.py
echo "Done extracting Financial features"
echo "--------"
echo "Done extracting features!!!"
echo "--------"
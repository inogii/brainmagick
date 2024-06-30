conda create -n i ipython python=3.8 -y
conda activate i
conda install pytorch torchaudio cudatoolkit=11.3 -c pytorch -y
pip install -U -r requirements.txt
pip install -e .
# For the Broderick dataset, you will further need the following model.
python -m spacy download en_core_web_md
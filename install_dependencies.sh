# pip install git+https://github.com/liaad/kep --timeout=10000
# pip install git+https://github.com/boudinfl/pke --timeout=10000
# pip install git+https://github.com/LIAAD/yake.git --timeout=10000
# install requirements file
pip install -r requirements.txt
python -m nltk.downloader stopwords --timeout=10000
python -m spacy download en_core_web_lg --timeout=10000
python -m spacy download en --timeout=10000

# uncomment build trec_eval not necessary as trec_eval executable is atatched with code and below command moves it to ur /usr/local/bin
# mkdir temp_
# cd temp_
# git clone https://github.com/usnistgov/trec_eval.git
# cd trec_eval
# # replace BIN variable which is path to binary where trec_eval should be installed
# bin_path=$(which pip | sed 's+/pip++g')
# sed -i "s+BIN = /usr/local/bin+BIN = $n+g" Makefile
# make install
# cd ../../
# rm -rf temp_


sudo cp trec_eval /usr/local/bin/

# uncoment below 4 lines only when models are not present in embedding folder
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&confirm=r8GA&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM" -O GoogleNews-vectors-negative300.bin.gz && rm -rf /tmp/cookies.txt

# wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/u/0/uc?export=download&confirm=r8GA&id=0B6VhzidiLvjSOWdGM0tOX1lUNEk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B6VhzidiLvjSOWdGM0tOX1lUNEk" -O torontobooks_unigrams.bin && rm -rf /tmp/cookies.txt

# mv torontobooks_unigrams.bin src/main/embedding/
mv GoogleNews-vectors-negative300.bin.gz src/main/embedding/

git clone https://github.com/epfml/sent2vec
cd sent2vec
git checkout f827d014a473aa22b2fef28d9e29211d50808d48
make
pip install cython
cd src
python setup.py build_ext
pip install .
cd ../../
rm sent2vec


# cd src/main/evaluation
# unzip en_kp_list.zip
# cd ../../
# download models
# mkdir data

# cd data
# mkdir Models
# mkdir Models/Unsupervised
# mkdir Models/Unsupervised/lda
# wget http://www.ccc.ipt.pt/~ricardo/kep/data.zip
# unzip data.zip
# rm data.zip

# mkdir Datasets
# cd Datasets
# wget https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/SemEval2017.zip
# unzip SemEval2017.zip
# rm SemEval2017.zip

# wget https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/SemEval2010.zip
# unzip SemEval2010.zip
# rm SemEval2010.zip

# wget https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip
# unzip Inspec.zip
# rm Inspec.zip
# cd ../../
# unzip keyphrase_expansion_data.zip
# mv keyphrase_expansion src/data/Datasets/

# mv SemEval2010_lda.gz src/data/Models/Unsupervised/lda/
# mv Inspec_lda.gz src/data/Models/Unsupervised/lda/
# mv SemEval2017_lda.gz src/data/Models/Unsupervised/lda/
python -m spacy download en
# python train_lda.py
# mv keyphrase_expansion_lda.gz src/data/Models/Unsupervised/lda/


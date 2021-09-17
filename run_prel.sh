pip install numpy
pip install pandas
pip install konlpy
pip install gensim==4.0.1
CUDA_VISIBLE_DEVICES="0" python3 /home/work/model/main.py --mode prel \
                                  --train_data_file /home/work/data/ddata.wst.dev \
                                  --img_emb_data /home/work/model/model_file/imgEmbDataSet.json \
                                  --stop_word /home/work/model/model_file/stopword.txt \
                                  --model_path /home/work/model/model_file/dec_model_state_dict \
                                  --res_size 600 \
                                  --emb_size 300 \
                                  --img_emb_size 1500 \
                                  --key_size 600 \
                                  --max_story_size 30 \
                                  --max_sentence_len 30 \
                                  --hops 3 \
                                  --batch_size 100 \
                                  --epoch 10 \
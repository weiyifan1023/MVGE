# MVGE

[Multi-View Graph Representation Learning for Answering Hybrid Numerical Reasoning Question](https://scholar.google.com.hk/citations?view_op=view_citation&hl=zh-CN&user=Kmp8kVMAAAAJ&citation_for_view=Kmp8kVMAAAAJ:u5HHmVD_uO8C)

__Please kindly cite our work if you find our codes useful, thank you.__
```bash
@misc{wei2023multiview,
      title={Multi-View Graph Representation Learning for Answering Hybrid Numerical Reasoning Question}, 
      author={Yifan Wei and Fangyu Lei and Yuanzhe Zhang and Jun Zhao and Kang Liu},
      year={2023},
      eprint={2305.03458},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



## requirements

To create an environment with `conda` and activate it.

```bash
conda create -n reghnt python==3.7
conda activate reghnt
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html     # Adjust according to your CUDA version
pip install allennlp==0.8.4 transformers==4.21.1 nltk==3.5 pandas==1.1.5 numpy==1.21.6
conda install -c dglteam dgl-cuda11.1==0.6.1    # Adjust according your CUDA version
pip install sentencepiece
```
Next, you should install `torch-scatter==2.0.5` (python3.7 CUDA11.1) by
__Download [torch-scatter wheel](https://data.pyg.org/whl/torch-1.7.0%2Bcu110/torch_scatter-2.0.5-cp37-cp37m-win_amd64.whl).__ (already existed)

__Or download [other version](https://pytorch-geometric.com/whl/) according to your Python, Pytorch and CUDA version. Then move it to `RegHNT/`.__
```
pip install torch_scatter-2.0.5-cp37-cp37m-linux_x86_64.whl
```

We adopt `RoBERTa` as our encoder to develop our MVGE and use the following commands to prepare RoBERTa model
```bash
cd dataset_reghnt
mkdir roberta.large && cd roberta.large
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
```

## Training

### Preprocessing dataset
We use the preprocessed data by [TagOp Model](https://github.com/NExTplusplus/tat-qa) and they are already in this repository.

### Prepare dataset

```bash
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/MVGE python mvge/prepare_dataset.py --mode [train/dev/test]
```

Note: The result will be written into the folder `./cache` default.

### Train
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python mvge/trainer.py --data_dir ./cache/ \
--save_dir ./try --batch_size 48 --eval_batch_size 1 --max_epoch 100 --warmup 0.06 --optimizer adam --learning_rate 1e-4 \
--weight_decay 5e-5 --seed 42 --gradient_accumulation_steps 12 --bert_learning_rate 1e-5 --bert_weight_decay 0.01 \
--log_per_updates 50 --eps 1e-6 --encoder roberta_large --roberta_model dataset_reghnt/roberta.large
```

## Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python mvge/predictor.py --data_dir ./cache/ \
--test_data_dir ./cache/ --save_dir mvge --eval_batch_size 1 --model_path ./try \
--encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode dev
```
```
python tatqa_eval.py --gold_path=dataset_reghnt/tatqa_dataset_dev.json --pred_path=mvge/pred_result_on_dev.json
```

## Testing
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$(pwd) python mvge/predictor.py \
--data_dir ./cache/ --test_data_dir ./cache/ --save_dir mvge \
--eval_batch_size 1 --model_path ./try --encoder roberta_large --roberta_model dataset_reghnt/roberta.large --mode test
```

Note: The training process may take around 3 days using a single 24GB RTX3090.



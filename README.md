# MUGE Multimodal Retrieval Baseline

**Update: We have released a much more competitive MUGE retrieval baseline in our [Chinese-CLIP repo](https://github.com/OFA-Sys/Chinese-CLIP). More model scales and the technique report have all been released. We strongly recommend to try it!**

This repo is implemented based on the **[open_clip project](https://github.com/mlfoundations/open_clip)**, with modifications to adapt to the Chinese Multimodal Retrieval task

## Requirements and Installation
This repo is successfully tested on the following environment:

* python == 3.6.4
* pytorch == 1.7.1
* CUDA Version == 10.2

To install the requirements, run the following command:

```
pip install -r requirements.txt
```

For other CUDA versions (9.2, 10.1, 11.0), please refer to this [guide](https://pytorch.org/get-started/previous-versions/#linux-and-windows-7) on official Pytorch website and edit the `requirements.txt` to correctly install the compatible version of `torch` and `torchvision`.

## Getting Started

Assume the downloaded dataset and downloaded pretrained weights are placed under this directory `${DATAPATH}`. The following experiment is performed on a single server with 8 V100-16G GPUs.

### Prepare CLIP and BERT Weights

In this repo, we build a [CLIP](https://arxiv.org/abs/2103.00020) model and employ pretrained Openai ViT-B-16 ([download](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)) and Chinese RoBERTa (ymcui's [project](https://github.com/ymcui/Chinese-BERT-wwm), [download](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view?usp=drive_open)) weights to initialize the image-side and text-side, respectively.

For ViT-B-16 weight, run the following command to transform the checkpoint format from a JIT-model to state_dict:
```
python src/preprocess/transform_openai_pretrain_weights.py \ 
    --raw-ckpt-path ${DATAPATH}/ViT-B-16.pt \
    --new-ckpt-path ${DATAPATH}/ViT-B-16.state_dict.pt
```

For RoBERTa weight, unzip the downloaded zipfile and place the `pytorch_model.bin` under the `${DATAPATH}`.


### Prepare the Transformed Images

The images need to be transformed to feed into the CLIP model. However, online transformation during training and inference is slow. Here we perform the image transformation before the experiment.

```
python src/preprocess/transform_images.py \ 
    --data_dir ${DATAPATH} \
    --image_resolution 224
```

The transformed image dataset costs around 100G disk space.

### Training

```
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -u src/training/main.py \
    --save-frequency 1 \
    --train-data="${DATAPATH}/train_queries.jsonl"  \
    --train-img="${DATAPATH}/train_imgs.224.npz"  \
    --val-data="${DATAPATH}/valid_queries.jsonl"  \
    --val-img="${DATAPATH}/valid_imgs.224.npz"  \
    --clip-weight-path="${DATAPATH}/ViT-B-16.state_dict.pt" \
    --bert-weight-path="${DATAPATH}/pytorch_model.bin" \
    --warmup 500 \
    --batch-size=32 \
    --lr=8e-5 \
    --wd=0.001 \
    --epochs=10 \
    --model ViT-B-16
```

The training will cost a few hours. The log and checkpoint files will be saved under the `logs` directory.

**Caution**: Since the training convergence and stablility of in-batch contrastive learning are highly dependent on the global batch-size. If you use a much smaller batch-size than the default setting (32 per-GPU \* 8 GPU), please try to use a smaller learning rate to avoid training divergence. (related [issue](https://github.com/MUGE-2021/image-retrieval-baseline/issues/1)). We recommend to use more GPUs and larger global batch-size to achieve more stable convergence and better model performance.

### Inference and Evaluation

Run the following command to compute image and query features using the trained CLIP model:

```
# only supports single-GPU inference
export CUDA_VISIBLE_DEVICES=0

python -u src/eval/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/test_imgs.224.npz" \
    --text-data="${DATAPATH}/test_queries.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --resume="logs/${experiment_name}/checkpoints/epoch_5.pt" \
    --model ViT-B-16
```

After obtaining the testing features, run the following command to perform kNN search to generate top-10 prediction jsonl file:
```
python -u src/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/test_imgs.224.img_feat.jsonl" \
    --text-feats="${DATAPATH}/test_queries.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/test_predictions.jsonl"
```

The jsonl file can be submitted to MUGE challenge site. In expectation, the evaluated model will get a **mean-recall of around 50**. We strongly believe the baseline can be easily tuned and improved to achieve much better points :)

We also provide the evaluation script to evaluate model's mean-recall on validation set. Run the following command:
```
python src/eval/evaluation.py valid_queries.jsonl valid_predictions.jsonl output.json
```
The score will be saved in `output.json`. The script is the same as the MUGE evaluation server.

## Reference
```
@inproceedings{M6,
  author    = {Junyang Lin and
               Rui Men and
               An Yang and
               Chang Zhou and
               Ming Ding and
               Yichang Zhang and
               Peng Wang and
               Ang Wang and
               Le Jiang and
               Xianyan Jia and
               Jie Zhang and
               Jianwei Zhang and
               Xu Zou and
               Zhikang Li and
               Xiaodong Deng and
               Jie Liu and
               Jinbao Xue and
               Huiling Zhou and
               Jianxin Ma and
               Jin Yu and
               Yong Li and
               Wei Lin and
               Jingren Zhou and
               Jie Tang and
               Hongxia Yang},
  title     = {{M6:} {A} Chinese Multimodal Pretrainer},
  year      = {2021},
  booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining},
  pages     = {3251–3261},
  numpages  = {11},
  location  = {Virtual Event, Singapore},
}

@article{M6-T,
  author    = {An Yang and
               Junyang Lin and
               Rui Men and
               Chang Zhou and
               Le Jiang and
               Xianyan Jia and
               Ang Wang and
               Jie Zhang and
               Jiamang Wang and
               Yong Li and
               Di Zhang and
               Wei Lin and
               Lin Qu and
               Jingren Zhou and
               Hongxia Yang},
  title     = {{M6-T:} Exploring Sparse Expert Models and Beyond},
  journal   = {CoRR},
  volume    = {abs/2105.15082},
  year      = {2021}
}

@software{ilharco_gabriel_2021_5143773,
  author       = {Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig},
  title        = {OpenCLIP},
  month        = jul,
  year         = 2021,
  note         = {If you use this software, please cite it as below.},
  publisher    = {Zenodo},
  version      = {0.1},
  doi          = {10.5281/zenodo.5143773},
  url          = {https://doi.org/10.5281/zenodo.5143773}
}

@inproceedings{Radford2021LearningTV,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

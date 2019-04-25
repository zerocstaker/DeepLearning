# Deep Learning Project: Self-Critical sequence Training for Image Captioning

This project is based on the idea of [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563), [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998), and [Attention is all you need](https://arxiv.org/abs/1706.03762).

The code is based on ruotianluo's unofficial implementation(https://github.com/ruotianluo/self-critical.pytorch), which includes implementation of the baselines.

## Instructions

The instruction from the unofficial implementation is a little complicated, so here are the instructions:

1. ### Download preprocessed coco captions from [Karpathy's homepage](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and put it into `data/` folder.

2. ### Download MSCOCO data (2014 training and 2014 eval), and replace one corrupted image from this [link](https://github.com/karpathy/neuraltalk2/issues/4).

3. ### Download image embeddings:

   1. For Att2in model, download Resnet50,resnet101, or resnet152 [with this link](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM).
   2. For Topdown: follow this [link](https://github.com/peteanderson80/bottom-up-attention)

4. ### Preprocessing:

   1. Preprocess labels with the following script:

      `python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk`

   2. Preprocess image embeddings

      1. For Att2in model, run the following:

         `python scripts/prepro_feats.py --input_json data/dataset_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT`

         Note that this will result in files very large

      2. For topdown: run the following:

         `python script/make_bu_data.py --output_dir data/cocobu`
   3. Preprocess ngram for evaluating cider
         `python scripts/prepro_ngrams.py --input_json .../dataset_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train`

5. ### Training

   1. Att2in:
      1. `python train.py --id fc --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_fc --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30`
      2. Self critical: `python train.py --id fc --caption_model fc --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-5 --start_from log_fc --checkpoint_path log_fc_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30`
   2. Topdown: the same as Att2in except change the first two parameters to 'topdown'
   3. Transformer: run the script `./transformer.sh` 
6. ### Evaluation
   1. To get BLEU, CIDEr, ROUGEL scores, run `python eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 `
   2. To see the captions for a specific set of images in an image folder, run
      `$ python eval.py --model model.pth --infos_path infos.pkl --image_folder path_to_images --num_images 10`
      This will create "vis.json" in 'vis' folder, and in order to see it, run

      ```
      $ cd vis
      $ python -m SimpleHTTPServer
      ```

6. ### Model
Our model can be downloaded here:https://drive.google.com/open?id=1DMnnM1rKyf_ArUsTngGD9cS_501DuDAM


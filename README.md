# data2text-plan-py

This repo contains code for [Data-to-Text Generation with Content Selection and Planning](https://arxiv.org/abs/1809.00582) (Puduppully, R., Dong, L., & Lapata, M.; AAAI 2019); this code is based on an earlier fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). The Pytorch version is 0.3.1.

## Requirements

All dependencies can be installed via:

```bash
pip install -r requirements.txt
```
Note that the Pytorch version is 0.3.1 and Python version is 2.7.
The path to Pytorch wheel in ```requirements.txt``` is configured with CUDA 8.0. You may change it to the desired CUDA version.

## Dataset

The boxscore-data json files can be downloaded from the [boxscore-data repo](https://github.com/harvardnlp/boxscore-data).

The input dataset for data2text-plan-py can be created by running the script ```create_dataset.py``` in ```scripts``` folder.
The dataset so obtained is available at link https://drive.google.com/open?id=1R_82ifGiybHKuXnVnC8JhBTW8BAkdwek

## Preprocessing
Assuming the OpenNMT-py input files reside at `~/boxscore-data`, the following command will preprocess the data

```
BASE=~/boxscore-data
IDENTIFIER=cc

mkdir $BASE/preprocess
python preprocess.py -train_src1 $BASE/rotowire/src_train.txt -train_tgt1 $BASE/rotowire/train_content_plan.txt -train_src2 $BASE/rotowire/inter/train_content_plan.txt -train_tgt2 $BASE/rotowire/tgt_train.txt -valid_src1 $BASE/rotowire/src_valid.txt -valid_tgt1 $BASE/rotowire/valid_content_plan.txt -valid_src2 $BASE/rotowire/inter/valid_content_plan.txt -valid_tgt2 $BASE/rotowire/tgt_valid.txt -save_data $BASE/preprocess/roto -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -train_ptr $BASE/rotowire/train-roto-ptrs.txt
```

The train-roto-ptrs.txt file is available along with the dataset and can also be created by the following command
```
python data_utils.py -mode ptrs -input_path $BASE/rotowire/train.json -train_content_plan $BASE/rotowire/inter/train_content_plan.txt -output_fi $BASE/rotowire/train-roto-ptrs.txt
```

## Training (and Downloading Trained Models)
The command for training the Neural Content Planning model with conditional copy NCP+CC is as follows:
```
BASE=~/boxscore-data
IDENTIFIER=cc

python train.py -data $BASE/preprocess/roto -save_model $BASE/gen_model/$IDENTIFIER/roto -encoder_type1 mean -decoder_type1 pointer -enc_layers1 1 -dec_layers1 1 -encoder_type2 brnn -decoder_type2 rnn -enc_layers2 2 -dec_layers2 2 -batch_size 5 -feat_merge mlp -feat_vec_size 600 -word_vec_size 600 -rnn_size 600 -seed 1234 -start_checkpoint_at 4 -epochs 25 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -report_every 100 -copy_attn -truncated_decoder 100 -gpuid $GPUID -attn_hidden 64 -reuse_copy_attn -start_decay_at 4 -learning_rate_decay 0.97 -valid_batch_size 5
```
The NCP+CC model can be downloaded from  https://www.dropbox.com/sh/vo5wb2fuq7m0bk0/AABikW0KomOKIor24wD8VSFWa?dl=0

## Generation
During inference, we first generate the content plan

```
MODEL_PATH=<path to model1>

python translate.py -model $MODEL_PATH -src1 $BASE/rotowire/inf_src_valid.txt -output $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt -batch_size 10 -max_length 80 -gpu $GPUID -min_length 35 -stage1 
```

This script generates the content plan with records from input of content plan with indices
```
python scripts/create_content_plan_from_index.py $BASE/rotowire/inf_src_valid.txt $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.h5-tuples.txt  $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.txt
```

The accuracy of content plan in first stage can be evaluated using the following command
```
python non_rg_metrics.py $BASE/transform_gen/roto-gold-val-beam5_gens.h5-tuples.txt $BASE/transform_gen/roto_stage1_$IDENTIFIER-beam5_gens.h5-tuples.txt 
```

The output summary is generated using the command
```
MODEL_PATH2=<path to model2>

python translate.py -model $MODEL_PATH -model2 $MODEL_PATH2 -src1 $BASE/rotowire/inf_src_valid.txt -tgt1 $BASE/gen/roto_stage1_$IDENTIFIER-beam5_gens.txt -src2 $BASE/gen/roto_stage1_inter_$IDENTIFIER-beam5_gens.txt -output $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.txt -batch_size 10 -max_length 850 -min_length 150 -gpu $GPUID
```

## Automatic evaluation using IE metrics
Metrics of RG, CS, CO are computed using the below commands.
```
python data_utils.py -mode prep_gen_data -gen_fi $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.txt -dict_pfx "roto-ie" -output_fi $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 -input_path "/boxcore-json/rotowire"

th extractor.lua -gpuid  $GPUID -datafile roto-ie.h5 -preddata $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5 -dict_pfx "roto-ie" -just_eval

python non_rg_metrics.py $BASE/transform_gen/roto-gold-val-beam5_gens.h5-tuples.txt $BASE/transform_gen/roto_stage2_$IDENTIFIER-beam5_gens.h5-tuples.txt 
```

## Evaluation using BLEU script
The BLEU perl script can be obtained from  https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl
Command to compute BLEU score:
```
~/multi-bleu.perl $BASE/rotowire/inf_tgt_valid.txt < $BASE/gen/roto_stage2_$IDENTIFIER-beam5_gens.txt
```

## IE models
For training the IE models, follow the updated code in https://github.com/ratishsp/data2text-1 which contains bug fixes for number handling. The repo contains the downloadable links for IE models too.

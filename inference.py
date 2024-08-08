
import json
from transformers import T5Config
from MSA import MSAT5
from typing import List, Tuple
import torch
import torch
import os 
import logging
import argparse
from msadata import MSAInferenceDataSet, MSABatchConverter, Alphabet
import time

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
  
logger.setLevel(logging.INFO)


def msa_generate(args, model, dataset, msa_collator, tokenizer):
    """
    Generate msa for given dataset
    """
    with torch.no_grad():
        output_dir = os.path.join(args.output_dir, args.mode, f"A{args.augmentation_times}T{args.trials_times}R{args.repetition_penalty}")
        args_dict = vars(args)
        os.makedirs(output_dir,exist_ok=True)
        with open(os.path.join(output_dir,'params.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)
        logger.info('generate src files-num: {}'.format(len(dataset)))
       
        for protein_data in dataset:
            infer_time_avg = 0.0
            
            msa_name = protein_data['name']
            original_seq = protein_data['seq']
            esm = protein_data['emb']
            input_ids = [original_seq, esm]
            esm, src_ids = msa_collator.infer_batch_convert(input_ids, args.num_alignments)
            esm = esm.to(args.device)
            src_ids = src_ids.to(args.device)
            
            _, original_seq_num, original_seq_len = src_ids.size()
            for trial in range(args.trials_times):
                msa_name = os.path.basename(msa_name).split('.')[0]
                msa_output_dir = os.path.join(output_dir,msa_name)
                os.makedirs(msa_output_dir,exist_ok=True)
                a3m_file_name = os.path.join(msa_output_dir,f"generation_{trial}.a3m")
                if os.path.exists(a3m_file_name):
                    logger.info(f'File {a3m_file_name} already exists, skip')
                    continue
                start = time.time()
                output = model.generate(src_ids, esm, do_sample=True, top_k=5, top_p=0.95, repetition_penalty=args.repetition_penalty, \
                                    max_length=original_seq_len+1, gen_seq_num=original_seq_num*args.augmentation_times) # TODO
                end = time.time()
                infer_time_avg += (end - start) / args.trials_times
                generate_seq = [tokenizer.decode(seq_token, skip_special_tokens=True).replace(' ','') for seq_token in output[0]]
                generate_seq = list(filter(lambda x: len(set(x)) >= 4 and len(x) == len(original_seq), generate_seq))
                # generate_seq = list(filter(lambda x: len(set(x)) >= 4 and len(x) == len(msa[0][1]), generate_seq)) # filter our sequences with all gap '-'
                with open(a3m_file_name,'w') as fw:
                    generate_msa_list = []
        
                    generate_msa_list.append('>' + msa_name)
                    generate_msa_list.append(original_seq)
                    for i, seq in enumerate(generate_seq):
                        seq_name = 'MSAT5_Generate_condition_on_{}_seq_from_{}_{}'.format(src_ids.size(1),msa_name,i)
                        generate_msa_list.append('>' + seq_name)
                        generate_msa_list.append(seq)
                        
                    fw.write("\n".join(generate_msa_list))
                    logger.info(f'Generate successful for {msa_name} trial: {trial}')
            
            print('avg infer time:', infer_time_avg)

def inference(args):
    config = T5Config.from_pretrained('./config/')
    tokenizer = Alphabet.from_architecture(name="msa_transformer")
    msadata_collator = MSABatchConverter(tokenizer)
    if args.checkpoints:
        logger.info("loading model from {}".format(args.checkpoints))
        model = MSAT5.from_pretrained(args.checkpoints).to(args.device)
    else:
        logger.warning("Loading a random model")
        model = MSAT5(config).to(args.device)
    dataset = MSAInferenceDataSet(args)
    
    msa_generate(
        args,
        model=model,
        msa_collator=msadata_collator,
        dataset=dataset,
        tokenizer=tokenizer
    ) 
        

def parsing_arguments():
    parser=argparse.ArgumentParser()
    # general params
    parser.add_argument('--do_train', action='store_true', help="whether further fine-tune")
    parser.add_argument('--do_predict', action='store_true', help="only create new seqs")
    parser.add_argument('--checkpoints', type=str,\
                        help="the folder path of model checkpoints, e.g '/msat5-base/checkpoint-740000'")
    parser.add_argument('--data_path', type=str, \
                        help="the folder path of poor data eg. './dataset/casp15/cfdb/'")
    parser.add_argument('-o', '--output_dir', type=str, default="./output/", help="the folder path of output files")
    parser.add_argument('--num_alignments', type=int, default=32, help="num alignments to generate")
    parser.add_argument('--device', type=str, default="cpu", help="the inference device")

    # Generation parmas
    parser.add_argument('--mode', type=str, choices=['orphan','artificial'], required=True, help="whether task is real world orphan enhancement or artificial enhancement")
    parser.add_argument('--repetition_penalty', type=float, default=1.2, help="repetition penalty for generation")
    parser.add_argument('-a','--augmentation_times', type=int, default=1, help="times of generated quality compared to original msa x1 x3 x5")
    parser.add_argument('-t', '--trials_times', type=int, default=1)    
    # More advanced generation params
    parser.add_argument('--do_sample', type=bool, default=True, help="Whether or not to use sampling ; use greedy decoding otherwise.")
    # parser.add_argument('--num_beams', type=int, default=36, help="Number of beams for beam search. 1 means no beam search.")
    # parser.add_argument('--num_beam_groups', type=int, default=6, help="Number of groups to divide num_beams into in order to ensure diversity among different groups of beams.")
    parser.add_argument('--diversity_penalty', type=float, default=1.0, help="diversity penalty for generation, ")
    parser.add_argument('--temperature', type=float, default=1.0, help="The value used to modulate the next token probabilities.")
    parser.add_argument('--top_p', type=float, default=1.0, help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.")
    args=parser.parse_args()
    assert not (args.do_train and args.do_predict), "select one mode from train and predict"
    return args
    
if __name__=="__main__":
    args = parsing_arguments()
    args_dict = vars(args)
    args_str = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info('paramerters: %s', args_str)
    inference(args)

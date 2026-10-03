"""Argument parser"""

import argparse


def parse_opt():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    # --------------------------- glo_data path -------------------------#

    parser.add_argument('--data_path', default='/root/autodl-tmp/data',
                        help='path to datasets')
                        
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--model_name', default='./runs/bert_f30kSGR/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--logger_name', default='./runs/bert_f30kSGR/glo',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--bert_path', default='/root/tmp/uncased_L-12_H-768_A-12/',
                        help='path of pre-trained BERT.')
    parser.add_argument('--ft_bert', action='store_false',
                        help='Fine-tune the text encoder.')
    parser.add_argument('--bert_size', default=768, type=int,
                        help='Dimensionality of the text embedding')
    # ----------------------- training setting ----------------------#
    parser.add_argument('--seed',default=114514, type = int,
                        help = 'Random seed Number')
    parser.add_argument('--gpu_id', default=1, type=int, 
                        help='GPU to use.')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--num_epochs', default=50, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=30, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--other_lr', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--bert_lr', default=.00002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of glo_data loader workers.')
    parser.add_argument('--log_step', default=500, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=1000, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--warmup', default=-1, type=float)

    # ------------------------- model setting -----------------------#
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--sim_dim', default=256, type=int,
                        help='Dimensionality of the sim embedding.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--bi_gru', action='store_false',
                        help='Use bidirectional GRU.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--no_txtnorm', action='store_true',
                        help='Do not normalize the text embeddings.')
    parser.add_argument('--module_name', default='SGR', type=str,
                        help='SGR, SAF')
    parser.add_argument('--sgr_step', default=3, type=int,
                        help='Step of the SGR.')
    parser.add_argument('--focal_type', default="glo",
                        help='equal|prob|glo')

    parser.add_argument('--self_regulator', default='coop_rcar',
                        help='only_rar, only_rcr, coop_rcar')
    parser.add_argument('--rcar_step', default=2, type=int,
                        help='step of RCR cooperation with RAR')
    parser.add_argument('--rcr_step', default=2, type=int,
                        help='step of RCR')
    parser.add_argument('--rar_step', default=2, type=int,
                        help='step of RAR')
    parser.add_argument('--agg_func', default="LogSumExp",
                        help='LogSumExp|Mean|Max|Sum')
    parser.add_argument('--attn_type', default='i2t'
                        help='{t2i,i2t}')
    parser.add_argument('--t2i_smooth', default=10.0, type=float,
                        help='The value of t2i softmax lambda')
    parser.add_argument('--i2t_smooth', default=3.0, type=float,
                        help='The value of i2t softmax lambda')

    opt = parser.parse_args()
    print(opt)
    return opt

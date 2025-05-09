import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run KDCDR.")
    parser.add_argument('--weights_path', nargs='?', default='../',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='Data/data',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='./',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='sport_cell',
                        help='Choose a dataset')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size.')
    parser.add_argument('--isconcat',type = int ,default=1,
                        help='does transfer inter domain?')

    parser.add_argument('--regs', type=float, default=0.00001,
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--initial_type',  default='x')
    parser.add_argument('--adj_type', nargs='?', default='norm',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='ngcf',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=1,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[1,5,10,20]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--weight_id', type=float, default=0.1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')
    parser.add_argument('--weight_loss',nargs='?', default='[1.,1.]')
    parser.add_argument('--n_interaction',type=int, default=3)
    parser.add_argument('--neg_num',type=int, default=4)
    parser.add_argument('--connect_type',type=str, default='concat',
                        help='concat or mean')
    parser.add_argument('--layer_fun',type=str, default='gcf',
                        help='feature propagation way')
    parser.add_argument('--fuse_type_in',type=str, default='la2add',
                        help='inter-domain feature fuse type')
    parser.add_argument('--sparcy_flag',type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--lambda_s', type=float, default=0.8)
    parser.add_argument('--lambda_t', type=float, default=0.8)
    parser.add_argument('--t', type=float, default=5)
    parser.add_argument('--link_loss', type=float, default=0.9)
    parser.add_argument('--flag', type=str, default='zhipu')
    parser.add_argument('--galign_gcn_blocks', type=int, default=3)
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.8)
    return parser.parse_args()

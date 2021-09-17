from mevac import *
import argparse


parser = argparse.ArgumentParser(description='AI Fashion Coordinator.')
parser.add_argument('--mode', type=str,
                    default='train',
                    help='train or test or pred mode')
parser.add_argument('--train_data_file', type=str,
                    default='./data/ddata.txt')
parser.add_argument('--test_data_file', type=str,
                    default='./data/ac_eval_t1.dev')
parser.add_argument('--meta_data_file', type=str,
                    default='./data/mdata.txt')
parser.add_argument('--image_path', type=str,
                    default='./data/image/')
parser.add_argument('--epoch', type=int,
                    default=10)
parser.add_argument('--batch_size', type=int,
                    default=20)
parser.add_argument('--emb_size', type=int,
                    default=300)
parser.add_argument('--win_size', type=int,
                    default=3)
parser.add_argument('--img_size', type=tuple,
                    default=(400, 400))
parser.add_argument('--img_emb_size', type=int,
                    default=1500)
parser.add_argument('--key_size', type=int,
                    default=600)
parser.add_argument('--res_size', type=int,
                    default=600)
parser.add_argument('--hops', type=int,
                    default=3)
parser.add_argument('--max_story_size', type=int,
                    default=30)
parser.add_argument('--max_sentence_len', type=int,
                    default=30)
parser.add_argument('--learning_rate', type=float,
                    default=1e-5)
parser.add_argument('--anti_zero', type=float,
                    default=1e+6)
parser.add_argument('--random_seed', type=int,
                    default=225)


args = parser.parse_args()



if __name__ == '__main__':
    mode = args.mode

    if mode == 'prel':
        dia_index_list, dia_code_list, dia_data_list, image_list, dia_tag_list = get_data_list(data_type='train')
        make_embedding_model(dia_data_list, None, args.emb_size, args.win_size)
        embedding_model = load_embedding_model()
        make_img_metaData_file(embedding_model, args.emb_size)
    else:
        mevac = mevac(args)
        if mode == 'train':
            mevac.train()
        elif mode == 'test':
            mevac.test()
        elif mode == 'pred':
            mevac.predict()
        elif mode == 'img_emb_train':
            mevac.img_emb_train()


















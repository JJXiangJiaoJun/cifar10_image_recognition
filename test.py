import os
import mxnet as mx
from model import resnet50_cifar10

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse

import pandas as pd
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from train import get_train_val_dataset
from mxnet.gluon.data.vision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Test for test dataset')

    parser.add_argument('--resume', type=str, default='',
                        help='Resume parameters for net,you can use ./xxxxxxx.params.')
    parser.add_argument('--load-prefix', type=str, default='',
                        help='Prefix when load the parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving .csv file prefix.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size when test.Default is 32')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Use GPUs when testing.you can use such as 1,3 to specify GPUs.')
    parser.add_argument('--hybrid', action='store_true', dest='hybrid',
                        help='Use net.hybridize() to acclerate testing.')

    args = parser.parse_args()

    return args


def get_test_dataset(root='../CIFAR10/data/kaggle_cifar10/'):
    test_filename = 'test'
    test_filepath = os.path.join(root, test_filename)
    test_dataset = ImageFolderDataset(root=test_filepath)
    return test_dataset


def get_test_dataloader(test_dataset, batch_size):
    test_transformer = transforms.Compose([transforms.Resize(128),
                                           transforms.ToTensor(),
                                           # Normalize
                                           transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                std=(0.2023, 0.1994, 0.2010))])

    test_dataloader = DataLoader(test_dataset.transform_first(test_transformer),
                                 batch_size, shuffle=False, last_batch='keep')
    return test_dataloader


def split_and_load_data(batch, ctx_list, batch_size):
    """

    :param batch: 
    :param ctx_list: 
    :param batch_size: 
    :return:
     new_batch:list of NDArray [[data1,data2,data3],[label1,label2,label3]]
        result of split data for each gpus
    """
    num_ctx = len(ctx_list)
    num_sample_pre_batch = batch_size // num_ctx
    # total_batch = batch_size*num_ctx
    new_batch = []
    # split one mini-batch to each ctx

    for i, data in enumerate(batch):
        new_data = []
        for j, ctx in enumerate(ctx_list):
            begin = j * num_sample_pre_batch
            end = min((j + 1) * num_sample_pre_batch, batch_size)
            split_data = mx.nd.slice_axis(data, axis=0, begin=begin, end=end)
            new_data.append(split_data.as_in_context(ctx))
        new_batch.append(new_data)
    return new_batch


def inference(net, train_dataset, test_dataset, test_dataloader, ctx_list, args):
    """Test pipeline"""
    test_preds = []
    print('Testing on ', ctx_list)
    if args.hybrid:
        print('hybridize')
        net.hybridize(static_alloc=True)
    total_batch = len(test_dataset) // args.batch_size + 1
    for i, batch in enumerate(test_dataloader):
        batch_size = len(batch[0])
        batch = split_and_load_data(batch, ctx_list, batch_size)
        preds_label = []
        for data, _ in zip(*batch):
            pred_score = net(data)
            pred_score = mx.nd.argmax(pred_score, axis=-1).astype(int).asnumpy()
            preds_label.extend(pred_score)
        print('Complete {}/{} batch/total'.format(i, total_batch))
        test_preds.extend(preds_label)

    sorted_ids = list(range(1, len(test_dataset) + 1))
    sorted_ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': sorted_ids, 'label': test_preds})
    df['label'] = df['label'].apply(lambda x: train_dataset.synsets[int(x)])

    csv_file_path = args.save_prefix + '_submission.csv'
    csv_file_dir = os.path.dirname(csv_file_path)
    if csv_file_dir and not os.path.exists(csv_file_dir):
        os.makedirs(csv_file_dir)
    print('Write .csv files to {:s}'.format(csv_file_path))
    df.to_csv(csv_file_path, sep=',', index=False)


if __name__ == '__main__':
    args = parse_args()

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx_list = ctx if ctx else [mx.cpu(0)]

    net_name = '_'.join(('resenet50v1', 'cifar10'))
    args.load_prefix += net_name

    # load model
    net = resnet50_cifar10()
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
        params = net.collect_params()
    else:
        net.initialize()
    # reset context
    net.collect_params().reset_ctx(ctx_list)
    # get testdata
    test_dataset = get_test_dataset()
    train_dataset, _ = get_train_val_dataset()

    test_dataloader = get_test_dataloader(test_dataset, args.batch_size)

    inference(net, train_dataset, test_dataset, test_dataloader, ctx_list, args)

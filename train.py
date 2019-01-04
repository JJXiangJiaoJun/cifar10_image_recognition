import os
import argparse

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import mxnet as mx
from mxnet import autograd, nd
from model.ResNetV1b import resnet50_cifar10
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision import ImageFolderDataset
from mxnet.gluon.data import DataLoader
from mxnet.gluon import Trainer
import time
from mxnet.gluon import loss as gloss


def parse_arg():
    parser = argparse.ArgumentParser(description='Scripts use for training model end2end in cifar10.')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size used when training. Default is 4 ')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs.you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs. Default is 50')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume for training,you can use like ./xxxxxx.params to load param.')
    parser.add_argument('--mixup', action='store_true',
                        help='Use mixup data for training.')
    parser.add_argument('--lr', type=str, default='',
                        help='Initial learning rate use for training. Default is 0.01 ')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='Learning rate decay rate.Default is 0.1,New_lr = lr*lr_decay')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epoches at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum,default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay,default is 5e-4.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameters prefix.')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving paramters epoch interval.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validation model epoch interval.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Loging interval for mini-batch.Default is 100.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Start epoch for resuming. Default is 0 for new training')
    parser.add_argument('--last-gamma', action='store_true', dest='last_gamma',
                        help='Set Bottleneck last gamma to zero.')
    parser.add_argument('-v''--verbose', action='store_true', dest='verbose',
                        help='Print some useful information when training.')
    parser.add_argument('--hybrid', action='store_true', dest='hybrid',
                        help='Use net.hybridize() to speed up training.')
    parser.add_argument('--final-drop', type=float, default=0.0,
                        help='Drop out rate use for last layer of net.')
    args = parser.parse_args()
    args.lr = float(args.lr) if args.lr else 0.01
    args.wd = float(args.wd) if args.wd else 5e-4

    return args


# define the eval metrics
class AccuracyMetric(mx.metric.EvalMetric):
    def __init__(self, name='Accuracy', axis=-1, **kwargs):
        super(AccuracyMetric, self).__init__(name=name, **kwargs)
        self._axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`  [cls_labels]
            The labels of the data.

        preds : list of `NDArray`   [scores]
            Predicted values.
        """
        if not isinstance(labels, (list)):
            labels = [labels]
        if not isinstance(preds, (list)):
            preds = [preds]

        pred_scores = preds[0]
        cls_labels = labels[0]

        num_inst = cls_labels.size

        pred_labels = mx.nd.argmax(pred_scores, axis=self._axis)
        self.sum_metric += (pred_labels == cls_labels.astype('float32')).sum().asscalar()
        self.num_inst += num_inst


class ValidMetric(mx.metric.EvalMetric):
    def __init__(self, name='ValidAcc', axis=-1, **kwargs):
        super(ValidMetric, self).__init__(name=name, **kwargs)
        self._axis = axis

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`  [cls_labels]
            The labels of the data.

        preds : list of `NDArray`   [scores]
            Predicted values.
        """
        if not isinstance(labels, (list)):
            labels = [labels]
        if not isinstance(preds, (list)):
            preds = [preds]

        pred_scores = preds[0]
        cls_labels = labels[0]

        num_inst = cls_labels.size

        pred_labels = mx.nd.argmax(pred_scores, axis=self._axis)

        self.sum_metric += (pred_labels == cls_labels.astype('float32')).sum().asscalar()
        self.num_inst += num_inst


def get_train_val_dataset(mixup=False, root='../CIFAR10/data/kaggle_cifar10/train_valid_test/'):
    # use mixup dataset
    """
    :param args: 
    :param root: str 
        root path of the data
    :return: 
    """
    if mixup:
        train_filename = 'train_valid'
    else:
        train_filename = 'train'
    valid_filename = 'valid'
    train_filepath = os.path.join(root + train_filename)
    valid_filepath = os.path.join(root + valid_filename)
    # get train_dataset
    train_dataset = ImageFolderDataset(root=train_filepath)
    valid_dataset = ImageFolderDataset(root=valid_filepath)
    return train_dataset, valid_dataset


def get_train_val_dataloader(train_dataset, valid_dataset, batch_size):
    """
    :param train_dataset: 
    :param valid_dataset: 
    :return: train_dataloader,valid_dataloader
    """

    # create data augmetation
    train_transformer = transforms.Compose([
        transforms.Resize(256),  # Resize the image to 256
        transforms.RandomResizedCrop(128, scale=(0.64, 1.0),
                                     ratio=(1.0, 1.0)),
        # Random flip
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        # Normalize
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))])
    valid_transformer = transforms.Compose([transforms.Resize(128),
                                            transforms.ToTensor(),
                                            # Normalize
                                            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                 std=(0.2023, 0.1994, 0.2010))])

    # get dataloader
    train_dataloader = DataLoader(train_dataset.transform_first(train_transformer),
                                  batch_size=batch_size, shuffle=True, last_batch='keep')
    valid_dataloader = DataLoader(valid_dataset.transform_first(valid_transformer),
                                  batch_size=batch_size, shuffle=True, last_batch='keep')

    return train_dataloader, valid_dataloader


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
            split_data = nd.slice_axis(data, axis=0, begin=begin, end=end)
            new_data.append(split_data.as_in_context(ctx))
        new_batch.append(new_data)
    return new_batch


def evaluate(net, valid_dataloader, valid_metric, ctx_list, hybird):
    valid_metric.reset()
    if hybird:
        net.hybridize(static_alloc=True)
    for i, batch in enumerate(valid_dataloader):
        metrics = []
        batch_size = len(batch[0])
        batch = split_and_load_data(batch, ctx_list, batch_size)
        for data, label in zip(*batch):
            # forward
            preds = net(data)
            metrics.append([[label], [preds]])
        # update metric
        for record in metrics:
            valid_metric.update(record[0], record[1])

    return valid_metric.get()


def save_parameters(net, logger, best_acc, current_acc, epoch, save_interval, prefix):
    current_acc = float(current_acc)
    if current_acc > best_acc[0]:
        logger.info('[Epoch {}] acc {} higer than current best {} saving to {}'.format(
            epoch, current_acc, best_acc[0], '{:s}_best.params'.format(prefix)))
        best_acc[0] = current_acc
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_acc.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_acc))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}'.format(prefix, epoch, current_acc)))


def train(net, train_dataloader, valid_dataloader, ctx_list, args):
    """Training pipline """
    # optimizer
    trainer = Trainer(net.collect_params(),
                      'sgd',
                      {'learning_rate': args.lr,
                       'wd': args.wd,
                       'momentum': args.momentum})
    # loss
    acc_metric = AccuracyMetric()
    loss_metric = mx.metric.Loss('SoftMaxCrossEntropyLoss')
    valid_metric = ValidMetric()
    cross_entropy_loss = gloss.SoftmaxCrossEntropyLoss()

    metric1 = [loss_metric]
    metric2 = [acc_metric]

    # create a logging
    logging.basicConfig()
    # get a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fd = logging.FileHandler(log_file_path)
    logger.addHandler(fd)
    logger.info(args)
    if args.verbose:
        logger.info('Trainabel paramters:')
        logger.info(net.collect_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    logger.info('Traing on {}'.format(ctx_list))
    # create
    best_acc = [0]
    lr_steps = sorted([int(step) for step in args.lr_decay_epoch.split(',') if step.strip()])
    lr_decay = float(args.lr_decay)

    for epoch in range(args.start_epoch, args.epochs):
        ttime = time.time()
        btime = time.time()

        # lr_decay
        if lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info('[Epoch {}] set learning rate to {:.4f}'.format(epoch, new_lr))

        acc_metric.reset()
        if args.hybrid:
            net.hybridize(static_alloc=True)
        # get mini-batch data
        # batch [data,label]
        for i, batch in enumerate(train_dataloader):
            batch_size = len(batch[0])
            batch = split_and_load_data(batch, ctx_list, batch_size)

            losses = []
            metrics = []
            with autograd.record():
                for data, cls_label in zip(*batch):
                    # forward
                    pred_scores = net(data)
                    # loss
                    loss = cross_entropy_loss(pred_scores, cls_label)
                    # record loss and preds
                    losses.append(loss)
                    metrics.append([[cls_label], [pred_scores]])
            # backward
            autograd.backward(losses)
            # optimizer params
            trainer.step(batch_size)
            # update metrics...
            for record in metrics:
                acc_metric.update(record[0], record[1])
            for record in losses:
                loss_metric.update(0, record)
            if args.log_interval and not (i + 1) % args.log_interval:
                # logging

                info = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metric1 + metric2])
                msg = '[Epoch {}][Batch {}],Speed: {:.3f} samples/sec,{}'.format(
                    epoch, i, args.log_interval * batch_size / (time.time() - btime), info)
                logger.info(msg)
                btime = time.time()

        info = ','.join(['{}={:.3f}'.format(*loss_metric.get())])
        msg = '[Epoch {}] Traning cost : {:.3f},{}'.format(
            epoch, time.time() - ttime, info)
        logger.info(msg)
        if args.val_interval and not (epoch + 1) % args.val_interval:
            name, current_acc = evaluate(net, valid_dataloader, valid_metric, ctx_list, args.hybrid)
            info = '{}={:.3f}'.format(name, current_acc)
            msg = '[Epoch {}] Validation {}.'.format(epoch, info)
            logger.info(msg)
        else:
            current_acc = 0

        save_parameters(net, logger, best_acc, current_acc, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    # parse argument
    args = parse_arg()
    # fix seed for mxnet,numpy


    last_gamma = False
    if args.last_gamma:
        last_gamma = True
    final_drop = args.final_drop
    # create ctx
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    net_name = '_'.join(('resenet50v1', 'cifar10'))
    args.save_prefix += net_name
    # load model
    net = resnet50_cifar10(last_gamma=last_gamma, final_drop=final_drop)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())

    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # train_data
    train_dataset, valid_dataset = get_train_val_dataset(args.mixup)
    train_dataloader, valid_dataloader = get_train_val_dataloader(train_dataset, valid_dataset,
                                                                  batch_size=args.batch_size, )
    train(net, train_dataloader, valid_dataloader, ctx, args)

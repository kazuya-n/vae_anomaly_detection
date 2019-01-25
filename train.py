#!/usr/bin/env python
"""
Reference
https://github.com/crcrpar/chainer-VAE
https://github.com/chainer/chainer/tree/master/examples/vae
https://qiita.com/shinmura0/items/811d01384e20bfd1e035
"""
from __future__ import print_function
import argparse
import os

import matplotlib
# Disable interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd

import chainer
from chainer import cuda, training
from chainer.training import extensions
import chainer.functions as F

import cvae_net

import PIL

from tqdm import tqdm

matplotlib.use('Agg')
# Size of cutting image
CutSize = 16
# Path of dataset folder
Path = "images/"


# Make datasets from image
def road_data():
    print("Loading datasets...")
    d = pd.read_csv(f"{Path}list.txt", header=None)
    train = []
    with tqdm(total=len(list(d.iterrows()))) as pbar:
        for i, data in tqdm(d.itertuples()):
            pil_img = PIL.Image.open(Path + data)
            img = np.array(pil_img.resize((240, 180)).convert("L"))
            imarray = img.astype(np.float32) / 255.
            train.append((imarray))
            pbar.update(1)
    train = np.array(train)
    train = train.reshape(1000, 180, 240, 1)
    print(f"Done, {train.shape}")

    return train


def cut_img(x, number, width=CutSize, height=CutSize):
    print("cutting images ...")
    x_out = []
    x_shape = x.shape

    for i in tqdm(range(number)):
        shape_0 = np.random.randint(0, x_shape[0])
        shape_1 = np.random.randint(0, x_shape[1]-height)
        shape_2 = np.random.randint(0, x_shape[2]-width)
        temp = x[shape_0, shape_1:shape_1+height, shape_2:shape_2+width, 0]
        x_out.append(temp.reshape((height, width, x_shape[3])))
    print("Complete.")
    out = np.array(x_out)

    return out


def save_img(x_anomaly, img_anomaly, name):
    path = 'results/'
    if not os.path.exists(path):
        os.mkdir(path)
    img_max = img_anomaly.max()
    img_min = img_anomaly.min()
    img_anomaly = (img_anomaly-img_min)/(img_max-img_min) * 9 + 1

    plt.subplot(2, 1, 1)
    print(img_anomaly.shape)
    plt.imshow(x_anomaly[0, :, :], cmap='gray')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.imshow(img_anomaly[0, 0, :, :], cmap='Blues', norm=colors.LogNorm())
    plt.axis('off')
    plt.colorbar()

    plt.clim(1, 10)

    plt.savefig(path + name + ".png")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=1000, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=80, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--out', '-o', type=str, default='./results/',
                        help='dir to save snapshots.')
    parser.add_argument('--interval', '-i', type=int, default=20, help='interval of save images.')
    parser.add_argument
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = cvae_net.CVAE(1, 16, args.dimz, 16)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Prepare Dataset
    data = road_data().astype(np.float32)
    # Prepare test(not separated) images
    test_img = data[950:1000, :, :, :]
    test_img = np.transpose(test_img, (0, 3, 1, 2))
    # Prepare separated images
    data = cut_img(data, 1000000)
    data = np.transpose(data, (0, 3, 1, 2))
    train, test = chainer.datasets.split_dataset(data, 999900)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)
    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    if not os.path.exists(os.path.join(args.out, 'cg.dot')):
        print('dump computational graph of `main/loss`')
        trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'),
                   trigger=(args.interval, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    # if you want to output different log files epoch by epoch,
    # use below statement.
    # trainer.extend(extensions.LogReport(log_name='log_'+'{epoch}'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # if you want to show the result images epoch by epoch,
    # use the extension below.
    @training.make_extension(trigger=(args.interval, 'epoch'))
    def save_images(trainer):
        height = 16
        width = 16
        move = 2
        test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
        x = cuda.cupy.asarray(test_img)[test_ind, :]
        for index, image in enumerate(x):
            x_batch = []
            img_anomaly = xp.zeros((1, 1, image.shape[1], image.shape[2]))
            # Making batch of separated test image
            for i in tqdm(range(int((image.shape[1]-height)/move)+1)):
                for j in range(int((image.shape[2]-width)/move)+1):
                    x_sub = image[0, i*move:i*move+height, j*move:j*move+width]
                    x_sub = x_sub.reshape(1, height, width)
                    x_batch.append(x_sub)
            x_batch = xp.asarray(x_batch, dtype=xp.float32)
            print(f"{x_batch.shape}")
            # Encoding
            x_batch = chainer.Variable(x_batch)
            mu, ln_var = model.encode(x_batch)
            z = F.gaussian(mu, ln_var)
            mu, sigma = model.decode(z)
            # Evaluate anormaly to calculate loss
            batch_index = 0
            for i in tqdm(range(int((image.shape[1]-height)/move)+1)):
                for j in range(int((image.shape[2]-width)/move)+1):
                    loss = 0.5 * (x_batch[batch_index, 0, :, :] - mu[batch_index, 0, :, :])**2 / sigma[batch_index, 0, :, :]
                    loss = F.sum(loss)
                    img_anomaly[0, 0, i*move:i*move+height, j*move:j*move+width] += loss.data
                    batch_index += 1
            # Saving
            save_img(chainer.cuda.to_cpu(image),
                     chainer.cuda.to_cpu(img_anomaly),
                     f"{index}_{trainer.updater.epoch}")

    trainer.extend(save_images)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()

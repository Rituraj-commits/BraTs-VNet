import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=3, help='input batch size')
parser.add_argument('-g','--gpu', type=int, default=0)
parser.add_argument('-nEpochs','--epochs', type=int, default=300)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("-lr","--learning_rate", type = float, default = 0.001)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('-opt','--optimizer', type=str, default='adam',
                    choices=('sgd', 'adam'))
parser.add_argument('-dr','--dataset_path',type=str,default='/media/ri2raj/External HDD/Task01_BrainTumour/')
args = parser.parse_args()
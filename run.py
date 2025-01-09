
import argparse
import iars
import os

#saves figures/models/etc. to desktop folder called IARS
#2 sub dirs:
# IARS/models
# IARS/figures/Dataset/

def error_handling(args):
    if args.lr < 0:
        raise ValueError("The learning rate must be greater than 0")
    if args.e <= 0:
        raise ValueError("The number of epochs must be greater than 0")
    if args.i < 0:
        raise ValueError("The number of iterations must be greater than 0")
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='temp')
    parser.add_help
    #HYPER PARAMETERS
    #lr
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, default 0.001")
    #epochs
    parser.add_argument("--e", type=int, default=100, help="training epochs, default 100")
    #number of iterations
    parser.add_argument("--i", type=int, default=1, help="number of training loops (i.e will run 100 epochs 1 time), default 1" )

    #TRAINING
    #dataset
    parser.add_argument("--ds", type=str, default="BasicMotions", help="choose dataset to train, default BasicMotions")

    #SAVING

    #figure directory
    parser.add_argument("--dir", type=str, default="IARS/figures", help="change the directory of where to save figures, default: IARS/figures")

    args = parser.parse_args()

    error_handling(args)

    iars.training(args)
    

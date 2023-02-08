import argparse

def getCLIOptions():
    ap = argparse.ArgumentParser()
    ap.add_argument("-eps", "--epsilon", default=0.03, type=float, help='norm threshold epsilon')
    ap.add_argument("-alp", "--alpha", default=2/255, type=float, help='PGD update weight, alpha')
    ap.add_argument("-s", "--step", default=10, type=int, help='PGD update steps')
    ap.add_argument("-ep", "--epoch", default=20, type=int, help='Total epoches')
    ap.add_argument("-bs", "--batch-size", default=6, type=int, help='training batch size')
    ap.add_argument("-seed", "--random-seed", type=int, default=17, help="random seed in optimization")
    ap.add_argument("-at", "--adv-type", type=str, required=True, choices=['object', 'image', 'object_l0'], help="select adversatial generating method")
    ap.add_argument("-lp", "--log-postfix", type=str, required=True, help='Log postfix as notes')
    # params for l0 norm
    ap.add_argument("--adam_lr", default=0.5, type=float, help='adam learning rate for l0 norm')
    ap.add_argument("--mask_wt", default=0.06, type=float, help='mask optim weight for l0 norm')
    ap.add_argument("--l0_thresh", default=0.1, type=float, help='norm threshold epsilon for l0 norm')
    args = vars(ap.parse_args())
    return args

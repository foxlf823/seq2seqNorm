import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-whattodo', type=int, default=1, help='1-train ner, 2-train norm, 3-test')
parser.add_argument('-verbose', action='store_true', help='1-print debug logs')
parser.add_argument('-random_seed', type=int, default=1)
parser.add_argument('-output', default='./output')
parser.add_argument('-iter', type=int, default=100)
parser.add_argument('-gpu', type=int, default=-1)
parser.add_argument('-tune_wordemb', action='store_true', default=False)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-word_emb_dim', type=int, default=100)
parser.add_argument('-hidden_dim', type=int, default=100)
parser.add_argument('-char_emb_dim', type=int, default=50)
parser.add_argument('-char_hidden_dim', type=int, default=50)
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-dropout', type=float, default=0, help='0~0.5')
parser.add_argument('-l2', type=float, default=1e-8)
parser.add_argument('-patience', type=int, default=20)
parser.add_argument('-gradient_clip', type=float, default=5.0)
parser.add_argument('-config', default='./config.txt')
parser.add_argument('-use_char', action='store_true', default=False)
parser.add_argument('-bidirect', action='store_true', default=False)
parser.add_argument('-use_teacher_forcing', action='store_true', default=False)
parser.add_argument('-pos_emb_dim', type=int, default=20)
parser.add_argument('-pretraining', action='store_true', default=False)
parser.add_argument('-expected_accuracy', type=int, default=95)

opt = parser.parse_args()


def config_file_to_dict(input_file):
    config = {}
    fins = open(input_file, 'r').readlines()
    for line in fins:
        line = line.strip()
        if line == '':
            continue
        if len(line) > 0 and line[0] == "#":
            continue

        pairs = line.split()
        if len(pairs) > 1:
            for idx, pair in enumerate(pairs):
                if idx == 0:
                    items = pair.split('=')
                    if items[0] not in config:
                        feat_dict = {}
                        config[items[0]] = feat_dict
                    feat_dict = config[items[0]]
                    feat_name = items[1]
                    one_dict = {}
                    feat_dict[feat_name] = one_dict
                else:
                    items = pair.split('=')
                    one_dict[items[0]] = items[1]
        else:
            items = pairs[0].split('=')
            if items[0] in config:
                print("Warning: duplicated config item found: %s, updated." % (items[0]))
            config[items[0]] = items[-1]

    return config

def read_config(config_file):
    config = config_file_to_dict(config_file)
    return config

config = read_config(opt.config)


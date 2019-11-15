import os
import shutil
import logging
import json
import torch
import argparse
import math
from Transformer import Transformer
from tensorboardX import SummaryWriter
from utils import BatchManager, load_data, build_vocab, dump_tensors
from sumeval.metrics.bleu import BLEUCalculator
from translate import greedy, print_summaries
from sumeval.metrics.rouge import RougeCalculator
import config

parser = argparse.ArgumentParser(description='Selective Encoding for Couplet Generation in DyNet')

parser.add_argument('--n_epochs', type=int, default=12, help='Number of epochs [default: 5]')
parser.add_argument('--n_train', type=int, default=690491,
                    help='Number of training data [default: 690491]')
parser.add_argument('--n_valid', type=int, default=80000,
                    help='Number of validation data [default: 80000])')
parser.add_argument('--n_eval', type=int, default=4000, help='Number of evaluation data [default:4000]')
parser.add_argument('--batch_size', type=int, default=64, help='Mini batch size [default: 64]')
parser.add_argument('--emb_dim', type=int, default=512, help='Embedding size [default: 512]')
parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
parser.add_argument('--maxout_dim', type=int, default=2, help='Maxout size [default: 2]')
parser.add_argument('--ckpt_file', type=str, default='./models/params_v2_0.pkl')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/train.log',
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)

model_dir = './models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


def run_batch(valid_x, valid_y, model):
    _, x = valid_x.next_batch()
    _, y = valid_y.next_batch()
    logits = model(x, y)
    loss = model.loss_layer(logits.view(-1, logits.shape[-1]),
                            y[:, 1:].contiguous().view(-1))
    return loss

def myeval(valid_x, valid_y, vocab, model):
    rouge = RougeCalculator(stopwords=True, lang="zh")
    bleu_ch = BLEUCalculator(lang="zh")

    model.eval()
    eval_batch_num = 0
    sum_rouge_1 = 0
    sum_rouge_2 = 0
    sum_rouge_L = 0
    score_ch = 0
    sum_loss = 0
    limit = 63
    logging.info('Evaluating on %d minibatches...' % limit)
    i2w = {key: value for value, key in vocab.items()}
    ckpt_file = args.ckpt_file[9:]
    fout_pred = open(os.path.join('tmp/systems', '%s.txt' % ckpt_file), "w")
    fout_y = open(os.path.join('tmp/models', 'ref_%s.txt' % ckpt_file), "w")
    while eval_batch_num < limit:
        with torch.no_grad():
            loss = run_batch(valid_x, valid_y, model)
            sum_loss += loss
            _, x = valid_x.next_batch()
            pred = greedy(model, x, vocab)
            _, y = valid_y.next_batch()
            y = y[:,1:].tolist()
            for idx in range(len(pred)):
                line_pred = [i2w[tok] for tok in pred[idx] if tok != vocab[config.end_tok] and tok != vocab[config.pad_tok]]
                line_y = [i2w[tok] for tok in y[idx] if tok != vocab[config.end_tok] and tok != vocab[config.pad_tok]]
                fout_pred.write(" ".join(line_pred) + "\n")
                fout_y.write(" ".join(line_y) + "\n")
                sum_rouge_1 += rouge.rouge_n(references=" ".join(line_y),summary=" ".join(line_pred),n=1)
                sum_rouge_2 += rouge.rouge_n(references=" ".join(line_y),summary=" ".join(line_pred),n=2)
                sum_rouge_L += rouge.rouge_l(references=" ".join(line_y),summary=" ".join(line_pred))
                score_ch += bleu_ch.bleu(" ".join(line_y), " ".join(line_pred))
            eval_batch_num += 1
    fout_pred.close()
    fout_y.close()
    avg_rouge_1 = sum_rouge_1/(len(pred) * limit)
    avg_rouge_2 = sum_rouge_2/(len(pred) * limit)
    avg_rouge_L = sum_rouge_L/(len(pred) * limit)
    avg_bleu_ch = score_ch/(len(pred) * limit)
    avg_loss = sum_loss/limit
    print("ROUGE_1 = ",avg_rouge_1)
    print("ROUGE_2 = ",avg_rouge_2)
    print("ROUGE_L = ",avg_rouge_L)
    print("BLEU = ", avg_bleu_ch)
    print("Perplexity = ", math.pow(2, avg_loss))
    model.train()

def train(train_x, train_y, valid_x, valid_y, model, optimizer, vocab, scheduler, n_epochs=1, epoch=0):
    logging.info("Start to train with lr=%f..." % optimizer.param_groups[0]['lr'])
    n_batches = train_x.steps

    model.train()
    for epoch in range(epoch, n_epochs):
        valid_x.bid = 0
        valid_y.bid = 0

        if os.path.isdir('runs/epoch%d' % epoch):
            shutil.rmtree('runs/epoch%d' % epoch)
        writer = SummaryWriter('runs/epoch%d' % epoch)

        for idx in range(n_batches):
            optimizer.zero_grad()

            loss = run_batch(train_x, train_y, model)
            loss.backward()  # do not use retain_graph=True
            # torch.nn.utils.clip_grad_value_(model.parameters(), 5)

            optimizer.step()

            if idx <= n_batches:
                train_loss = loss.cpu().detach().numpy()
                model.eval()
                with torch.no_grad():
                    valid_loss = run_batch(valid_x, valid_y, model)
                # 加入过拟合判断
                if valid_loss > train_loss * 100:
                    logging.info('Over-fitting, stop training.')
                    logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
                                 % (epoch, idx + 1, train_loss, valid_loss))
                    break
                logging.info('epoch %d, step %d, training loss = %f, validation loss = %f'
                             % (epoch, idx + 1, train_loss, valid_loss))
                writer.add_scalar('scalar/train_loss', train_loss, (idx + 1) // 50)
                writer.add_scalar('scalar/valid_loss', valid_loss, (idx + 1) // 50)
                model.train()
                torch.cuda.empty_cache()
            if (idx + 1) % 500 == 0:
                myeval(valid_x, valid_y, vocab, model)
            # dump_tensors()

        if epoch < 4:
            scheduler.step()  # make sure lr will not be too small
        save_state = {'state_dict': model.state_dict(),
                      'epoch': epoch + 1,
                      'lr': optimizer.param_groups[0]['lr']}
        torch.save(save_state, os.path.join(model_dir, 'params_v1_%d.pkl' % epoch))
        logging.info('Model saved in dir %s' % model_dir)
        writer.close()


def main():
    print(args)

    data_dir = 'data/'
    TRAIN_X = os.path.join(data_dir, 'train/in.txt')
    TRAIN_Y = os.path.join(data_dir, 'train/out.txt')
    VALID_X = os.path.join(data_dir, 'dev/in.txt')
    VALID_Y = os.path.join(data_dir, 'dev/out.txt')
    EVAL_X = os.path.join(data_dir, 'test/in.txt')
    EVAL_Y = os.path.join(data_dir, 'test/out.txt')

    small_vocab_file = os.path.join(data_dir, 'vocab.json')
    if os.path.exists(small_vocab_file):
        print("Vocab exists!")
        small_vocab = json.load(open(small_vocab_file))
    else:
        small_vocab = build_vocab([TRAIN_X, TRAIN_Y], small_vocab_file, vocab_size=800000)

    max_src_len = 34
    max_tgt_len = 34

    bs = args.batch_size
    n_train = args.n_train
    n_valid = args.n_valid
    n_eval = args.n_eval

    vocab = small_vocab

    train_x = BatchManager(load_data(TRAIN_X, max_src_len, n_train), bs, vocab)
    train_y = BatchManager(load_data(TRAIN_Y, max_tgt_len, n_train), bs, vocab)
    valid_x = BatchManager(load_data(VALID_X, max_src_len, n_valid), bs, vocab)
    valid_y = BatchManager(load_data(VALID_Y, max_tgt_len, n_valid), bs, vocab)
    eval_x = BatchManager(load_data(EVAL_X, max_src_len, n_eval), bs, vocab)
    eval_y = BatchManager(load_data(EVAL_Y, max_tgt_len, n_eval), bs, vocab)
    print("vocab length is: "+ str(len(vocab)))
    model = Transformer(len(vocab), len(vocab), max_src_len, max_tgt_len, 6, 8, 256, 64, 64, 1024, src_tgt_emb_share=True, tgt_prj_emb_share=True).cuda()
    saved_state = {'epoch': 0, 'lr': 0.001}
    if os.path.exists(args.ckpt_file):
        saved_state = torch.load(args.ckpt_file)
        model.load_state_dict(saved_state['state_dict'])
        logging.info('Load model parameters from %s' % args.ckpt_file)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=saved_state['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)
    scheduler.step()  # last_epoch=-1, which will not update lr at the first time

    # myeval(valid_x, valid_y, vocab, model)
    # train(train_x, train_y, valid_x, valid_y, model, optimizer, vocab, scheduler, args.n_epochs, saved_state['epoch'])
    myeval(eval_x, eval_y, vocab, model)

if __name__ == '__main__':
    main()


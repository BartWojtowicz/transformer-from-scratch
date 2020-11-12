import torch
import tqdm
import torch.nn.functional as F

from torch import nn
from torchtext import data, datasets, vocab
from argparse import ArgumentParser

from modules import ClassificationTransformer


def run_classification(args):
    # Loading IMDB data 
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train, max_size = args.vocab_size - 2)
    LABEL.build_vocab(train)

    train_iter, test_iter = data.BucketIterator.splits((train, test), batch_size=args.batch_size, device='cuda')

    model = ClassificationTransformer(emb = args.emb_size, heads = args.num_heads, depth = args.depth, seq_length = args.max_length, num_tokens = args.vocab_size, num_classes = 2)
    model.cuda()

    optimizer = torch.optim.Adam(lr = args.lr, params = model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (args.lr_warmup / args.batch_size), 1.0))

    # todo: implement using lightning =] 
    for e in range(args.num_epochs):
        model.train(True)
        for batch in tqdm.tqdm(train_iter):
            optimizer.zero_grad()

            input = batch.text[0] # at text[1] we have lengths per sequence
            label = batch.label - 1 # labels are 1,2 by default, thus  we subtract 1

            # clipping input sequences to provided max_length
            if input.size(1) > args.max_length:
                input = input[:, :args.max_length]
            
            out = model(input)
            loss = F.nll_loss(out, label) # negative log likelihood loss
            loss.backward()

            # gradient clipping
            if args.gradient_clipping > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            model.eval()
            total, correct = [0.0, 0.0]

            for batch in test_iter:
                input = batch.text[0]
                label = batch.label - 1

                if input.size(1) > args.max_length:
                    input = input[:, :args.max_length]
                
                out = model(input).argmax(dim=1)
                total += float(input.size(0))
                correct += float((label == out).sum().item())

            acc = correct / total
            print(f'TEST ACC: {acc:.4}')



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-e", "--num-epochs",
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=10, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-E", "--embedding", dest="emb_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    args = parser.parse_args()
    run_classification(args)
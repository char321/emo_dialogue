import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time
import pickle

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

from model import BiModel, Model, MaskedNLLLoss
from dataloader import IEMOCAPDataset

# function to split the training set into training set and validation set
def get_train_valid_sampler(train_set, valid=0.1):
    size = len(train_set)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_IEMOCAP_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    train_set = IEMOCAPDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(train_set, valid)

    # init data loader
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,  # sampler idx
                              collate_fn=train_set.collate_fn,  # function to form batch
                              num_workers=num_workers,  # default optimal value
                              pin_memory=pin_memory)

    valid_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_set = IEMOCAPDataset(path=path, train=False)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, cuda, optimizer=None, is_train=False):
    losses = []
    predictions = []
    labels = []
    masks = []
    alphas, alphas_f, alphas_b, vids = [], [], [], []

    assert not is_train or optimizer != None

    if is_train:
        model.train()
    else:
        model.eval()

    for data in dataloader:
        if is_train:
            optimizer.zero_grad()
        # import ipdb;ipdb.set_trace()
        # text, visual, audio, speakers, len_labels, labels
        textf, visuf, acouf, q_mask, u_mask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]

        #log_prob = model(torch.cat((textf,acouf,visuf),dim=-1), qmask,umask,att2=True) # seq_len, batch, n_classes

        # alpha is the attention score
        log_prob, alpha, alpha_f, alpha_b = model(textf, q_mask, u_mask, use_att=True)  # log_prob -> seq_len * batch * n_classes
        temp_log_prob = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])  # (batch * seq_len) * n_classes
        temp_label = label.view(-1)  # (batch * seq_len)
        loss = loss_function(temp_log_prob, temp_label, u_mask)  # value
        pred = torch.argmax(temp_log_prob, 1)  # (batch * seq_len)

        predictions.append(pred.data.cpu().numpy())
        labels.append(temp_label.data.cpu().numpy())
        masks.append(u_mask.view(-1).cpu().numpy())

        losses.append(loss.item() * masks[-1].sum())
        if is_train:
            loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            # append attention scores
            alphas += alpha
            alphas_f += alpha_f
            alphas_b += alpha_b
            vids += data[-1]

    if predictions != []:
        # flatten the outputs
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        masks = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'), []

    avg_loss = round(np.sum(losses) / np.sum(masks), 4)
    # use masks as sample_weight to ignore the padding part
    avg_accuracy = round(accuracy_score(labels, predictions, sample_weight=masks) * 100, 2)
    avg_f_score = round(f1_score(labels, predictions, sample_weight=masks, average='weighted') * 100, 2)

    return avg_loss, avg_accuracy, labels, predictions, masks, avg_f_score, [alphas, alphas_f, alphas_b, vids]

if __name__ == '__main__':

    # default argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--rec-dropout-rate', type=float, default=0.1, metavar='rec_dropout_rate', help='recurrent dropout rate')
    parser.add_argument('--dropout-rate', type=float, default=0.1, metavar='dropout_rate', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=20, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True, help='class weight')
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    batch_size = args.batch_size
    n_classes = 6
    cuda = args.cuda
    n_epochs = args.epochs

    # dimensions
    d_m = 100  # dimension of the utterance representation
    d_g = 500  # dimension of global state
    d_p = 500  # dimension of party state (speaker update is sufficient)
    d_e = 300  # dimension of emotion representation
    d_h = 300  # dimension of hidden layer
    d_a = 100  # concat attention
    pkl_path = './IEMOCAP_features/IEMOCAP_features_raw.pkl'

    # init model
    model = BiModel(d_m, d_g, d_p, d_e, d_h,
                    n_classes=n_classes,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    rec_dropout_rate=args.rec_dropout_rate,
                    dropout_rate=args.dropout_rate)

    if cuda:
        model.cuda()

    # get weights of classes
    loss_weights = torch.FloatTensor([1/0.086747, 1/0.144406, 1/0.227883, 1/0.160585, 1/0.127711, 1/0.252668])

    # define loss function
    if args.class_weight:
        loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_function = MaskedNLLLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    # load dataset
    train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(pkl_path, valid=0.0, batch_size=batch_size, num_workers=2)
    # init
    best_loss, best_label, best_pred, best_mask = None, None, None, None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_f_score, _ = train_or_eval_model(model, loss_function, train_loader, e, cuda, optimizer, True)
        valid_loss, valid_acc, _, _, _, val_f_score, _ = train_or_eval_model(model, loss_function, valid_loader, e, cuda)
        test_loss, test_acc, test_label, test_pred, test_mask, test_f_score, attentions = train_or_eval_model(model, loss_function, test_loader, e, cuda)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_attn = test_loss, test_label, test_pred, test_mask, attentions

        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'. \
              format(e + 1, train_loss, train_acc, train_f_score, valid_loss, valid_acc, val_f_score, \
                     test_loss, test_acc, test_f_score, round(time.time() - start_time, 2)))

    if args.tensorboard:
        writer.close()

    print('Test performance..')
    print('Loss {} accuracy {}'.format(best_loss, round(accuracy_score(best_label, best_pred, sample_weight=best_mask) * 100, 2)))
    print(classification_report(best_label, best_pred, sample_weight=best_mask, digits=4))
    print(confusion_matrix(best_label, best_pred, sample_weight=best_mask))
    # with open('best_attention.p','wb') as f:
    #     pickle.dump(best_attn+[best_label,best_pred,best_mask],f)

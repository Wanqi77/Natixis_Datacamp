
import time
import torch.utils.data as Data
import torch.nn as nn
import torch
import transformers
from Model import Encoder, Decoder_Classif
from Dataset import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
torch.backends.cudnn.benchmark = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)


model = "distilbert-base-uncased"
data_path = 'data/sums.json'
stock_data_size = 20
input_size = 768
encoded_size = 20
hidden_size = 64
dropout = 0.5

start_epoch = 1
epochs = 2  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement
batch_size = 32
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 1e-2  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_loss = 1.  # best loss score right now
print_freq = 8  # print training/validation stats every __ batches
fine_tune_encoder = False # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
save_path = './checkpoint/' # checkpoint save path


def train_classif(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    encoder.train()
    decoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    for i, data in enumerate(train_loader):
        text = data['text_token'].type(torch.IntTensor)
        stock_price = data['stock'].type(torch.FloatTensor)
        label = data['target_classif'].type(torch.FloatTensor)

        # Forward prop.
        decoder_input = encoder(text)
        label_pred = decoder(decoder_input, stock_price)

        # Calculate loss
        loss = criterion(label_pred.reshape(-1), label)

        # Back prop.
        decoder_optimizer.zero_grad()

        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metric
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}] [{1}/{2}]\n'
                  'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                  'Data Load Time {data_time.val:.3f}s (Average:{data_time.avg:.3f}s)\n'
                  'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses))


def validate(val_loader, encoder, decoder, criterion):

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    with torch.no_grad():
        accuracy_accumulator = 0
        num_samples = 0
        for i, data in enumerate(val_loader):
            text = data['text_token'].type(torch.IntTensor)
            stock_price = data['stock'].type(torch.FloatTensor)
            label = data['target_classif'].type(torch.FloatTensor)

            # Forward prop.
            decoder_input = encoder(text)
            label_pred = decoder(decoder_input, stock_price).reshape(-1)

            # Calculate loss
            loss = criterion(label_pred, label)

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()
            accuracy_accumulator += ((label_pred > 0.5) == label).sum().item()
            num_samples += label.shape[0]

            print(accuracy_accumulator)
            print(num_samples)

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\n'
                        'Batch Time {batch_time.val:.3f}s (Average:{batch_time.avg:.3f}s)\n'
                        'Loss {loss.val:.4f} (Average:{loss.avg:.4f})\n'.format(i, len(val_loader), batch_time=batch_time,loss=losses))
        
        accuracy = accuracy_accumulator / num_samples

    return losses.avg, label, label_pred, accuracy


def main():
    global model, data_path, stock_data_size, input_size, encoded_size, hidden_size, dropout, start_epoch, epochs, epochs_since_improvement, batch_size, workers, encoder_lr, decoder_lr, grad_clip, alpha_c, best_loss , print_freq, fine_tune_encoder, checkpoint, save_path
    if checkpoint is None:
        decoder = Decoder_Classif(stock_data_size=stock_data_size, input_size=input_size, encoded_size=encoded_size, hidden_size=hidden_size)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                            lr=decoder_lr)
        encoder = Encoder(model)
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_loss = checkpoint['testLoss']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr)

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.BCEWithLogitsLoss().to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    dataset = PredictDataset(data_file=data_path, test_size=0.2, max_len=512, tokenizer=tokenizer)
    train_loader = Data.DataLoader(dataset.train_set(), batch_size=batch_size, shuffle=False)
    val_loader = Data.DataLoader(dataset.test_set(), batch_size=batch_size, shuffle=False)

    for epoch in range(start_epoch, start_epoch + epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train_classif(train_loader=train_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                epoch=epoch)

        # One epoch's validation, return the average loss of each batch in this epoch
        loss, label, label_pred, acc = validate(val_loader=val_loader,
                                    encoder=encoder, decoder=decoder, criterion=criterion)
        
        print('Validation: Epoch [{0}/{1}]\n'
                            'Loss {loss:.4f}\n'
                            'Accuracy {acc:.4f}\n'.format(epoch, epochs, loss=loss, acc=acc))


        # Check if there was an improvement
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(save_path, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, loss, is_best)


if __name__ == '__main__':
    
    main()



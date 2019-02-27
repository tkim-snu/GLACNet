import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import random
from data_loader import get_loader
from build_vocab import Vocabulary
from model import EncoderStory, DecoderStory
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build data loader
    train_data_loader = get_loader(args.train_image_dir, args.train_sis_path, vocab, train_transform, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data_loader = get_loader(args.val_image_dir, args.val_sis_path, vocab, val_transform, args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder = EncoderStory(args.img_feature_size, args.hidden_size, args.num_layers)
    decoder = DecoderStory(args.embed_size, args.hidden_size, vocab)

    pretrained_epoch = 0
    if args.pretrained_epoch > 0:
        pretrained_epoch = args.pretrained_epoch
        encoder.load_state_dict(torch.load('./models/encoder-' + str(pretrained_epoch) + '.pkl'))
        decoder.load_state_dict(torch.load('./models/decoder-' + str(pretrained_epoch) + '.pkl'))

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
        print("Cuda is enabled...")

    criterion = nn.CrossEntropyLoss()
    params = decoder.get_params() + encoder.get_params()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_train_step = len(train_data_loader)
    total_val_step = len(val_data_loader)

    min_avg_loss = float("inf")
    overfit_warn = 0

    for epoch in range(args.num_epochs):

        if epoch < pretrained_epoch:
            continue

        encoder.train()
        decoder.train()
        avg_loss = 0.0
        for bi, (image_stories, targets_set, lengths_set, photo_squence_set, album_ids_set) in enumerate(train_data_loader):
            decoder.zero_grad()
            encoder.zero_grad()
            loss = 0
            images = to_var(torch.stack(image_stories))

            features, _ = encoder(images)

            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = to_var(data[1])
                lengths = data[2]

                outputs = decoder(feature, captions, lengths)

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    loss += criterion(result[0], result[1][0:result[2]])

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)
            loss.backward()
            optimizer.step()

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch + 1, args.num_epochs, bi, total_train_step,
                        loss.item(), np.exp(loss.item())))
            
        avg_loss /= (args.batch_size * total_train_step * 5)
        print('Epoch [%d/%d], Average Train Loss: %.4f, Average Train Perplexity: %5.4f' %(epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        # Save the models
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' %(epoch+1)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' %(epoch+1)))

        # Validation
        encoder.eval()
        decoder.eval()
        avg_loss = 0.0
        for bi, (image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set) in enumerate(val_data_loader):
            loss = 0
            images = to_var(torch.stack(image_stories))

            features, _ = encoder(images)

            for si, data in enumerate(zip(features, targets_set, lengths_set)):
                feature = data[0]
                captions = to_var(data[1])
                lengths = data[2]

                outputs = decoder(feature, captions, lengths)

                for sj, result in enumerate(zip(outputs, captions, lengths)):
                    loss += criterion(result[0], result[1][0:result[2]])

            avg_loss += loss.item()
            loss /= (args.batch_size * 5)

            # Print log info
            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch + 1, args.num_epochs, bi, total_val_step,
                        loss.item(), np.exp(loss.item())))

        avg_loss /= (args.batch_size * total_val_step * 5)
        print('Epoch [%d/%d], Average Val Loss: %.4f, Average Val Perplexity: %5.4f' %(epoch + 1, args.num_epochs, avg_loss, np.exp(avg_loss)))

        #Termination Condition
        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss) else 0
        min_avg_loss = min(min_avg_loss, avg_loss)

        if overfit_warn >= 5:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--image_size', type=int, default=224 ,
                        help='size for input images')
    parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--train_image_dir', type=str, default='./data/train' ,
                        help='directory for resized train images')
    parser.add_argument('--val_image_dir', type=str, default='./data/val' ,
                        help='directory for resized val images')
    parser.add_argument('--train_sis_path', type=str,
                        default='./data/sis/train.story-in-sequence.json',
                        help='path for train sis json file')
    parser.add_argument('--val_sis_path', type=str,
                        default='./data/sis/val.story-in-sequence.json',
                        help='path for val sis json file')
    parser.add_argument('--log_step', type=int , default=20,
                        help='step size for prining log info')
    parser.add_argument('--img_feature_size', type=int , default=1024 ,
                        help='dimension of image feature')
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=1024 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=2 ,
                        help='number of layers in lstm')

    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    args = parser.parse_args()
    print(args)
    main(args)

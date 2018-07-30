import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import random
import os
import subprocess
from data_loader import get_loader
from torch.autograd import Variable
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderStory, DecoderStory
from PIL import Image
import json

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def transform_image(image, transform=None):
    image = image.resize([224, 224], Image.LANCZOS)

    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int, default=224 ,
                    help='size for input images')
parser.add_argument('--sis_path', type=str,
                    default='./data/sis/test.story-in-sequence.json')
parser.add_argument('--result_path', type=str,
                    default='./result.json')
parser.add_argument('--log_step', type=int , default=10,
                    help='step size for prining log info')
parser.add_argument('--model_num', type=int , default=0,
                    help='step size for prining log info')

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--vocab_path', type=str, default='./models/vocab.pkl',
                    help='path for vocabulary wrapper')

parser.add_argument('--img_feature_size', type=int , default=1024 ,
                    help='dimension of image feature')
parser.add_argument('--embed_size', type=int , default=256 ,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=1024 ,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=2 ,
                    help='number of layers in lstm')




args = parser.parse_args()


challenge_dir = '../VIST-Challenge-NAACL-2018/'
image_dir = './data/test/'
sis_path = args.sis_path
result_path = args.result_path
embed_path = './models/embed-' + str(args.model_num) + '.pkl'
encoder_path = './models/encoder-' + str(args.model_num) + '.pkl'
decoder_path = './models/decoder-' + str(args.model_num) + '.pkl'


transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))])

with open(args.vocab_path, 'rb') as f:
    vocab = pickle.load(f)

data_loader = get_loader(image_dir, sis_path, vocab, transform, args.batch_size, shuffle=False, num_workers=args.num_workers)

encoder = EncoderStory(args.img_feature_size, args.hidden_size, args.num_layers)
decoder = DecoderStory(args.embed_size, args.hidden_size, len(vocab))

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))

encoder.eval()
decoder.eval()

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    print("Cuda is enabled...")

criterion = nn.CrossEntropyLoss()

results = []
total_step = len(data_loader)
avg_loss = 0.0
for bi, (image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set) in enumerate(data_loader):
    loss = 0

    images = torch.stack(image_stories)
    if torch.cuda.is_available():
        images = images.cuda()

    features, _ = encoder(images)
    for si, data in enumerate(zip(features, targets_set, lengths_set, photo_sequence_set, album_ids_set)):
        feature = data[0]
        captions = data[1]
        lengths = data[2]
        photo_sequence = data[3]
        album_ids = data[4]

        if torch.cuda.is_available():
            captions = captions.cuda()

        outputs = decoder(feature, captions, lengths)

        for sj, result in enumerate(zip(outputs, captions, lengths)):
            loss += criterion(result[0], result[1][0:result[2]])

        inference_results = decoder.inference(feature)

        sentences = []
        target_sentences = []

        for i, result in enumerate(inference_results):
            words = []
            for word_id in result:
                word = vocab.idx2word[word_id.item()]
                words.append(word)
                if word == '<end>':
                    break

            try:
                words.remove('<start>')
            except Exception:
                pass

            try:
                words.remove('<end>')
            except Exception:
                pass

            sentences.append(' '.join(words))

        result = {}
        result["duplicated"] = False
        result["album_id"] = album_ids[0]
        result["photo_sequence"] = photo_sequence
        result["story_text_normalized"] = sentences[0] + " " + sentences[1] + " " + sentences[2] + " " + sentences[3] + " " + sentences[4]

        #print(result["story_text_normalized"])
        results.append(result)

    avg_loss += loss.item()
    loss /= (args.batch_size * 5)

    # Print log info
    if bi % args.log_step == 0:
        print('Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' %(bi, total_step, loss.item(), np.exp(loss.item())))

avg_loss /= (args.batch_size * total_step * 5)
print('Average Loss: %.4f, Average Perplexity: %5.4f' %(avg_loss, np.exp(avg_loss)))


for i in reversed(range(len(results))):
    if not results[i]["duplicated"]:
        for j in range(i):
            if np.array_equal(results[i]["photo_sequence"], results[j]["photo_sequence"]):
                results[j]["duplicated"] = True

filtered_res = []
for result in results:
    if not result["duplicated"]:
        del result["duplicated"]
        filtered_res.append(result)

print("Total story size : %d" %(len(filtered_res)))

evaluation_info = {}
evaluation_info["version"] = "initial version"

output = {}
output["team_name"] = "SnuBiVtt"
output["evaluation_info"] = evaluation_info
output["output_stories"] = filtered_res

with open(result_path, "w") as json_file:
    #json_file.write(json.dumps(output, ensure_ascii=False))
    json_file.write(json.dumps(output))

subprocess.call(["java", "-jar", challenge_dir + "runnable_jar/EvalVIST.jar", "-testFile", result_path, "-gsFile", sis_path])

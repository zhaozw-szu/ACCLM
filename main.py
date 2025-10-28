import json
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import matplotlib.pyplot as plt
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from data import num_classes_brand, label_dict_brand, display_labels_brand, label_dict_platform

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------
class HAN(nn.Module):
    def __init__(self, max_sentence_num, max_sentence_length, num_classes, vocab_size, embedding_size, hidden_size):
        super(HAN, self).__init__()
        self.vocab_size = vocab_size
        self.max_sentence_num = max_sentence_num
        self.max_sentence_length = max_sentence_length
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.bn = nn.BatchNorm1d(self.max_sentence_length)
        self.sentence_encoder = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.document_encoder = nn.GRU(input_size=self.hidden_size * 2, hidden_size=self.hidden_size, num_layers=1,
                                       batch_first=True, bidirectional=True)
        self.attention_layer1 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(self.hidden_size * 2, 1),
            nn.Tanh()
        )
        self.attention_layer2 = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(self.hidden_size * 2, 1),
            nn.Tanh()
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.hidden_size * 2, self.num_classes)
        )
        self.bn = nn.BatchNorm1d(self.max_sentence_length)
        self.bns = nn.BatchNorm1d(self.max_sentence_num)

    def forward(self, x):
        B, _sentence_num, _sentence_len = x.shape
        x = x.view(-1, self.max_sentence_length)
        sentence_lengths = (x != PAD).sum(dim=1).to(torch.device('cpu'))
        document_lengths = (sentence_lengths.view(B, _sentence_num, 1) != 1).sum(dim=1).view(-1).to(torch.device('cpu'))
        x = self.embedding(x)
        x = self.bn(x)
        pack_x = torch.nn.utils.rnn.pack_padded_sequence(x, sentence_lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        sentence_encoding, _ = self.sentence_encoder(pack_x)
        sentence_encoding, _ = torch.nn.utils.rnn.pad_packed_sequence(sentence_encoding, batch_first=True)
        attention_weight = self.attention_layer1(sentence_encoding)
        attention_weight = torch.softmax(attention_weight, dim=1)
        sentence_vector = torch.sum(attention_weight * sentence_encoding, dim=1)
        sentence_vector = sentence_vector.view(-1, self.max_sentence_num, self.hidden_size * 2)
        sentence_vector = self.bns(sentence_vector)
        pack_s = torch.nn.utils.rnn.pack_padded_sequence(sentence_vector, document_lengths.cpu(), batch_first=True,
                                                         enforce_sorted=False)
        document_encoding, _ = self.document_encoder(pack_s)
        document_encoding, _ = torch.nn.utils.rnn.pad_packed_sequence(document_encoding, batch_first=True)
        attention_weight = self.attention_layer2(document_encoding)
        attention_weight = torch.softmax(attention_weight, dim=1)
        document_vector = torch.sum(attention_weight * document_encoding, dim=1)
        output = self.output_layer(document_vector)
        return output


def make_data(train_Datas, test_Datas, max_sentence_num, max_sentence_len):
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []

    def pad_or_splice(sentence, max_sentence_len=max_sentence_len):
        if len(sentence) > max_sentence_len - 1:
            sentence = sentence[:max_sentence_len - 1] + [EOS_TAG]
        elif len(sentence) == max_sentence_len - 1:
            sentence = sentence + [EOS_TAG]
        elif len(sentence) == 1 and sentence[0] == EOS_TAG:
            sentence = sentence + [PAD_TAG] * (max_sentence_len - len(sentence))
        else:
            sentence = sentence + [EOS_TAG] + [PAD_TAG] * (max_sentence_len - len(sentence) - 1)
        return sentence

    def transform(doc, max_sentence_num=max_sentence_num):
        if len(doc) > max_sentence_num:
            doc = doc[:max_sentence_num]
        elif len(doc) < max_sentence_num:
            doc = doc + [EOS_TAG] * (max_sentence_num - len(doc))
        doc = [d.split("/") for d in doc]
        new_doc = []
        for lst in doc:
            new_lst = []
            for idx, item in enumerate(lst):
                if item.isdigit():
                    if "@" in lst[idx - 1]:
                        item = item.zfill(12)
                    digits = [item[i:i + split_len] for i in range(0, len(item), split_len)]
                    new_lst += digits
                else:
                    new_lst.append(item)
            new_doc.append(new_lst)
        doc = new_doc
        doc = [pad_or_splice(d) for d in doc]
        doc_idx = [dictionary.doc2idx(words, UNK) for words in doc]
        return doc_idx

    for data in tqdm(train_Datas):
        train_inputs.append(transform(data[0]))
        train_labels.append(data[1])
    for data in tqdm(test_Datas):
        test_inputs.append(transform(data[0]))
        test_labels.append(data[1])
    return train_inputs, train_labels, test_inputs, test_labels


def get_jsonrange(dd, dataset):
    train_data_filename = []
    train_data, test_data = load_data_VISION(dataset=dataset, seed=SEED, rate=0.8,
                                             dd=dd) if dataset == "VISION" else None
    for i in range(len(train_data)):
        train_data_filename.append(train_data[i][2])
    dictionary = Dictionary()
    jsonpath = "container_json/VISION.json" if dataset == "VISION" else None
    all_data = json.load(open(jsonpath, "r", encoding="utf-8"))
    for idx, d in enumerate(tqdm(all_data)):
        if d["filename"] in train_data_filename:
            data = d["data"]
            data = [s.replace(" ", "-").replace(":", "/").replace("&", "/").replace("=", "/") for s in
                    data]
            data = [s.split("/") for s in data]
            dictionary.add_documents(data)
    dictionary.filter_extremes(no_below=3, no_above=1, keep_n=None)
    dictionary.add_documents([['<UNK>', '<EOS>', '<PAD>']])
    dictionary_dict = {'token2id': dictionary.token2id, 'dfs': dictionary.dfs}
    kk = dictionary_dict["token2id"].keys()
    kk = [d for d in kk if d.isdigit()]
    for k in kk:
        try:
            k = str(k)
            v = dictionary_dict["token2id"][k]
            del dictionary_dict["token2id"][k]
            del dictionary_dict["dfs"][str(v)]
        except:
            continue
    # 00~99
    splitnum = [str(i) for i in range(0, 100)]
    splitnum.extend([str(i).zfill(2) for i in range(10)])

    now_idx = len(list(dictionary_dict["token2id"].keys()))
    for idx, kk in enumerate(dictionary_dict["token2id"].keys()):
        dictionary_dict["token2id"][kk] = idx
    for idx, k in enumerate(splitnum):
        dictionary_dict["token2id"][k] = now_idx + idx
    json.dump(dictionary_dict, open(f"dictionary.json", "w"), indent=2)
# ----------------------------------------------------------------

def load_data_VISION(dataset="VISION", seed=SEED, rate=0.8, dd=""):
    train_result = []
    test_result = []
    label_dict = label_dict_brand[dataset]
    jsonpath1 = f"container_json/{dataset}.json"
    all_data1 = json.load(open(jsonpath1, "r", encoding="utf-8"))
    for idx in range(len(all_data1)):
        if "WA" in all_data1[idx]["filename"] or "YT" in all_data1[idx]["filename"]:
            all_data1[idx]["label"] = "YT" if "YT" in all_data1[idx]["filename"] else "WA"
        else:
            all_data1[idx]["label"] = "non-SN"
    if dd == "non-SN":
        # 只在non-SN上
        all_data = [[d for d in all_data1 if d["label"] == "non-SN"]]
    elif dd == "all":
        # 在non-SN和所有平台上
        all_data = [[d for d in all_data1 if d["label"] == l] for l in label_dict_platform[dataset].keys()]
    elif dd == "!non-SN":
        # 只在YT和WA上
        all_data = [[d for d in all_data1 if d["label"] == l] for l in ["YT", "WA"]]
    else:
        raise Exception("dd error")

    for data in all_data:
        data_list = [
            [d for d in data if d["filename"].split("/")[-2].split("_")[1] == l] for l in
            label_dict_brand["VISION"].keys()
        ]
        for adevice in data_list:
            random.shuffle(adevice)
            for d in adevice[:int(len(adevice) * rate)]:
                label = d["filename"].split("/")[-2].split("_")[1]
                data = d["data"]
                filename = d["filename"]
                train_result.append([data, label_dict[label], filename])
            for d in adevice[int(len(adevice) * rate):]:
                label = d["filename"].split("/")[-2].split("_")[1]
                data = d["data"]
                filename = d["filename"]
                test_result.append([data, label_dict[label], filename])
    return train_result, test_result

def test(model, test_loader, criterion, dataset, device, dd, log_name):
    display_labels = display_labels_brand[dataset]
    dict_path = f'weights/{dataset}_{dd}_best.pkl'
    model.load_state_dict(torch.load(dict_path, map_location=device))
    num = 0
    correct = 0.0
    y_label = np.array([])
    y_pred = np.array([])
    model.eval()
    test_loader = tqdm(test_loader)
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            pred_classes = torch.max(pred, dim=1)[1]
            test_loader.desc = "pred:{} label:{}".format(pred_classes.cpu().tolist(), batch_y.cpu().tolist())
            correct += (pred_classes == batch_y).sum()
            y_label = np.concatenate((y_label, batch_y.cpu().numpy()))
            y_pred = np.concatenate((y_pred, pred_classes.cpu().numpy()))
            num += len(batch_y)
        acc = correct.item() / num
        print(f'loss: {loss.item():.12f}  | acc: {acc * 100:.2f}%')
        accuracy = accuracy_score(y_label, y_pred)
        balanced_acc = balanced_accuracy_score(y_label, y_pred)
        print("Accuracy:", accuracy)
        print("Balanced Accuracy:", balanced_acc)
        print("macro f1 score: ", get_f1_macro(y_label, y_pred))
        print(display_labels)
        get_f1_macro_detail(y_label, y_pred, num_classes_brand[dataset])


def get_f1_macro(y_pred, y_label):
    return f1_score(y_label, y_pred, average="macro")

def get_f1_macro_detail(y_label, y_pred, N):
    f1_scores = f1_score(y_label, y_pred, labels=np.arange(N), average=None)
    print("F1 score for each category:", f1_scores)

if __name__ == '__main__':
    dataset = "VISION"
    epochs, batch_size = 10, 6
    num_classes = num_classes_brand[dataset]
    dd_list = ["non-SN", "!non-SN", "all"]
    dd = dd_list[0]
    get_jsonrange(dd, dataset)
    split_len = 2
    with open('dictionary.json', 'r') as f:
        dictionary_dict = json.load(f)
    dictionary = Dictionary()
    dictionary.dfs = dictionary_dict['dfs']
    dictionary.token2id = dictionary_dict['token2id']
    UNK_TAG = "<UNK>"
    PAD_TAG = "<PAD>"
    EOS_TAG = "<EOS>"
    UNK = dictionary.token2id[UNK_TAG]
    PAD = dictionary.token2id[PAD_TAG]
    EOS = dictionary.token2id[EOS_TAG]
    size = [220, 80]
    max_sentence_num, max_sentence_length = size[0], size[1]
    hidden_size, embedding_size = 96, 1024
    vocab_size = len(dictionary)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data, test_data = load_data_VISION(dataset=dataset, seed=SEED, rate=0.8, dd=dd)
    _, _, test_inputs, test_labels = make_data(train_data, test_data, max_sentence_num=max_sentence_num, max_sentence_len=max_sentence_length)
    test_inputs, test_labels = torch.LongTensor(test_inputs), torch.LongTensor(test_labels)
    test_dataset = Data.TensorDataset(test_inputs, test_labels)
    test_loader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    han = HAN(max_sentence_num=max_sentence_num, max_sentence_length=max_sentence_length, num_classes=num_classes,
              vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size)
    han.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    print("Test start!")
    test(han, test_loader, criterion, dataset, device, dd=dd, log_name='test')
    print("Test complete!")
    



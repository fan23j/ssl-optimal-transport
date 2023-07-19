import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from gensim.models import KeyedVectors, FastText
from sklearn.metrics import average_precision_score

from ..base_trainer import BaseTrainer

def get_word_vector(word, word2vec):
    # Check if the word is in the vocabulary
    if word in word2vec.key_to_index:
        return word2vec[word]
    # If the word is not in the vocab, treat it as a compound word
    else:
        words = word.split(" ")
        word_vecs = [word2vec[w] for w in words if w in word2vec.key_to_index]
        if not word_vecs:
            raise ValueError(f"None of the words in {word} are in the vocabulary.")
        return np.mean(word_vecs, axis=0)

def get_label_and_negative_vectors(cfg, labels_list, word2vec):
    num_classes = len(labels_list)
    
    # Initialize vectors
    label_vectors = np.zeros((num_classes, cfg.MODEL.OUTPUT_FEATURES))
    negative_vectors = np.zeros((num_classes, cfg.MODEL.OUTPUT_FEATURES))

    for i, label in enumerate(labels_list):
        try:
            label_vectors[i, :] = get_word_vector(label, word2vec)
        except ValueError as e:
            print(e)

    all_labels = set(word2vec.key_to_index.keys())
    for i, label in enumerate(labels_list):
        negative_labels = all_labels - set([label])
        for negative_label in negative_labels:
            try:
                negative_vectors[i, :] += get_word_vector(negative_label, word2vec)
            except ValueError as e:
                print(e)
        # Getting the mean of the negative vectors
        negative_vectors[i, :] /= len(negative_labels)

    return label_vectors, negative_vectors

class SimCLRClassifyAnythingTrainer(BaseTrainer):
    def __init__(self, cfg, model, optimizer, lr_scheduler):
        super(SimCLRClassifyAnythingTrainer, self).__init__(
            cfg, model, optimizer, lr_scheduler
        )

        print("load word2vec model from " + cfg.MODEL.WORD2VEC)
        # self.word2vec = KeyedVectors.load_word2vec_format(
        #     cfg.MODEL.WORD2VEC, binary=False
        # )

        # print("constructing labels vector...")
        # labels_list = cfg.DATASET.LABELS
        # self.label_vectors, self.negative_vectors = get_label_and_negative_vectors(cfg, labels_list, self.word2vec)
        # import pudb; pudb.set_trace()
        self.label_vectors = torch.load('coco_label_vectors.pt')
        self.negative_vectors = torch.load('coco_negative_vectors.pt')

        with torch.no_grad():
            self.label_vectors = (
                torch.tensor(self.label_vectors).float().cuda(non_blocking=True)
            )
            self.negative_vectors = (
                torch.tensor(self.negative_vectors).float().cuda(non_blocking=True)
            )

    def train(self, epoch, data_loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_correct_1, total_correct_5, total_num, data_bar = (
            0.0,
            0.0,
            0,
            tqdm(data_loader),
        )
        average_loss_states = {}
        figure = None

        with torch.enable_grad() if is_train else torch.no_grad():
            for it, batch in enumerate(data_bar):
                data = batch["out_1"]
                target = batch["target"]
                data, target = (
                    data.cuda(non_blocking=True),
                    target.cuda(non_blocking=True),
                )
                features = self.model(data)

                loss, loss_states, cosim_softmax = self.loss(
                    features=features, labels_vector=self.label_vectors, negative_vectors=self.negative_vectors, targets=target
                )

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                elif it == 0:
                    # save cosim_softmax for first batch in val
                    plt.imshow(cosim_softmax.cpu().detach().numpy(), cmap="viridis")
                    plt.colorbar()
                    figure = plt

                total_num += data.size(0)
                # Accumulate average_loss_states
                for k, v in loss_states.items():
                    if k not in average_loss_states:
                        average_loss_states[k] = 0.0
                    average_loss_states[k] += v.item()

                # Create a string of loss states
                loss_state_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in loss_states.items()
                )

                # apply sigmoid to cosim_softmax to get predictions between 0 and 1
                preds = torch.sigmoid(cosim_softmax)

                average_precisions = []
                for class_idx in range(target.shape[1]):
                    class_preds = preds[:, class_idx].cpu().detach().numpy()
                    class_targets = target[:, class_idx].cpu().detach().numpy()
                    average_precision = average_precision_score(class_targets, class_preds)
                    average_precisions.append(average_precision)
                mAP = np.mean(average_precisions)

                data_bar.set_description(
                    "{} Epoch: [{}/{}] {} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                        "Train" if is_train else "Test",
                        epoch,
                        self.cfg.TRAIN.EPOCHS,
                        loss_state_str,
                        mAP * 100,
                        total_correct_5 / total_num * 100,
                    )
                )
        self.lr_scheduler.step()
        # Average the accumulated loss_states
        for k in average_loss_states:
            average_loss_states[k] /= len(data_loader)

        average_loss_states["ACC@1"] = total_correct_1 / total_num * 100
        average_loss_states["ACC@5"] = total_correct_5 / total_num * 100
        if not is_train:
            average_loss_states["figure"] = figure
        return average_loss_states

    def val(self, epoch, test_data_loader):
        return self.train(epoch, test_data_loader, is_train=False)

    def test(self, data_loader):
        self.model.eval()
        total_correct_1, total_correct_5, total_num, data_bar = (
            0.0,
            0.0,
            0,
            tqdm(data_loader),
        )
        predictions = []
        labels = []
        average_loss_states = {}

        with torch.no_grad():
            for batch in data_bar:
                data = batch["out_1"]
                target = batch["target"]
                data, target = data.cuda(non_blocking=True), target.cuda(
                    non_blocking=True
                )

                out = self.model(data)
                _, predicted = torch.max(out.data, 1)
                predictions.extend(predicted.cpu().tolist())
                labels.extend(target.cpu().tolist())

                _, loss_states = self.loss(out, target)

                total_num += data.size(0)
                # Accumulate average_loss_states
                for k, v in loss_states.items():
                    if k not in average_loss_states:
                        average_loss_states[k] = 0.0
                    average_loss_states[k] += v.item()

                # Create a string of loss states
                loss_state_str = ", ".join(
                    f"{k}: {v:.4f}" for k, v in loss_states.items()
                )

                prediction = torch.argsort(out, dim=-1, descending=True)
                total_correct_1 += torch.sum(
                    (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()
                total_correct_5 += torch.sum(
                    (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()
                ).item()

                data_bar.set_description(
                    "{}: {} ACC@1: {:.2f}% ACC@5: {:.2f}%".format(
                        "Test",
                        loss_state_str,
                        total_correct_1 / total_num * 100,
                        total_correct_5 / total_num * 100,
                    )
                )

        return zip(labels, predictions)

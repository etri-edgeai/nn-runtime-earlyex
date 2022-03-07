import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
import seaborn as sn
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, auc, ConfusionMatrixDisplay
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

import matplotlib.pyplot as plt

class CELoss(object):
    def compute_bin_boundaries(self, probabilities = np.array([])):
        #uniform bin spacing
        if probabilities.size == 0:
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]
        else:
            #size of bins 
            bin_n = int(self.n_data/self.n_bins)

            bin_boundaries = np.array([])

            probabilities_sort = np.sort(probabilities)  

            for i in range(0,self.n_bins):
                bin_boundaries = np.append(bin_boundaries,probabilities_sort[i*bin_n])
            bin_boundaries = np.append(bin_boundaries,1.0)

            self.bin_lowers = bin_boundaries[:-1]
            self.bin_uppers = bin_boundaries[1:]

    def get_probabilities(self, output, labels, logits):
        #If not probabilities apply softmax!
        if logits:
            self.probabilities = softmax(output, axis=1)
        else:
            self.probabilities = output

        self.labels = labels
        self.confidences = np.max(self.probabilities, axis=1)
        self.predictions = np.argmax(self.probabilities, axis=1)
        self.accuracies = np.equal(self.predictions,labels)

    def binary_matrices(self):
        idx = np.arange(self.n_data)
        #make matrices of zeros
        pred_matrix = np.zeros([self.n_data,self.n_class])
        label_matrix = np.zeros([self.n_data,self.n_class])
        #self.acc_matrix = np.zeros([self.n_data,self.n_class])
        pred_matrix[idx,self.predictions] = 1
        label_matrix[idx,self.labels] = 1

        self.acc_matrix = np.equal(pred_matrix, label_matrix)

    def compute_bins(self, index = None):
        self.bin_prop = np.zeros(self.n_bins)
        self.bin_acc = np.zeros(self.n_bins)
        self.bin_conf = np.zeros(self.n_bins)
        self.bin_score = np.zeros(self.n_bins)

        if index == None:
            confidences = self.confidences
            accuracies = self.accuracies
        else:
            confidences = self.probabilities[:,index]
            accuracies = self.acc_matrix[:,index]


        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = np.greater(confidences,bin_lower.item()) * np.less_equal(confidences,bin_upper.item())
            self.bin_prop[i] = np.mean(in_bin)

            if self.bin_prop[i].item() > 0:
                self.bin_acc[i] = np.mean(accuracies[in_bin])
                self.bin_conf[i] = np.mean(confidences[in_bin])
                self.bin_score[i] = np.abs(self.bin_conf[i] - self.bin_acc[i])

class MaxProbCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        self.n_bins = n_bins
        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().compute_bins()

#http://people.cs.pitt.edu/~milos/research/AAAI_Calibration.pdf
class ECELoss(MaxProbCELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_score)

class MCELoss(MaxProbCELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.max(self.bin_score)

#https://arxiv.org/abs/1905.11001
#Overconfidence Loss (Good in high risk applications where confident but wrong predictions can be especially harmful)
class OELoss(MaxProbCELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        super().loss(output, labels, n_bins, logits)
        return np.dot(self.bin_prop,self.bin_conf * np.maximum(self.bin_conf-self.bin_acc,np.zeros(self.n_bins)))

#https://arxiv.org/abs/1904.01685
class SCELoss(CELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        sce = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().compute_bin_boundaries()
        super().get_probabilities(output, labels, logits)
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bins(i)
            sce += np.dot(self.bin_prop,self.bin_score)

        return sce/self.n_class

class TACELoss(CELoss):
    def loss(self, output, labels, threshold = 0.01, n_bins = 15, logits = True):
        tace = 0.0
        self.n_bins = n_bins
        self.n_data = len(output)
        self.n_class = len(output[0])

        super().get_probabilities(output, labels, logits)
        self.probabilities[self.probabilities < threshold] = 0
        super().binary_matrices()

        for i in range(self.n_class):
            super().compute_bin_boundaries(self.probabilities[:,i]) 
            super().compute_bins(i)
            tace += np.dot(self.bin_prop,self.bin_score)

        return tace/self.n_class

#create TACELoss with threshold fixed at 0
class ACELoss(TACELoss):
    def loss(self, output, labels, n_bins = 15, logits = True):
        return super().loss(output, labels, 0.0 , n_bins, logits)

class ConfidenceHistogram(MaxProbCELoss):
    def plot(self, output, labels, n_bins = 20, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)
        #scale each datapoint
        n = len(labels)
        w = np.ones(n)/n

        plt.rcParams["font.family"] = "serif"
        #size and axis limits 
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)    
        #plot histogram
        plt.hist(self.confidences,n_bins,weights = w,color='b',range=(0.0,1.0),edgecolor = 'k')

        #plot vertical dashed lines
        acc = np.mean(self.accuracies)
        conf = np.mean(self.confidences)                
        plt.axvline(x=acc, color='tab:grey', linestyle='--', linewidth = 3)
        plt.axvline(x=conf, color='tab:grey', linestyle='--', linewidth = 3)
        if acc > conf:
            plt.text(acc+0.03,0.9,'Acc',rotation=90,fontsize=11)
            plt.text(conf-0.07,0.9,'Conf',rotation=90, fontsize=11)
        else:
            plt.text(acc-0.07,0.9,'Acc',rotation=90,fontsize=11)
            plt.text(conf+0.03,0.9,'Conf',rotation=90, fontsize=11)

        plt.ylabel('% of Samples',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.tight_layout()
        if title is not None:
            plt.title(title,fontsize=16)
        return plt

class ReliabilityDiagram(MaxProbCELoss):
    def plot(self, output, labels, n_bins = 10, logits = True, title = None):
        super().loss(output, labels, n_bins, logits)

        #computations
        delta = 1.0/n_bins
        x = np.arange(0,1,delta)
        mid = np.linspace(delta/2,1-delta/2,n_bins)
        error = np.abs(np.subtract(mid,self.bin_acc))

        plt.rcParams["font.family"] = "serif"
        #size and axis limits
        plt.figure(figsize=(3,3))
        plt.xlim(0,1)
        plt.ylim(0,1)
        #plot grid
        plt.grid(color='tab:grey', linestyle=(0, (1, 5)), linewidth=1,zorder=0)
        #plot bars and identity line
        plt.bar(x, self.bin_acc, color = 'b', width=delta,align='edge', edgecolor = 'k',label='Outputs',zorder=5)
        plt.bar(x, error, bottom=np.minimum(self.bin_acc,mid), color = 'mistyrose', alpha=0.5, width=delta,align='edge',edgecolor = 'r',hatch='/',label='Gap',zorder=10)
        ident = [0.0, 1.0]
        plt.plot(ident,ident,linestyle='--',color='tab:grey',zorder=15)
        #labels and legend
        plt.ylabel('Accuracy',fontsize=13)
        plt.xlabel('Confidence',fontsize=13)
        plt.legend(loc='upper left',framealpha=1.0,fontsize='medium')
        if title is not None:
            plt.title(title,fontsize=16)
        plt.tight_layout()

        return plt


def confused(output, labels, num_class, name):
        pred = np.argmax(output, axis=1)
        cfm = confusion_matrix(labels, pred)
        cfmd = ConfusionMatrixDisplay(confusion_matrix=cfm)
        cfmd.plot()
        plt.savefig(name, bbox_inches='tight') 
        plt.clf()




def roc_curved(output, labels, num_class, name):
    pred = np.argmax(output, axis=1)
    conf = np.amax(output, axis=1)
    p = []
    opt_thres = np.zeros(num_class)
    for n in range(num_class):
        lw = 2
        # conditions = [(labels[pred == n] == n), (labels[pred == n] != n)]
        conditions = [(labels[pred == n] == n), (labels[pred == n] != n)]
        choices = [1,0]
        y = np.select(conditions, choices)
        # print(y)

        X = conf[pred == n]

        fpr, tpr, thresholds = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)
        # print(fpr, tpr, thresholds)
        fpr_micro, tpr_micro, _ = roc_curve(y.ravel(), X.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("RoC for class {}".format(n))
        plt.legend(loc="lower right")
        namef = name +'_'+ str(n) + '.png' 
        plt.savefig(namef, bbox_inches='tight')
        gmeans =np.sqrt(tpr * (1-fpr))
        opt_thres[n] = thresholds[np.argmax(gmeans)]
        # opt_thres[n] = thresholds[np.argmax(tpr - fpr)]
        plt.clf()
    return np.amax(opt_thres)

def roc_curved2(output, labels, num_class, name):
    pred = np.argmax(output, axis=1)
    conf = np.amax(output, axis=1)
    p = []
    acc = np.equal(pred, labels)
    y = acc
    X = conf

    fpr, tpr, thresholds = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, 
        tpr, 
        color="darkorange", 
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} Curve".format(name))
    plt.legend(loc="lower right")
    namef = name +'.png' 
    plt.savefig(namef, bbox_inches='tight')
    gmeans =np.sqrt(tpr * (1-fpr))
    opt_thres = thresholds[np.argmax(gmeans)]
    plt.clf()
    return opt_thres

def roc_curved3(output, labels, num_class, name, branch=0, total=4):
    pred = np.argmax(output, axis=1)
    conf = np.amax(output, axis=1)
    p = []
    acc = np.equal(pred, labels)
    y = acc
    X = conf

    fpr, tpr, thresholds = roc_curve(y, X)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, 
        tpr, 
        color="darkorange", 
        lw=2,
        label="ROC curve (area = %0.2f)" % roc_auc
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("{} Curve".format(name))
    plt.legend(loc="lower right")
    namef = name +'.png' 
    plt.savefig(namef, bbox_inches='tight')
    gmeans = np.sqrt(tpr * (1-fpr))
    # print("gmeans: ", gmeans)
    # opt_thres = thresholds[np.argmax(gmeans)]
    opt_thres = thresholds[np.argmax(np.where(fpr<=0.05))]
    tot = total - 1
    branch_discount = opt_thres * (-branch) / (tot)
    opt_thres = opt_thres + branch_discount
    plt.clf()
    return opt_thres

import numpy as np


class ConfusionToStats(object):

    def __init__(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix
        # hier sollte es 2x2 sein, check it!

        self.tp = confusion_matrix[0, 0]
        self.tn = confusion_matrix[1, 1]
        self.fp = confusion_matrix[0, 1]
        self.fn = confusion_matrix[1, 0]

        self._tpr = None
        self._tnr = None
        self._ppv = None
        self._npv = None
        self._fnr = None
        self._fpr = None
        self._fdr = None
        self._for_ = None
        self._acc = None
        self._f1 = None
        self._bm = None
        self._mk = None

        self.calculate_stats()

    @classmethod
    def from_all_but_one(cls, confusion_matrix, idx):

        confusion_matrix_small = np.zeros((2, 2))

        off_elements = np.setdiff1d(np.arange(np.shape(confusion_matrix)[0]), idx)

        tp = confusion_matrix[idx, idx]
        tn = confusion_matrix[off_elements, off_elements]
        fp = confusion_matrix[idx, off_elements]
        fn = confusion_matrix[off_elements, idx]

        confusion_matrix_small[0, 0] = tp
        confusion_matrix_small[1, 1] = tn
        confusion_matrix_small[0, 1] = fp
        confusion_matrix_small[1, 0] = fn

        return cls(confusion_matrix_small)

    @classmethod
    def from_two_classes(cls):
        pass

    def calculate_stats(self):
        self._tpr = 1.*self.tp / (self.tp + self.fn)
        self._tnr = 1.*self.tn / (self.tn + self.fp)
        self._ppv = 1. * self.tp / (self.tp + self.fp)
        self._npv = 1. * self.tn / (self.tn + self.fn)
        self._fnr = 1. * self.fn / (self.fn + self.tp)
        self._fpr = 1. * self.fp / (self.fp + self.tn)
        self._fdr = 1. * self.fp / (self.fp + self.tp)
        self._for_ = 1. * self.fn / (self.fn + self.tn)
        self._acc = 1. * (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        self._f1 = 1. * 2*self.tp / (2*self.tp + self.fp + self.fn)
        self._bm = self._tpr + self._tnr - 1
        self._mk = self._ppv + self._npv - 1

    @property
    def tpr(self):
        if self._tpr is None:
            raise TypeError
        else:
            return self._tpr

    @property
    def tnr(self):
        if self._tnr is None:
            raise TypeError
        else:
            return self._tnr

    @property
    def ppv(self):
        if self._ppv is None:
            raise TypeError
        else:
            return self._ppv

    @property
    def npv(self):
        if self._npv is None:
            raise TypeError
        else:
            return self._npv

    @property
    def fnr(self):
        if self._fnr is None:
            raise TypeError
        else:
            return self._fnr

    @property
    def fpr(self):
        if self._fpr is None:
            raise TypeError
        else:
            return self._fpr

    @property
    def fdr(self):
        if self._fdr is None:
            raise TypeError
        else:
            return self._fdr

    @property
    def for_(self):
        if self._for_ is None:
            raise TypeError
        else:
            return self._for_

    @property
    def acc(self):
        if self._acc is None:
            raise TypeError
        else:
            return self._acc

    @property
    def f1(self):
        if self._f1 is None:
            raise TypeError
        else:
            return self._f1

    @property
    def bm(self):
        if self._bm is None:
            raise TypeError
        else:
            return self._bm

    @property
    def mk(self):
        if self._mk is None:
            raise TypeError
        else:
            return self._mk

    def make_plots(self):
        raise NotImplementedError

    def print_prec_recall(self):
        print(self.confusion_matrix)
        print('Recall: {}'.format(self._tpr))
        print('Precision: {}'.format(self._ppv))

    def __repr__(self):
        return "Not done yet."


if __name__ == "__main__":
    cm = np.zeros((2, 2))
    cm[0, 0] = 20
    cm[1, 1] = 12
    cm[1, 0] = 6
    cm[0, 1] = 4
    cts = ConfusionToStats(cm)
    cts.calculate_stats()
    print(cts.tpr)

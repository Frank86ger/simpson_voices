
import numpy as np
from validation_utils.binom_stats import BinomialEstimator


class ConfusionToStats(object):

    def __init__(self, confusion_matrix):
        """yada
        yada

        Parameters
        ----------
        confusion_matrix : array-like
            [[TP, FP],
             [FN, TN]]
        """
        self.confusion_matrix = np.array(confusion_matrix, dtype=int)
        # TODO check for 2x2

        self.tp = self.confusion_matrix[0, 0]
        self.tn = self.confusion_matrix[1, 1]
        self.fp = self.confusion_matrix[0, 1]
        self.fn = self.confusion_matrix[1, 0]

        self.pp = self.tp + self.fp
        self.pn = self.tn + self.fn
        self.ap = self.tp + self.fn
        self.an = self.tn + self.fp

        self.all = self.tp + self.tn + self.fp + self.fn

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
        raise NotImplementedError

    def calculate_stats(self):
        self._tpr = BinomialEstimator(self.ap, self.tp)
        self._tnr = BinomialEstimator(self.an, self.tn)
        self._ppv = BinomialEstimator(self.pp, self.tp)
        self._npv = BinomialEstimator(self.pn, self.tn)
        self._fnr = BinomialEstimator(self.ap, self.fn)
        self._fpr = BinomialEstimator(self.an, self.fp)
        self._fdr = BinomialEstimator(self.pp, self.fp)
        self._for_ = BinomialEstimator(self.pn, self.fn)
        self._acc = BinomialEstimator(self.tp + self.tn + self.fp + self.fn, self.tp + self.tn)
        self._f1 = BinomialEstimator(2*self.tp + self.fp + self.fn, 2*self.tp)

        self._bm = self._tpr.p_expected_biased + self._tnr.p_expected_biased
        self._mk = self._ppv.p_expected_biased + self._npv.p_expected_biased

    @property
    def tpr(self):
        if self._tpr is None:
            raise TypeError
        else:
            return self._tpr.p_expected_biased

    @property
    def tnr(self):
        if self._tnr is None:
            raise TypeError
        else:
            return self._tnr.p_expected_biased

    @property
    def ppv(self):
        if self._ppv is None:
            raise TypeError
        else:
            return self._ppv.p_expected_biased

    @property
    def npv(self):
        if self._npv is None:
            raise TypeError
        else:
            return self._npv.p_expected_biased

    @property
    def fnr(self):
        if self._fnr is None:
            raise TypeError
        else:
            return self._fnr.p_expected_biased

    @property
    def fpr(self):
        if self._fpr is None:
            raise TypeError
        else:
            return self._fpr.p_expected_biased

    @property
    def fdr(self):
        if self._fdr is None:
            raise TypeError
        else:
            return self._fdr.p_expected_biased

    @property
    def for_(self):
        if self._for_ is None:
            raise TypeError
        else:
            return self._for_.p_expected_biased

    @property
    def acc(self):
        if self._acc is None:
            raise TypeError
        else:
            return self._acc.p_expected_biased

    @property
    def f1(self):
        if self._f1 is None:
            raise TypeError
        else:
            return self._f1.p_expected_biased

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

import numpy as np
import scipy.special as special
from bokeh.plotting import figure, show


class BinomialEstimator(object):
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.p_expected_biased = 1.*k/n
        self.p_expected_unbiased = self.calc_p_expected_unbiased()
        self.left_90, self.right_90 = self.get_confidence_interval(0.9)
        self.left_95, self.right_95 = self.get_confidence_interval(0.95)
        self.left_99, self.right_99 = self.get_confidence_interval(0.99)

    def binom_dist(self, p):
        # B(k | n, p)
        return special.binom(self.n, self.k) * p ** self.k * (1 - p) ** (self.n - self.k)

    def conditional_p_binom(self, p):
        # B(p | n, k)
        return self.binom_dist(p) * (self.n + 1)

    def calc_p_expected_unbiased(self):
        return 1.*(self.k + 1) / (self.n + 2)

    def calc_p_variance(self):
        return (self.k + 1) * (self.k + 2) / ((self.n + 2) * (self.n + 3)) - (self.k + 1) ** 2 / (self.n + 2) ** 2

    def get_confidence_interval(self, confidence, show_plot=False):

        n_mesh = 10000
        conf_thresh = confidence / 2
        p_mesh = np.arange(n_mesh, dtype=float) / n_mesh
        p_pdf = self.conditional_p_binom(p_mesh)
        idx_expected_p = np.searchsorted(p_mesh, self.p_expected_unbiased)

        cumsum_right = np.cumsum(p_pdf[idx_expected_p:]) / n_mesh
        right_idx = np.searchsorted(cumsum_right, conf_thresh)
        right = p_mesh[right_idx + idx_expected_p - 1]

        cumsum_left = np.cumsum(p_pdf[idx_expected_p::-1]) / n_mesh
        left_idx = np.searchsorted(cumsum_left, conf_thresh)
        left = p_mesh[idx_expected_p - left_idx]

        if show_plot:
            p = figure(plot_width=600, plot_height=600, title='title')
            p.line(p_mesh, p_pdf, line_width=4, color='black')
            p.line([left, left], [0, np.max(p_pdf)], line_width=2, color='black')
            p.line([right, right], [0, np.max(p_pdf)], line_width=2, color='black')
            p.line([self.p_expected_unbiased, self.p_expected_unbiased], [0, np.max(p_pdf)], line_width=3, color='red')
            p.line([self.p_expected_biased, self.p_expected_biased], [0, np.max(p_pdf)], line_width=1, color='blue')
            show(p)

        return left, right

    def __add__(self, other):
        # TODO convolve, mean, confidence interval
        raise NotImplementedError

    def __mul__(self, other):
        # TODO int 1/|t| f1(t) f2(z/t) dt, mean, confidence interval
        raise NotImplementedError


if __name__ == "__main__":
    be = BinomialEstimator(10, 1)
    print(be.p_expected_unbiased)
    print(be.get_confidence_interval(0.8, show_plot=True))
    print(be.calc_p_variance())

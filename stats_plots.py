"""
TODO: implement bm, mk, lr+-, dor in confusion_to_stats and print here
"""

import numpy as np
from bokeh.layouts import layout
from bokeh.models import Range1d
from bokeh.models import Label
from bokeh.plotting import figure, show

from confusion_to_stats import ConfusionToStats


class StatsPlotter(object):
    def __init__(self, base_stats):
        self.base_stats = base_stats
        self.colors = {'TP': '#7CFC00',
                       'FN': '#DC143C',
                       'TN': '#32CD32',
                       'FP': '#FF0000',
                       'aP': '#D3D3D3',
                       'aN': '#A9A9A9',
                       'pP': '#FFD700',
                       'pN': '#FFA500',
                       }

        self.stats = {'TP': base_stats.tp,
                      'TN': base_stats.tn,
                      'FP': base_stats.fp,
                      'FN': base_stats.fn,
                      'pP': base_stats.pp,
                      'pN': base_stats.pn,
                      'aP': base_stats.ap,
                      'aN': base_stats.an,
                      }

        self.measure = {'TPR': base_stats.tpr,
                        'TNR': base_stats.tnr,
                        'PPV': base_stats.ppv,
                        'NPV': base_stats.npv,
                        'FNR': base_stats.fnr,
                        'FPR': base_stats.fpr,
                        'FDR': base_stats.fdr,
                        'FOR': base_stats.for_,
                        'ACC': base_stats.acc,
                        'F1': base_stats.f1,
                        'BM': base_stats.bm,
                        'MK': base_stats.mk,
                        }

    @classmethod
    def create_plot_from_confusion(cls, data_, out_path=None):
        pass

    @staticmethod
    def deg_2_cart(deg_):
        return np.cos(deg_), np.sin(deg_)

    def stats_2_deg(self, order):
        orders = {'actual': ['FN', 'FP', 'TN', 'TP'],
                  'predict': ['FP', 'FN', 'TN', 'TP'],
                  }
        sum_ = sum([self.stats[o] for o in orders[order]])
        percentages = [0.] + [1.*self.stats[o]/sum_ for o in orders[order]]
        percentiles = np.cumsum(percentages)
        start, end = percentiles[:-1], percentiles[1:]
        start = start*2*np.pi + np.pi/2
        end = end*2*np.pi + np.pi/2
        return start % (2*np.pi), end % (2*np.pi), orders[order]

    def setup_actual_plot(self):
        starts, ends, order = self.stats_2_deg('actual')
        mid_angles = (starts + 1. * (ends - starts) % (2 * np.pi) / 2) % (2 * np.pi)

        mid_angles_cum = [(mid_angles[1] + mid_angles[2]) / 2 % (2 * np.pi),
                          (mid_angles[0] + mid_angles[3]) / 2 % (2 * np.pi)]

        p = figure(plot_width=400, plot_height=400, title='ACTUAL classes split')

        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[0], end_angle=ends[0], fill_color=self.colors['FN'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[1], end_angle=ends[1], fill_color=self.colors['FP'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[2], end_angle=ends[2], fill_color=self.colors['TN'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[3], end_angle=ends[3], fill_color=self.colors['TP'])

        p.annular_wedge(x=[0], y=[0], inner_radius=1.1, outer_radius=1.5, start_angle=starts[3], end_angle=ends[3],
                        color=self.colors['aP'], line_width=1, line_color='black')
        p.annular_wedge(x=[0], y=[0], inner_radius=1.1, outer_radius=1.5, start_angle=starts[0], end_angle=ends[0],
                        color=self.colors['aP'], line_width=1, line_color='black')

        p.annular_wedge(x=[0], y=[0], inner_radius=1.5, outer_radius=1.9, start_angle=starts[1], end_angle=ends[1],
                        color=self.colors['aN'], line_width=1, line_color='black')
        p.annular_wedge(x=[0], y=[0], inner_radius=1.5, outer_radius=1.9, start_angle=starts[2], end_angle=ends[2],
                        color=self.colors['aN'], line_width=1, line_color='black')

        p.ray(x=[0, 0], y=[0, 0], length=3, angle=[ends[0], starts[3]], line_width=3, color='black')

        x_, y_ = self.deg_2_cart(mid_angles)
        texts = ["{}: {}".format(cat, self.stats[cat]) for cat in order]
        for idx in range(4):
            p.add_layout(Label(x=x_[idx]*0.6, y=y_[idx]*0.6, text=texts[idx],
                               text_align='center', text_font_size="12pt", text_color='blue'))

        conv_labels = ['FNR: {0:.2f}'.format(self.measure['FNR']),
                       'FPR: {0:.2f}'.format(self.measure['FPR']),
                       'TNR: {0:.2f}'.format(self.measure['TNR']),
                       'TPR: {0:.2f}'.format(self.measure['TPR']),
                       ]

        p.add_layout(Label(x=x_[0] * 1.4, y=y_[0] * 1.4, text=conv_labels[0],
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[1] * 1.9, y=y_[1] * 1.9, text=conv_labels[1],
                           text_align='center', text_font_size="12pt"))

        p.add_layout(Label(x=x_[2] * 1.9, y=y_[2] * 1.9, text=conv_labels[2],
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[3] * 1.4, y=y_[3] * 1.4, text=conv_labels[3],
                           text_align='center', text_font_size="12pt"))

        x_, y_ = self.deg_2_cart(mid_angles_cum)
        p.add_layout(Label(x=x_[0] * 2.3, y=y_[0] * 2.3, text='aN', text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[1] * 1.7, y=y_[1] * 1.7, text='aP', text_align='center', text_font_size="12pt"))

        p.y_range = Range1d(-2.5, 2.5, bounds=(0, None))
        p.x_range = Range1d(-2.5, 2.5, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

    def setup_predicted_plot(self):
        starts, ends, order = self.stats_2_deg('predict')
        mid_angles = (starts + 1. * (ends - starts) % (2 * np.pi) / 2) % (2 * np.pi)

        mid_angles_cum = [(mid_angles[1] + mid_angles[2]) / 2 % (2 * np.pi),
                          (mid_angles[0] + mid_angles[3]) / 2 % (2 * np.pi)]

        p = figure(plot_width=400, plot_height=400, title='PREDICTED classes split')

        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[0], end_angle=ends[0], fill_color=self.colors['FP'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[1], end_angle=ends[1], fill_color=self.colors['FN'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[2], end_angle=ends[2], fill_color=self.colors['TN'])
        p.wedge(x=[0], y=[0], radius=1., start_angle=starts[3], end_angle=ends[3], fill_color=self.colors['TP'])

        p.annular_wedge(x=[0], y=[0], inner_radius=1.1, outer_radius=1.5, start_angle=starts[3], end_angle=ends[3],
                        color=self.colors['aP'], line_width=1, line_color='black')
        p.annular_wedge(x=[0], y=[0], inner_radius=1.1, outer_radius=1.5, start_angle=starts[0], end_angle=ends[0],
                        color=self.colors['aP'], line_width=1, line_color='black')

        p.annular_wedge(x=[0], y=[0], inner_radius=1.5, outer_radius=1.9, start_angle=starts[1], end_angle=ends[1],
                        color=self.colors['aN'], line_width=1, line_color='black')
        p.annular_wedge(x=[0], y=[0], inner_radius=1.5, outer_radius=1.9, start_angle=starts[2], end_angle=ends[2],
                        color=self.colors['aN'], line_width=1, line_color='black')

        p.ray(x=[0, 0], y=[0, 0], length=3, angle=[ends[0], starts[3]], line_width=3, color='black')

        x_, y_ = self.deg_2_cart(mid_angles)
        texts = ["{}: {}".format(cat, self.stats[cat]) for cat in order]
        for idx in range(4):
            p.add_layout(Label(x=x_[idx] * 0.6, y=y_[idx] * 0.6, text=texts[idx],
                               text_align='center', text_font_size="12pt", text_color='blue'))

        conv_labels = ['FDR: {0:.2f}'.format(self.measure['FDR']),
                       'FOR: {0:.2f}'.format(self.measure['FOR']),
                       'NPV: {0:.2f}'.format(self.measure['NPV']),
                       'PPV: {0:.2f}'.format(self.measure['PPV']),
                       ]

        p.add_layout(Label(x=x_[0] * 1.4, y=y_[0] * 1.4, text=conv_labels[0],
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[1] * 1.9, y=y_[1] * 1.9, text=conv_labels[1],
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[2] * 1.9, y=y_[2] * 1.9, text=conv_labels[2],
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[3] * 1.4, y=y_[3] * 1.4, text=conv_labels[3],
                           text_align='center', text_font_size="12pt"))

        x_, y_ = self.deg_2_cart(mid_angles_cum)
        p.add_layout(Label(x=x_[0] * 2.3, y=y_[0] * 2.3, text='pN', text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=x_[1] * 1.7, y=y_[1] * 1.7, text='pP', text_align='center', text_font_size="12pt"))

        p.y_range = Range1d(-2.5, 2.5, bounds=(0, None))
        p.x_range = Range1d(-2.5, 2.5, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

        # show(p)

    def setup_confusion_plot(self):

        p = figure(plot_width=400, plot_height=400, title='Confusion')
        p.rect(x=1, y=1, width=1.9, height=1.9, fill_color=self.colors['FP'])
        p.rect(x=-1, y=1, width=1.9, height=1.9, fill_color=self.colors['TP'])
        p.rect(x=1, y=-1, width=1.9, height=1.9, fill_color=self.colors['TN'])
        p.rect(x=-1, y=-1, width=1.9, height=1.9, fill_color=self.colors['FN'])

        p.rect(x=-1, y=1, width=1.4, height=5.4, fill_color=self.colors['aP'], alpha=0.2)
        p.rect(x=1, y=1, width=1.4, height=5.4, fill_color=self.colors['aN'], alpha=0.2)
        p.rect(x=-1, y=1, width=5.4, height=1.4, fill_color=self.colors['pP'], alpha=0.2)
        p.rect(x=-1, y=-1, width=5.4, height=1.4, fill_color=self.colors['pN'], alpha=0.2)

        p.add_layout(Label(x=-1, y=1, text='TP: {}'.format(self.stats['TP']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1, y=-1, text='TN: {}'.format(self.stats['TN']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1, y=1, text='FP: {}'.format(self.stats['FP']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-1, y=-1, text='FN: {}'.format(self.stats['FN']),
                           text_align='center', text_font_size="12pt"))

        p.add_layout(Label(x=-3, y=1, text='pP: {}'.format(self.stats['pP']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-3, y=-1, text='pN: {}'.format(self.stats['pN']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-1, y=3, text='aP: {}'.format(self.stats['aP']),
                           text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1, y=3, text='aN: {}'.format(self.stats['aN']),
                           text_align='center', text_font_size="12pt"))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

    def setup_bar_plots(self):
        p = figure(plot_width=400, plot_height=400, title='Bar plots')

        # TPR / FNR
        p.hbar(left=0., right=self.measure['FNR'], y=0., height=1., color='red')
        p.hbar(left=self.measure['FNR'], right=1., y=0., height=1., color='green')
        fnr_text = 'FNR:{0:.2f}'.format(self.measure['FNR'])
        tpr_text = 'TPR:{0:.2f}'.format(self.measure['TPR'])
        p.add_layout(Label(x=-0.2, y=0, text=fnr_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-0.2, y=-0.5, text='miss rate', text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=0, text=tpr_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=-0.5, text='recall', text_align='center', text_font_size="12pt"))

        # TNR / FPR
        p.hbar(left=0., right=self.measure['FPR'], y=-2., height=1., color='red')
        p.hbar(left=self.measure['FPR'], right=1., y=-2., height=1., color='green')
        fpr_text = 'FPR:{0:.2f}'.format(self.measure['FPR'])
        tnr_text = 'TNR:{0:.2f}'.format(self.measure['TNR'])
        p.add_layout(Label(x=-0.2, y=-2, text=fpr_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-0.2, y=-2.5, text='fall-out', text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=-2, text=tnr_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=-2.5, text='specificity', text_align='center', text_font_size="12pt"))

        # PPV / FDR
        p.hbar(left=0., right=self.measure['FDR'], y=-4., height=1., color='red')
        p.hbar(left=self.measure['FDR'], right=1., y=-4., height=1., color='green')
        fdr_text = 'FDR:{0:.2f}'.format(self.measure['FDR'])
        ppv_text = 'PPV:{0:.2f}'.format(self.measure['PPV'])
        p.add_layout(Label(x=-0.2, y=-4, text=fdr_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-0.2, y=-4.5, text='false discovery rate', text_align='center', text_font_size="10pt"))
        p.add_layout(Label(x=1.2, y=-4, text=ppv_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=-4.5, text='precision', text_align='center', text_font_size="12pt"))

        # NPV / FOR
        p.hbar(left=0., right=self.measure['FOR'], y=-6., height=1., color='red')
        p.hbar(left=self.measure['FOR'], right=1., y=-6., height=1., color='green')
        for_text = 'FOR:{0:.2f}'.format(self.measure['FOR'])
        npv_text = 'NPV:{0:.2f}'.format(self.measure['NPV'])
        p.add_layout(Label(x=-0.2, y=-6, text=for_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=-0.2, y=-6.5, text='false omission rate', text_align='center', text_font_size="10pt"))
        p.add_layout(Label(x=1.2, y=-6, text=npv_text, text_align='center', text_font_size="12pt"))
        p.add_layout(Label(x=1.2, y=-6.5, text='negative predictive value', text_align='center', text_font_size="10pt"))

        p.x_range = Range1d(-0.5, 1.5, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

    def setup_stats_as_text(self):
        p = figure(plot_width=400, plot_height=800, title='All stats | 90% confidence intervals')

        y_pos = [2.85-(x*0.18) for x in range(20)]

        # tpr
        tpr_text = 'TPR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._tpr.p_expected_biased,
                                                                              self.base_stats._tpr.left_90,
                                                                              self.base_stats._tpr.p_expected_unbiased,
                                                                              self.base_stats._tpr.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[0], text=tpr_text, text_align='left', text_font_size="12pt"))
        tpr_text_aka = 'sensitivity, recall, hit_rate, true positive rate, pod'
        p.add_layout(Label(x=0.2, y=y_pos[0]-0.05, text=tpr_text_aka, text_align='left', text_font_size="10pt"))

        # fnr
        fnr_text = 'FNR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._fnr.p_expected_biased,
                                                                              self.base_stats._fnr.left_90,
                                                                              self.base_stats._fnr.p_expected_unbiased,
                                                                              self.base_stats._fnr.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[1], text=fnr_text, text_align='left', text_font_size="12pt"))
        fnr_text_aka = 'miss rate, false negative rate'
        p.add_layout(Label(x=0.2, y=y_pos[1]-0.05, text=fnr_text_aka, text_align='left', text_font_size="10pt"))

        # tnr
        tnr_text = 'TNR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._tnr.p_expected_biased,
                                                                              self.base_stats._tnr.left_90,
                                                                              self.base_stats._tnr.p_expected_unbiased,
                                                                              self.base_stats._tnr.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[2], text=tnr_text, text_align='left', text_font_size="12pt"))
        tnr_text_aka = 'specificity, selectivity, true negative rate'
        p.add_layout(Label(x=0.2, y=y_pos[2]-0.05, text=tnr_text_aka, text_align='left', text_font_size="10pt"))

        # fpr
        fpr_text = 'FPR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._fpr.p_expected_biased,
                                                                              self.base_stats._fpr.left_90,
                                                                              self.base_stats._fpr.p_expected_unbiased,
                                                                              self.base_stats._fpr.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[3], text=fpr_text, text_align='left', text_font_size="12pt"))
        fpr_text_aka = 'fall-out, false positive rate'
        p.add_layout(Label(x=0.2, y=y_pos[3]-0.05, text=fpr_text_aka, text_align='left', text_font_size="10pt"))

        # ppv
        ppv_text = 'PPV: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._ppv.p_expected_biased,
                                                                              self.base_stats._ppv.left_90,
                                                                              self.base_stats._ppv.p_expected_unbiased,
                                                                              self.base_stats._ppv.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[4], text=ppv_text, text_align='left', text_font_size="12pt"))
        ppv_text_aka = 'precision, positive predictive value'
        p.add_layout(Label(x=0.2, y=y_pos[4]-0.05, text=ppv_text_aka, text_align='left', text_font_size="10pt"))

        # fdr
        fdr_text = 'FDR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._fdr.p_expected_biased,
                                                                              self.base_stats._fdr.left_90,
                                                                              self.base_stats._fdr.p_expected_unbiased,
                                                                              self.base_stats._fdr.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[5], text=fdr_text, text_align='left', text_font_size="12pt"))
        fdr_text_aka = 'false discovery rate'
        p.add_layout(Label(x=0.2, y=y_pos[5]-0.05, text=fdr_text_aka, text_align='left', text_font_size="10pt"))

        # npv
        npv_text = 'NPV: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._npv.p_expected_biased,
                                                                              self.base_stats._npv.left_90,
                                                                              self.base_stats._npv.p_expected_unbiased,
                                                                              self.base_stats._npv.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[6], text=npv_text, text_align='left', text_font_size="12pt"))
        npv_text_aka = 'negative predictive value'
        p.add_layout(Label(x=0.2, y=y_pos[6]-0.05, text=npv_text_aka, text_align='left', text_font_size="10pt"))

        # for
        for_text = 'FOR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._for_.p_expected_biased,
                                                                              self.base_stats._for_.left_90,
                                                                              self.base_stats._for_.p_expected_unbiased,
                                                                              self.base_stats._for_.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[7], text=for_text, text_align='left', text_font_size="12pt"))
        for_text_aka = 'false omission rate'
        p.add_layout(Label(x=0.2, y=y_pos[7]-0.05, text=for_text_aka, text_align='left', text_font_size="10pt"))

        # acc
        acc_text = 'ACC: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._acc.p_expected_biased,
                                                                              self.base_stats._acc.left_90,
                                                                              self.base_stats._acc.p_expected_unbiased,
                                                                              self.base_stats._acc.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[8], text=acc_text, text_align='left', text_font_size="12pt"))
        acc_text_aka = 'accuracy'
        p.add_layout(Label(x=0.2, y=y_pos[8]-0.05, text=acc_text_aka, text_align='left', text_font_size="10pt"))

        # pre
        pre_text = 'PRE: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(1.*self.base_stats.ap/self.base_stats.all,
                                                                              0,
                                                                              0,
                                                                              0,)
        p.add_layout(Label(x=0.2, y=y_pos[9], text=pre_text, text_align='left', text_font_size="12pt"))
        pre_text_aka = 'prevalence'
        p.add_layout(Label(x=0.2, y=y_pos[9]-0.05, text=pre_text_aka, text_align='left', text_font_size="10pt"))

        # f1
        f1_text = 'F1: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._f1.p_expected_biased,
                                                                            self.base_stats._f1.left_90,
                                                                            self.base_stats._f1.p_expected_unbiased,
                                                                            self.base_stats._f1.right_90,)
        p.add_layout(Label(x=0.2, y=y_pos[10], text=f1_text, text_align='left', text_font_size="12pt"))
        f1_text_aka = 'F1 score'
        p.add_layout(Label(x=0.2, y=y_pos[10]-0.05, text=f1_text_aka, text_align='left', text_font_size="10pt"))

        # bm
        bm_text = 'BM: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._bm,
                                                                            0,
                                                                            0,
                                                                            0,)
        p.add_layout(Label(x=0.2, y=y_pos[11], text=bm_text, text_align='left', text_font_size="12pt"))
        bm_text_aka = 'bookmaker informedness'
        p.add_layout(Label(x=0.2, y=y_pos[11]-0.05, text=bm_text_aka, text_align='left', text_font_size="10pt"))

        # mk
        mk_text = 'MK: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(self.base_stats._mk,
                                                                            0,
                                                                            0,
                                                                            0,)
        p.add_layout(Label(x=0.2, y=y_pos[12], text=mk_text, text_align='left', text_font_size="12pt"))
        mk_text_aka = 'markedness'
        p.add_layout(Label(x=0.2, y=y_pos[12]-0.05, text=mk_text_aka, text_align='left', text_font_size="10pt"))

        # lrp
        lrp_val = self.base_stats.tpr / self.base_stats.fpr
        lrp_text = 'LR+: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(lrp_val,
                                                                              0,
                                                                              0,
                                                                              0,)
        p.add_layout(Label(x=0.2, y=y_pos[13], text=lrp_text, text_align='left', text_font_size="12pt"))
        lrp_text_aka = 'positive likelihood ratio'
        p.add_layout(Label(x=0.2, y=y_pos[13]-0.05, text=lrp_text_aka, text_align='left', text_font_size="10pt"))

        # lrn
        lrn_val = self.base_stats.fnr / self.base_stats.tnr
        lrn_text = 'LR-: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(lrn_val,
                                                                              0,
                                                                              0,
                                                                              0,)
        p.add_layout(Label(x=0.2, y=y_pos[14], text=lrn_text, text_align='left', text_font_size="12pt"))
        lrn_text_aka = 'negative likelihood ratio'
        p.add_layout(Label(x=0.2, y=y_pos[14]-0.05, text=lrn_text_aka, text_align='left', text_font_size="10pt"))

        # dor
        dor_val = lrp_val/lrn_val
        dor_text = 'DOR: {0:.2f} |unbiased: <{1:.2f}|{2:.2f}|{3:.2f}>'.format(dor_val,
                                                                              0,
                                                                              0,
                                                                              0,)
        p.add_layout(Label(x=0.2, y=y_pos[15], text=dor_text, text_align='left', text_font_size="12pt"))
        dor_text_aka = 'diagnostic odds ratio'
        p.add_layout(Label(x=0.2, y=y_pos[15]-0.05, text=dor_text_aka, text_align='left', text_font_size="10pt"))

        p.x_range = Range1d(0.0, 3.0, bounds=(0, None))
        p.y_range = Range1d(0.0, 3.0, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

    def setup_gird_plot(self):

        p1 = self.setup_confusion_plot()
        p2 = self.setup_actual_plot()
        p3 = self.setup_predicted_plot()
        p4 = self.setup_bar_plots()
        p5 = self.setup_stats_as_text()

        ll = layout([[layout([[p1, p4], [p2, p3]]), p5]])
        show(ll)


if __name__ == "__main__":
    confusion_ = np.array([[120, 20], [35, 80]])
    c2s = ConfusionToStats(confusion_)
    sp = StatsPlotter(c2s)
    sp.setup_gird_plot()

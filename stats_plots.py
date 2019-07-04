import numpy as np

from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid
from bokeh.models.glyphs import Wedge
from bokeh.io import curdoc, show

from bokeh.layouts import gridplot

from bokeh.models import Range1d
from bokeh.models import LabelSet, ColumnDataSource, Label
from bokeh.models import HoverTool,Text
from bokeh.plotting import figure, show


class StatsPlotter(object):
    def __init__(self, stats, order):
        self.stats = stats
        self.order = order
        self.colors = {'TP': '#7CFC00',
                       'FN': '#DC143C',
                       'TN': '#32CD32',
                       'FP': '#FF0000',
                       'aP': '#D3D3D3',
                       'aN': '#A9A9A9',
                       'pP': '#FFD700',
                       'pN': '#FFA500',
                       }
        self.orders = {'actual': ['FN', 'FP', 'TN', 'TP'],
                       'predicted': ['FP', 'FN', 'TN', 'TP'],
                       }

        self.stats = {'TP': 120,
                      'TN': 100,
                      'FP': 50,
                      'FN': 30,
                      }

        self.stats['pP'] = self.stats['TP'] + self.stats['FP']
        self.stats['pN'] = self.stats['TN'] + self.stats['FN']
        self.stats['aP'] = self.stats['TP'] + self.stats['FN']
        self.stats['aN'] = self.stats['TN'] + self.stats['FP']

        self.measure = {'TPR': 1. * self.stats['TP'] / (self.stats['TP'] + self.stats['FN']),
                        'TNR': 1. * self.stats['TN'] / (self.stats['TN'] + self.stats['FP']),
                        'PPV': 1. * self.stats['TP'] / (self.stats['TP'] + self.stats['FP']),
                        'NPV': 1. * self.stats['TN'] / (self.stats['TN'] + self.stats['FN']),
                        'FNR': 1. * self.stats['FN'] / (self.stats['FN'] + self.stats['TP']),
                        'FPR': 1. * self.stats['FP'] / (self.stats['FP'] + self.stats['TN']),
                        'FDR': 1. * self.stats['FP'] / (self.stats['FP'] + self.stats['TP']),
                        'FOR': 1. * self.stats['FN'] / (self.stats['FN'] + self.stats['TN']),
                        }

        self.order = 'actual'

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

        p = figure(plot_width=600, plot_height=600, title='ACTUAL classes split')

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
                               text_align='center', text_font_size="16pt", text_color='blue'))

        conv_labels = ['FNR: {0:.2f}'.format(self.measure['FNR']),
                       'FPR: {0:.2f}'.format(self.measure['FPR']),
                       'TNR: {0:.2f}'.format(self.measure['TNR']),
                       'TPR: {0:.2f}'.format(self.measure['TPR']),
                       ]

        p.add_layout(Label(x=x_[0] * 1.4, y=y_[0] * 1.4, text=conv_labels[0],
                           text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[1] * 1.9, y=y_[1] * 1.9, text=conv_labels[1],
                           text_align='center', text_font_size="16pt"))

        p.add_layout(Label(x=x_[2] * 1.9, y=y_[2] * 1.9, text=conv_labels[2],
                           text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[3] * 1.4, y=y_[3] * 1.4, text=conv_labels[3],
                           text_align='center', text_font_size="16pt"))

        x_, y_ = deg_2_cart(mid_angles_cum)
        p.add_layout(Label(x=x_[0] * 2.3, y=y_[0] * 2.3, text='aN', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[1] * 1.7, y=y_[1] * 1.7, text='aP', text_align='center', text_font_size="16pt"))

        p.y_range = Range1d(-2.5, 2.5, bounds=(0, None))
        p.x_range = Range1d(-2.5, 2.5, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

        # show(p)

    def setup_predicted_plot(self):
        starts, ends, order = self.stats_2_deg('predict')
        mid_angles = (starts + 1. * (ends - starts) % (2 * np.pi) / 2) % (2 * np.pi)

        mid_angles_cum = [(mid_angles[1] + mid_angles[2]) / 2 % (2 * np.pi),
                          (mid_angles[0] + mid_angles[3]) / 2 % (2 * np.pi)]

        p = figure(plot_width=600, plot_height=600, title='PREDICTED classes split')

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
                               text_align='center', text_font_size="16pt", text_color='blue'))

        conv_labels = ['FDR: {0:.2f}'.format(self.measure['FDR']),
                       'FOR: {0:.2f}'.format(self.measure['FOR']),
                       'NPV: {0:.2f}'.format(self.measure['NPV']),
                       'PPV: {0:.2f}'.format(self.measure['PPV']),
                       ]

        p.add_layout(Label(x=x_[0] * 1.4, y=y_[0] * 1.4, text=conv_labels[0],
                           text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[1] * 1.9, y=y_[1] * 1.9, text=conv_labels[1],
                           text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[2] * 1.9, y=y_[2] * 1.9, text=conv_labels[2],
                           text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[3] * 1.4, y=y_[3] * 1.4, text=conv_labels[3],
                           text_align='center', text_font_size="16pt"))

        x_, y_ = deg_2_cart(mid_angles_cum)
        p.add_layout(Label(x=x_[0] * 2.3, y=y_[0] * 2.3, text='pN', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=x_[1] * 1.7, y=y_[1] * 1.7, text='pP', text_align='center', text_font_size="16pt"))

        p.y_range = Range1d(-2.5, 2.5, bounds=(0, None))
        p.x_range = Range1d(-2.5, 2.5, bounds=(0, None))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        return p

        # show(p)

    def setup_confusion_plot(self):

        p = figure(plot_width=600, plot_height=600, title='Confusion')
        p.rect(x=1, y=1, width=1.9, height=1.9, fill_color=self.colors['FP'])
        p.rect(x=-1, y=1, width=1.9, height=1.9, fill_color=self.colors['TP'])
        p.rect(x=1, y=-1, width=1.9, height=1.9, fill_color=self.colors['TN'])
        p.rect(x=-1, y=-1, width=1.9, height=1.9, fill_color=self.colors['FN'])

        p.rect(x=-1, y=1, width=1.4, height=5.4, fill_color=self.colors['aP'], alpha=0.2)
        p.rect(x=1, y=1, width=1.4, height=5.4, fill_color=self.colors['aN'], alpha=0.2)
        p.rect(x=-1, y=1, width=5.4, height=1.4, fill_color=self.colors['pP'], alpha=0.2)
        p.rect(x=-1, y=-1, width=5.4, height=1.4, fill_color=self.colors['pN'], alpha=0.2)

        p.add_layout(Label(x=-1, y=1, text='TP: {}'.format(self.stats['TP']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1, y=-1, text='TN: {}'.format(self.stats['TN']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1, y=1, text='FP: {}'.format(self.stats['FP']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-1, y=-1, text='FN: {}'.format(self.stats['FN']), text_align='center', text_font_size="16pt"))

        p.add_layout(Label(x=-3, y=1, text='pP: {}'.format(self.stats['pP']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-3, y=-1, text='pN: {}'.format(self.stats['pN']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-1, y=3, text='aP: {}'.format(self.stats['aP']), text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1, y=3, text='aN: {}'.format(self.stats['aN']), text_align='center', text_font_size="16pt"))

        return p

    def setup_bar_plots(self):
        p = figure(plot_width=600, plot_height=600, title='Bar plots')

        # TPR / FNR
        p.hbar(left=0., right=self.measure['FNR'], y=0., height=1., color='red')
        p.hbar(left=self.measure['FNR'], right=1., y=0., height=1., color='green')
        fnr_text = 'FNR:{0:.2f}'.format(self.measure['FNR'])
        tpr_text = 'TPR:{0:.2f}'.format(self.measure['TPR'])
        p.add_layout(Label(x=-0.2, y=0, text=fnr_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-0.2, y=-0.5, text='miss rate', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=0, text=tpr_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-0.5, text='recall', text_align='center', text_font_size="16pt"))


        # TNR / FPR
        p.hbar(left=0., right=self.measure['FPR'], y=-2., height=1., color='red')
        p.hbar(left=self.measure['FPR'], right=1., y=-2., height=1., color='green')
        fpr_text = 'FPR:{0:.2f}'.format(self.measure['FPR'])
        tnr_text = 'TNR:{0:.2f}'.format(self.measure['TNR'])
        p.add_layout(Label(x=-0.2, y=-2, text=fpr_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-0.2, y=-2.5, text='fall-out', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-2, text=tnr_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-2.5, text='specificity', text_align='center', text_font_size="16pt"))

        # PPV / FDR
        p.hbar(left=0., right=self.measure['FDR'], y=-4., height=1., color='red')
        p.hbar(left=self.measure['FDR'], right=1., y=-4., height=1., color='green')
        fdr_text = 'FDR:{0:.2f}'.format(self.measure['FDR'])
        ppv_text = 'PPV:{0:.2f}'.format(self.measure['PPV'])
        p.add_layout(Label(x=-0.2, y=-4, text=fdr_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-0.2, y=-4.5, text='false discovery rate', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-4, text=ppv_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-4.5, text='precision', text_align='center', text_font_size="16pt"))

        # NPV / FOR
        p.hbar(left=0., right=self.measure['FOR'], y=-6., height=1., color='red')
        p.hbar(left=self.measure['FOR'], right=1., y=-6., height=1., color='green')
        for_text = 'FOR:{0:.2f}'.format(self.measure['FOR'])
        npv_text = 'NPV:{0:.2f}'.format(self.measure['NPV'])
        p.add_layout(Label(x=-0.2, y=-6, text=for_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=-0.2, y=-6.5, text='false omission rate', text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-6, text=npv_text, text_align='center', text_font_size="16pt"))
        p.add_layout(Label(x=1.2, y=-6.5, text='negative predictive value', text_align='center', text_font_size="16pt"))

        # p.y_range = Range1d(-2.5, 2.5, bounds=(0, None))
        p.x_range = Range1d(-0.5, 1.5, bounds=(0, None))

        return p

    def setup_gird_plot(self):

        p1 = self.setup_confusion_plot()
        p2 = self.setup_actual_plot()
        # show(p2)
        p3 = self.setup_predicted_plot()
        p4 = self.setup_bar_plots()
        # grid = gridplot([p2, p3], ncols=2)
        # grid = gridplot([p1, p2, p3, p4], ncols=4)
        grid = gridplot([[p1, p4], [p3, p2]]) #, ncols=2, nrows=2)
        show(grid)

        # show(p2)
        # grid = gridplot([[p1, p2], [None, p1]])
        # show(grid)


def stat_2_deg(stats, order):
    # oder: actual, predicted
    orders = {'actual': ['fn', 'fp', 'tn', 'tp'],
              'predicted': ['fp', 'fn', 'tn', 'tp'],
              }
    sum_ = sum(stats.values())
    percentages = [0.] + [1.*stats[o]/sum_ for o in orders[order]]
    percentiles = np.cumsum(percentages)
    start, end = percentiles[:-1], percentiles[1:]
    start = start*2*np.pi + np.pi/2
    end = end*2*np.pi + np.pi/2
    return start % (2*np.pi), end % (2*np.pi), orders[order]


def deg_2_cart(deg_):
    return np.cos(deg_), np.sin(deg_)


if __name__ =="__main__":
    sp = StatsPlotter(None, None)
    # sp.setup_actual_plot()
    # sp.setup_predicted_plot()
    sp.setup_gird_plot()
    # sp.setup_confusion_plot()

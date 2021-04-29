import numpy as np
from fpga_lib.gui.qt import QtWidgets, QtCore
from fpga_lib.gui.widgets import HistogramViewer, VBoxTab
from fpga_lib.analysis.fit import do_fit, eval_fit
from fpga_lib.parameters import IntParameter
from pyqtgraph import PlotWidget, LabelItem, ImageView, PlotItem, EllipseROI, InfiniteLine
from .fpga import FPGAExperiment, FPGAExperimentTab
from scipy.optimize import fmin
from lmfit import Parameter

__all__ = ['ReadoutOptV2Experiment']


class ReadoutOptV2Experiment(FPGAExperiment):
    readout_instance = None
    #n_regions = 4

    def get_widget(self, window, item):
        return ReadoutOptV2Tab(self, window, item)

    def sequence(self):
        for i in range(self.n_regions):
            self.prepare(i)
            self.readout_instance(traj='rel', se='se')

    def prepare(self, i):
        return

    def get_trajectories(self):
        return self.results['traj'].data

    def get_se_data(self):
        return self.results['se'].data

    def propose_envelope(self, region_trajs):
        tg =  region_trajs[0]
        te = region_trajs[1]
        env = (te - tg).conj()
        return env, 1j*env


class ReadoutOptV2Tab(FPGAExperimentTab):
    tab_actions = FPGAExperimentTab.tab_actions + [
        ('update_envelope', None),
        ('update_multithresh', None),
    ]
    def __init__(self, exp, window, name):
        super(ReadoutOptV2Tab, self).__init__(exp, window, name)
        self.traj_widget = ReadoutOptV2Widget(exp, parent=self)
        VBoxTab(self.view_box, 'Trajectories', self.traj_widget, pos=0)

    def run(self):
        self.traj_widget.current_env_view.envelopes = self.experiment.readout_instance.get_envelopes()
        self.traj_widget.proposed_env_view.envelopes = None
        super(ReadoutOptV2Tab, self).run()

    def plot(self):
        self.traj_widget.set_trajectories(self.experiment.get_trajectories(), self.experiment.get_se_data())
        super(ReadoutOptV2Tab, self).plot()

    def update_envelope(self):
        self.traj_widget.update_envelope()

    def update_multithresh(self):
        params = []
        for roi in self.traj_widget.current_env_view.rois:
            state = roi.getState()
            (x0, y0), (dx, dy), angle = state['pos'], state['size'], state['angle']
            params.append((x0, y0, dx, dy, angle))
        self.experiment.readout_instance.multithresh = params


class RegionTrajPlot(PlotWidget):
    def __init__(self, *args, **kwargs):
        super(RegionTrajPlot, self).__init__(*args, **kwargs)
        self.count_label = LabelItem()
        self.getPlotItem().layout.addItem(self.count_label, 4, 1)

    def set_count(self, val):
        self.count_label.setText(text='# pts = %d' % val)


class ReadoutOptV2Widget(QtWidgets.QSplitter):
    def __init__(self, exp, parent=None):
        self.exp = exp
        super(ReadoutOptV2Widget, self).__init__(parent=parent)

        self.current_env_view = HistogramViewerV2(parent=self)
        self.current_env_view.add_rois(exp.n_regions)
        self.current_env_view.region_changed.connect(self.region_changed)
        self.proposed_env_view = HistogramViewerV2()
        self.proposed_env_view.add_lines()
        self.region_traj_plots = [RegionTrajPlot() for _ in range(exp.n_regions)]
        self.versus_t_checkbox = QtWidgets.QCheckBox('versus time')
        self.versus_t_checkbox.clicked.connect(self.region_changed)

        left_box = QtWidgets.QWidget(parent=self)
        left_layout = QtWidgets.QVBoxLayout(left_box)
        state_selector_layout = QtWidgets.QHBoxLayout()
        self.state_checks = []
        for i in range(exp.n_regions):
            w = QtWidgets.QCheckBox(str(i))
            w.setChecked(True)
            w.toggled.connect(self.update_view)
            self.state_checks.append(w)
            state_selector_layout.addWidget(w)
        fit_button = QtWidgets.QPushButton('Fit')
        fit_button.clicked.connect(self.fit_rois)
        state_selector_layout.addWidget(fit_button)
        left_layout.addLayout(state_selector_layout)
        left_layout.addWidget(self.current_env_view)


        traj_plots_box = QtWidgets.QWidget(parent=self)
        traj_plots_layout = QtWidgets.QVBoxLayout(traj_plots_box)
        traj_plots_layout.addWidget(self.versus_t_checkbox)
        for plot in self.region_traj_plots:
            traj_plots_layout.addWidget(plot)

        self.addWidget(left_box)
        self.addWidget(traj_plots_box)
        self.addWidget(self.proposed_env_view)
        self.setSizes([self.width()//3]*3)

    def set_trajectories(self, trajs, se_data):
        self.trajs = trajs
        self.se_data = se_data
        self.update_view()
        self.fit_rois()

    def update_view(self):
        idxs = [i for i, w in enumerate(self.state_checks) if w.isChecked()]
        trajs = self.trajs[:,idxs].reshape((-1, self.trajs.shape[-1]))
        se_data = self.se_data[:,idxs].flatten()
        scale = self.current_env_view.set_trajectories(trajs, se_vals=se_data)
        self.proposed_env_view.set_trajectories(trajs, scale=scale)
        self.region_changed()

    def region_changed(self):
        region_trajs, counts = self.current_env_view.get_region_trajectories()
        if region_trajs is None:
            return

        # update region traj plots
        for traj, count, plot in zip(region_trajs, counts, self.region_traj_plots):
            plot.clear()
            plot.set_count(count)
            if self.versus_t_checkbox.isChecked():
                plot.plot(traj.real, pen=(255,0,0))
                plot.plot(traj.imag, pen=(0,255,0))
                plot.plot(abs(traj), pen=(0,0,255))
            else:
                plot.plot(traj.real, traj.imag, pen=(255,0,0))

        # update proposed envelope
        proposed_envelopes = self.exp.propose_envelope(region_trajs)
        self.proposed_env_view.set_envelopes(proposed_envelopes)

    def update_envelope(self):
        e1, e2 = self.proposed_env_view.envelopes
        t1, t2 = self.proposed_env_view.get_thresholds()
        trajs = self.current_env_view.get_region_trajectories()[0]
        self.exp.readout_instance.set_envelope(e1, e2, t1, t2, trajs)

    def fit_rois(self):
        # import matplotlib.pyplot as plt
        # f = plt.figure()
        # ax = f.add_subplot(111)
        # i = 0
        for roi, trajs in zip(self.current_env_view.rois, self.trajs.transpose((1,0,2))):
            H, xs, ys, _, _ = self.current_env_view.calc_hist(trajs, scale=self.current_env_view.scale)
            XS, YS = np.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2)
            params = do_fit('gaussian2d', XS.T, YS.T, H)
            # ax.imshow(H, extent=(xs[0], xs[-1], ys[0], ys[-1]))
            # f.savefig('sub_plot_%d.pdf'%i)
            # i += 1
            x0, y0, dx, dy = [params[s].value for s in 'x0,y0,sigmax,sigmay'.split(',')]
            print x0, y0, dx, dy, H.shape, XS.shape, YS.shape
            roi.setPos(x0-3*dx, y0-3*dy)
            roi.setSize(6*dx, 6*dy)


class HistogramViewerV2(ImageView):
    region_changed = QtCore.Signal()
    def __init__(self, envelopes=None, parent=None):
        super(HistogramViewerV2, self).__init__(view=PlotItem(), parent=parent)
        self.envelopes = envelopes
        self.ui.histogram.gradient.loadPreset('flame')
        self.rois = None
        self.set_histogram(False)
        histogram_action = QtWidgets.QAction('Histogram', self)
        histogram_action.setCheckable(True)
        histogram_action.triggered.connect(self.set_histogram)
        self.scene.contextMenu.append(histogram_action)
        self.log_check = QtWidgets.QCheckBox('log')
        self.log_check.clicked.connect(self.replot)
        # fit_button = QtWidgets.QPushButton('Fit')
        # fit_button.clicked.connect(self.fit_centroid)
        actions_layout = QtWidgets.QHBoxLayout()
        actions_layout.addWidget(self.log_check)
        # actions_layout.addWidget(fit_button)
        grid = self.layout()
        grid.addLayout(actions_layout, grid.rowCount(), 0)

    def add_rois(self, n_roi=2):
        self.rois = [EllipseROI([0, 0], [30, 30]) for _ in range(n_roi)]
        for roi in self.rois:
            roi.sigRegionChangeFinished.connect(self.region_changed)
            self.view.addItem(roi)

    def add_lines(self):
        self.h_line = InfiniteLine(pos=0, angle=0, movable=True)
        self.v_line = InfiniteLine(pos=0, angle=90, movable=True)
        self.view.addItem(self.h_line, ignoreBounds=False)
        self.view.addItem(self.v_line, ignoreBounds=False)

    def get_region_trajectories(self):
        trajs = []
        lengths = []
        for roi in self.rois:
            state = roi.getState()
            (x0, y0), (dx, dy), angle = state['pos'], state['size'], state['angle']
            rot_vals = (self.wvals - (x0 + 1j*y0)) * np.exp(-1j*np.pi*angle/180)
            x, y = rot_vals.real, rot_vals.imag
            idxs = ((x - dx/2)**2 / dx**2) + ((y - dy/2)**2 / dy**2) < 0.25
            if self.traj[idxs].size == 0:
                return None, None
            lengths.append(len(self.traj[idxs]))
            trajs.append(self.traj[idxs].mean(axis=0))
        return trajs, lengths

    def get_thresholds(self):
        return self.v_line.getXPos(), self.h_line.getYPos()

    def replot(self):
        self.set_trajectories(self.traj, scale=self.scale)

    def calc_hist(self, trajectories, se_vals=None, scale=None):
        if self.envelopes is None:
            ones = np.ones(trajectories.shape[-1])
            self.envelopes = ones, 1j*ones
        x_vals = np.sum(trajectories * self.envelopes[0], axis=1).real
        y_vals = np.sum(trajectories * self.envelopes[1], axis=1).real
        if se_vals is not None:
            mean_scale = np.median(x_vals / se_vals.real)
            # assert np.allclose(x_vals / se_vals.real, mean_scale, rtol=.5), (x_vals / se_vals.real)
            # assert np.allclose(y_vals / se_vals.imag, mean_scale, rtol=.5), (y_vals / se_vals.imag)
            i_mean_scale = int(np.round(mean_scale))
            # assert np.isclose(mean_scale, i_mean_scale, rtol=.01), mean_scale
            # assert np.isclose(np.log2(i_mean_scale), int(np.log2(i_mean_scale))), mean_scale
            scale = i_mean_scale
        else:
            assert scale is not None
        x_vals /= scale
        y_vals /= scale
        wvals = x_vals + 1j*y_vals
        H, xs, ys = np.histogram2d(wvals.real, wvals.imag, bins=25, normed=True)
        return H, xs, ys, scale, wvals

    def set_trajectories(self, trajectories, se_vals=None, scale=None):
        self.traj = trajectories
        H, xs, ys, self.scale, self.wvals = self.calc_hist(trajectories, se_vals=se_vals, scale=scale)
        if self.log_check.isChecked():
            H[H == 0] = H[H != 0].min()
            H = np.log10(H)
        self.setImage(H, pos=(xs[0], ys[0]), scale=(xs[1]-xs[0], ys[1]-ys[0]))
        # if self.rois is not None and tuple(self.rois[0].getState()['pos']) == (0, 0):
        #     self.fit_centroid()
        return self.scale

    def set_envelopes(self, envelopes):
        self.envelopes = envelopes
        self.set_trajectories(self.traj, scale=self.scale)

    # def fit_centroid(self):
    #     H, xs, ys = np.histogram2d(self.wvals.real, self.wvals.imag, bins=25, normed=True)
    #     XS, YS = np.meshgrid((xs[:-1]+xs[1:])/2, (ys[:-1]+ys[1:])/2)
    #
    #     N = len(self.rois)
    #
    #     def cost(params):
    #         x0s = params[:N]
    #         y0s = params[N:2*N]
    #         amps = params[2*N:3*N]
    #         sigma = params[3*N]
    #         ofs = params[3*N+1]
    #         r = ofs
    #         for x0, y0, amp in zip(x0s, y0s, amps):
    #             r += amp * np.exp(-((XS-x0)**2 + (YS-y0)**2) / (2*sigma**2))
    #         c = ((H - r.T)**2).sum()
    #         return c
    #
    #     xs0 = np.linspace(xs.min(), xs.max(), N+2)[1:-1]
    #     ys0 = np.linspace(ys.min(), ys.max(), N+2)[1:-1]
    #     amps0 = H.max() / 2 * np.ones(N)
    #     sigma0 = (xs.max() - xs.min()) / 6
    #     init_params = np.concatenate([
    #         xs0, ys0, amps0, [sigma0, 0]
    #     ])
    #
    #     pf = fmin(cost, init_params)
    #     sig = pf[-2]
    #     for i, roi in enumerate(self.rois):
    #         x, y = pf[i], pf[N+i]
    #         r = 3 * sig
    #         roi.setPos(x-r, y-r)
    #         roi.setSize(2*r, 2*r)


    def set_histogram(self, visible):
        self.ui.histogram.setVisible(visible)
        self.ui.roiBtn.setVisible(visible)
        self.ui.normGroup.setVisible(visible)
        self.ui.menuBtn.setVisible(visible)

from Bounder import Bounder
from Plotter import Plotter
import numpy as np
import ipywidgets as wid
from IPython.display import display, clear_output

GOOD_OBS_PROBS = "Good choices! :) The Observational Probabilities are " \
                 "consistent."
BAD_OBS_PROBS = '<span style="color:red">Bad choices! :( The Observational ' \
                'Probabilities are inconsistent. Try alternative slider ' \
                'positions.</span>'
BAD_EXP_PROBS = '<span style="color:red">Bad choices! :( The Experimental ' \
                'Probabilities are inconsistent. Try alternative slider ' \
                'positions.</span>'


class Widgeter:
    def __init__(self):
        """
        The main method of this class and the only one meant for external
        use is run_gui(). This method runs a GUI (Graphical User Interface)
        as a cell in a Jupyter notebook. The controls of the GUI are
        implemented using the library ipywidgets.

        Attributes
        ----------
        bounder_f : Bounder
            Bounder object for females
        bounder_m : Bounder
            Bounder object for males
        exogeneity : bool
        exp_sliders : List[wid.FloatSlider]
            list of 2 Experimental Probabilities sliders
        exp_tboxes : List[wid.BoundedFloatText]
            list of 2 text boxes attached to self.exp_sliders
        good_obs_data : bool
            good observtional data
        left_exp_probs_bds : np.array[shape=(2, 2)]
            left (low) bounds for E_{y|x}
        monotonicity : bool
        obs_f_sliders : List[wid.FloatSlider]
            list of 8 Observational Probabilities sliders for female patients
        obs_f_tboxes : List[wid.BoundedFloatText]
            list of 8 text boxes attached to self.obs_f_sliders
        obs_m_sliders : List[wid.FloatSlider]
            list of 8 Observational Probabilities sliders for male patients
        obs_m_tboxes : List[wid.BoundedFloatText]
            list of 8 text boxes attached to self.obs_m_sliders
        only_obs : bool
            Only Observational Probabilities, no Experimental ones
        pns3_bds_f : np.array[shape=(3, 2)]
            bounder_f.get_pns3_bds()
        pns3_bds_m : np.array[shape=(3, 2)]
            bounder_m.get_pns3_bds()
        right_exp_probs_bds : np.array[shape=(2, 2)]
            right (high) bounds for E_{y|x}
        strong_exogeneity : bool

        """
        self.only_obs = True
        self.good_obs_data = True
        self.monotonicity = False
        self.exogeneity = False
        self.strong_exogeneity = False

        o_y_bar_x_m = np.array([[.5, .5], [.5, .5]])
        px_m = np.array([.5, .5])
        self.bounder_m = Bounder(o_y_bar_x_m, px_m)
        self.pns3_bds_m = self.bounder_m.get_pns3_bds()

        o_y_bar_x_f = np.array([[.5, .5], [.5, .5]])
        px_f = np.array([.5, .5])
        self.bounder_f = Bounder(o_y_bar_x_f, px_f)
        self.pns3_bds_f = self.bounder_f.get_pns3_bds()

        self.left_exp_probs_bds = np.zeros((2, 2))
        self.right_exp_probs_bds = np.ones((2, 2))
        
        self.obs_m_sliders = None
        self.obs_f_sliders = None
        self.exp_sliders = None

        self.obs_m_tboxes = None
        self.obs_f_tboxes = None
        self.exp_tboxes = None

    def refresh_plot_using_slider_vals(
            self,
            o1b0_m, o1b1_m, px1_m,
            o1b0_f, o1b1_f, px1_f,
            e1b0, e1b1):
        """
        This method is called by wid.interactive() which requires it. Its
        inputs are all 8 slider values. After doing some housework,
        this method calls Plotter.plot_pns3_bds().

        Parameters
        ----------
        o1b0_m : float
            O_{1|0,m}
        o1b1_m : float
            O_{1|1,m}
        px1_m : float
            P(x=1) for males
        o1b0_f : float
            O_{1|0,f}
        o1b1_f : float
            O_{1|1,f}
        px1_f : float
            P(x=1) for females
        e1b0 : float
            E_{1|0} for both males and females
        e1b1 : float
            E_{1|1} for both males and females

        Returns
        -------
        None

        """

        o_y_bar_x_m = np.array([
            [1 - o1b0_m, 1 - o1b1_m],
            [o1b0_m, o1b1_m]])
        px_m = np.array([1 - px1_m, px1_m])
        self.bounder_m.set_obs_probs(o_y_bar_x_m, px_m)

        o_y_bar_x_f = np.array([
            [1 - o1b0_f, 1 - o1b1_f],
            [o1b0_f, o1b1_f]])
        px_f = np.array([1 - px1_f, px1_f])
        self.bounder_f.set_obs_probs(o_y_bar_x_f, px_f)

        if not self.only_obs:
            e_y_bar_x = np.array([
                [1 - e1b0, 1 - e1b1],
                [e1b0, e1b1]])
            self.bounder_m.set_exp_probs(e_y_bar_x)
            self.bounder_f.set_exp_probs(e_y_bar_x)

        bds_m = self.bounder_m.get_pns3_bds()
        bds_f = self.bounder_f.get_pns3_bds()
        Plotter.plot_pns3_bds(bds_m=bds_m, bds_f=bds_f)

    def refresh_slider_colors(self, obs_green):
        """
        This method toggles the colors and disabled status (green/red for
        enabled/disabled) of the 6 sliders for Observational Probabilities
        and the 2 sliders for Experimental Probabiities. In addition,
        it toggles the disabled status of the text boxes that are attached
        to each slider.

        Parameters
        ----------
        obs_green : bool
            True iff Observational sliders are green (enabled)

        Returns
        -------
        None

        """
        obs_m_slider_to_latex = {
            self.obs_m_sliders[0]: 'O_{1|0,m}',
            self.obs_m_sliders[1]: 'O_{1|1,m}',
            self.obs_m_sliders[2]: '\pi_{1,m}'
        }
        obs_f_slider_to_latex = {
            self.obs_f_sliders[0]: 'O_{1|0,f}',
            self.obs_f_sliders[1]: 'O_{1|1,f}',
            self.obs_f_sliders[2]: '\pi_{1,f}'
        }
        exp_slider_to_latex = {
            self.exp_sliders[0]: 'E_{1|0}',
            self.exp_sliders[1]: 'E_{1|1}'
        }

        def color_it(g, latex_str):
            if g:
                color = 'green'
            else:
                color = 'red'
            return '$\color{' + color + '}{' + latex_str + '}$'

        for x, tbox in zip(self.obs_m_sliders, self.obs_m_tboxes):
            x.disabled = not obs_green
            tbox.disabled = not obs_green
            x.description = color_it(obs_green,
                                     obs_m_slider_to_latex[x])
        for x, tbox in zip(self.obs_f_sliders, self.obs_f_tboxes):
            x.disabled = not obs_green
            tbox.disabled = not obs_green
            x.description = color_it(obs_green,
                                     obs_f_slider_to_latex[x])
        for x, tbox in zip(self.exp_sliders, self.exp_tboxes):
            x.disabled = obs_green
            tbox.disabled = obs_green
            x.description = color_it(not obs_green, exp_slider_to_latex[x])

    def refresh_plot(self):
        """
        This method is a clever way of inducing the method wid.interactive()
        to redraw the plot. The method jiggles the exp data sliders,
        thus causing wid.interactive() to redraw the plot.

        Returns
        -------
        None

        """

        for x in self.exp_sliders:
            delta = .1
            x.min -= delta
            x.value -= delta
            x.min += delta
            x.value += delta

    def set_exp_probs_bds(self):
        """
        This method asks class Bounder to merge the bounds for Experimental
        Probabilities of bounder_m and bounder_f. It outputs True iff it
        succeeds in this endeavor.

        Returns
        -------
        bool

        """
        self.left_exp_probs_bds, self.right_exp_probs_bds = \
            Bounder.get_joint_exp_probs_bds(self.bounder_m, self.bounder_f)
        if self.left_exp_probs_bds is None:
            return False
        else:
            return True

    def set_exp_sliders_to_valid_values(self):
        """
        This method is used after set_exp_probs_bds() has been called
        successfully. This method sets to valid values the min, value and
        max parameters of the Experimental Probabilities sliders.

        Returns
        -------
        None

        """

        diff = self.right_exp_probs_bds - self.left_exp_probs_bds
        assert (diff >= 0).all()
        # set value of E_{1|i}

        for i in [0, 1]:
            a, b = self.left_exp_probs_bds[1, i], \
                self.right_exp_probs_bds[1, i]
            self.exp_sliders[i].disabled = True
            self.exp_sliders[i].min = a
            self.exp_sliders[i].max = b
            self.exp_sliders[i].disabled = False
            self.exp_sliders[i].value = a

    def run_gui(self):
        """
        This is the main method of this class and the only one meant for
        external use. It draws a GUI.

        The GUI has 8 sliders, 6 sliders for Observational Probabilities and
        2 sliders for Experimental Probabilities. Each slider has a text box
        attached to it which can be used to enter input by typing numbers
        instead of moving the slider. In addition, the GUI has 2 clickable
        control buttons, 3 check boxes and one disabled text box that gives
        info about the current status of the calculations.

        Returns
        -------
        None

        """

        slider_params = dict(
            min=0,
            max=1,
            value=.5,
            step=.01,
            orientation='vertical')
        o1b0_m_slider = wid.widgets.FloatSlider(**slider_params)
        o1b1_m_slider = wid.widgets.FloatSlider(**slider_params)
        px1_m_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        self.obs_m_sliders = [o1b0_m_slider, 
                              o1b1_m_slider, 
                              px1_m_slider]

        o1b0_f_slider = wid.widgets.FloatSlider(**slider_params)
        o1b1_f_slider = wid.widgets.FloatSlider(**slider_params)
        px1_f_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        self.obs_f_sliders = [o1b0_f_slider, 
                              o1b1_f_slider, 
                              px1_f_slider]

        e1b0_slider = wid.widgets.FloatSlider(**slider_params)
        e1b1_slider = wid.widgets.FloatSlider(**slider_params)
        # order important, 1b0 before 1b1,
        # mnemonic 10 < 11
        self.exp_sliders = [e1b0_slider, e1b1_slider]

        header1 = wid.Label(value="Enter Observational Data\
            from a survey.")
        header2 = wid.Label("Then press the \
            'Add Experimental Data (RCT)' button\
            if you also have Experimental Data.")
        header3 = wid.Label(value=
            "Sliders with green/red labels are\
            enabled/disabled.")
        
        add_but = wid.Button(
            description='Add Experimental Data (RCT)',
            button_style='danger',
            layout=wid.Layout(width='200px')
        )
        status_sign = wid.HTMLMath(value=GOOD_OBS_PROBS)

        def add_but_do(btn):
            if self.only_obs and self.good_obs_data:
                self.refresh_slider_colors(obs_green=False)
                self.set_exp_sliders_to_valid_values()
                self.only_obs = False
        add_but.on_click(add_but_do)

        print_but = wid.Button(
            description='Print',
            button_style='warning',
            layout=wid.Layout(width='50px')
        )
        out = wid.Output()
        display(out)

        def print_but_do(btn):
            with out:
                print("###################################")
                print("Male:------------------------------")
                self.bounder_m.print_all_probs()
                Bounder.print_pns3_bds(self.pns3_bds_m)
                print("Female:----------------------------")
                self.bounder_f.print_all_probs()
                Bounder.print_pns3_bds(self.pns3_bds_f)
        print_but.on_click(print_but_do)

        mono_but = wid.Checkbox(
            value=self.monotonicity,
            description="Monotonicity",
            indent=False)

        def mono_but_do(change):
            new = change['new']
            self.monotonicity = new
            self.bounder_m.monotonicity = new
            self.bounder_f.monotonicity = new
            if not self.only_obs:
                self.refresh_plot()
        mono_but.observe(mono_but_do, names='value')

        exo_but = wid.Checkbox(
            value=self.exogeneity,
            description="Exogeneity",
            indent=False)

        def exo_but_do(change):
            new = change['new']
            self.exogeneity = new
            self.bounder_m.exogeneity = new
            self.bounder_f.exogeneity = new
            if not self.only_obs:
                self.refresh_plot()

        exo_but.observe(exo_but_do, names='value')

        strong_exo_but = wid.Checkbox(
            value=self.strong_exogeneity,
            description="Strong Exogeneity",
            indent=False)

        def strong_exo_but_do(change):
            new = change['new']
            self.strong_exogeneity = new
            self.bounder_m.strong_exogeneity = new
            self.bounder_f.strong_exogeneity = new
            if not self.only_obs:
                self.refresh_plot()
        strong_exo_but.observe(strong_exo_but_do, names='value')

        def box_the_sliders(sliders):
            vbox_list = []
            tbox_list = []
            for x in sliders:
                tbox = wid.BoundedFloatText(
                    step=x.step,
                    min=x.min,
                    max=x.max,
                    layout=wid.Layout(width='60px'))
                tbox_list.append(tbox)
                wid.jslink((x, 'value'), (tbox, 'value'))
                vbox_list.append(wid.VBox([x, tbox]))
            return wid.HBox(vbox_list), tbox_list
        constraints_box = wid.VBox([exo_but, strong_exo_but,
                                   mono_but])
        cmd_box = wid.HBox([print_but, add_but, status_sign])
        obs_m_box, self.obs_m_tboxes = box_the_sliders(self.obs_m_sliders)
        obs_f_box, self.obs_f_tboxes = box_the_sliders(self.obs_f_sliders)
        obs_box = wid.HBox([obs_m_box, obs_f_box],
            layout=wid.Layout(border='solid'))
        exp_box, self.exp_tboxes = box_the_sliders(self.exp_sliders)
        exp_box = wid.HBox([exp_box],
            layout=wid.Layout(border='solid'))
        all_boxes = wid.VBox([
            header1,
            header2,
            header3,
            cmd_box,
            wid.HBox([obs_box, exp_box, constraints_box])
        ])

        def fun(o1b0_m_slider, o1b1_m_slider, px1_m_slider,
                o1b0_f_slider, o1b1_f_slider, px1_f_slider,
                e1b0_slider, e1b1_slider):
            self.refresh_plot_using_slider_vals(
                o1b0_m_slider, o1b1_m_slider, px1_m_slider,
                o1b0_f_slider, o1b1_f_slider, px1_f_slider,
                e1b0_slider, e1b1_slider)
            if not self.set_exp_probs_bds():
                self.good_obs_data = False
                if self.only_obs:
                    status_sign.value = BAD_OBS_PROBS
                else:
                    status_sign.value = BAD_EXP_PROBS
            else:
                self.good_obs_data = True
                status_sign.value = GOOD_OBS_PROBS +\
                '<br>%.2f $\leq E_{1|0} \leq$ %.2f'\
                    % (self.left_exp_probs_bds[1, 0],
                    self.right_exp_probs_bds[1, 0]) +\
                '<br>%.2f $\leq E_{1|1}\leq$ %.2f' \
                    % (self.left_exp_probs_bds[1, 1],
                    self.right_exp_probs_bds[1, 1])

        slider_dict = {
            'o1b0_m_slider': o1b0_m_slider,
            'o1b1_m_slider': o1b1_m_slider,
            'px1_m_slider': px1_m_slider,
            'o1b0_f_slider': o1b0_f_slider,
            'o1b1_f_slider': o1b1_f_slider,
            'px1_f_slider': px1_f_slider,
            'e1b0_slider': e1b0_slider,
            'e1b1_slider': e1b1_slider
        }
        plot = wid.interactive_output(fun, slider_dict)
        # interactive_plot.layout.height = '800px'
        self.refresh_slider_colors(obs_green=True)
        display(all_boxes, plot)

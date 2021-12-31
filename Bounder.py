import numpy as np


class Bounder:
    def __init__(self, o_y_bar_x, px, e_y_bar_x=None,
                 monotonicity=False, exogeneity=False, strong_exo=False):
        """
        The theory and notation in this class is explained in the chapter
        entitled "Personalized Treatment Effects" of my free open source
        book Bayesuvius. That Bayesuvius chapter is totally based on a paper
        by Tian and Pearl. I also benefitted from comments by Scott Mueller.

        https://qbnets.wordpress.com/2020/11/30/my-free-book-bayesuvius-on-bayesian-networks/

        The ultimate goal of this class is to calculate the bounds for PNS,
        PN and PS, given the probability matrices O_{y|x}, P(x) and E_{y|x}.
        O_{y|x} and P(x) are called the Observational Probabilities (these
        from a survey), whereas E_{y|x} are called the Experimental
        Probabilities (these come from a RCT).

        Each column of  O_{y|x}, P(x) and E_{y|x} sums to 1, so we chose as
        independent dofs (degrees of freedom) the probabilties O_{1|0},
        O_{1|1}, P(x=1), E_{1|0} and E_{1|1}.

        A given O_{y|x} imposes bounds on E_{y|x} that must be obeyed for
        consistency. This class also calculates those bounds.

        In this app, we consider a Bounder object bounder_m for males,
        and a Bounder object bounder_f for females.

        In this app, we consider two cases:

        1. Only Observational data

        2. Both Observational and Experimental data. For case 2, we allow
        the user to impose the additional constraints of exogeneity, strong
        exogeneity and monotonicity.

        The more constraints, the tighter the bounds on the trio (PNS, PN,
        PS), which I like to refer to as PNS3.

        Attributes
        ----------
        e0b0 : float
            E_{0|0}
        e0b1 : float
            E_{0|1}
        e1b0 : float
            E_{1|0}
        e1b1 : float
            E_{1|1}
        e_y_bar_x : np.array[shape=(2, 2)]
            E_{y|x}
        exogeneity : bool
        monotonicity : bool
        o00 : float
            O_{0,0}
        o01 : float
            O_{0,1}
        o0b0 : float
            O_{0|0}
        o0b1 : float
            O_{0|1}
        o10 : float
            O_{1,0}
        o11 : float
            O_{1,1}
        o1b0 : float
            O_{1|0}
        o1b1 : float
            O_{1|1}
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        px : float
            P(x)
        px0 : float
            P(x=0)
        px1 : float
            P(x=1}
        strong_exo : bool

        Parameters
        ----------
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        px : np.array[shape=(2, )]
            P(x)
        e_y_bar_x : np.array[shape=(2, )], None
            E_{y|x}

        monotonicity : bool
        exogeneity : bool
        strong_exo : bool
        """

        Bounder.check_prob_vec(px)
        self.px = px
        self.px0 = px[0]
        self.px1 = px[1]

        Bounder.check_2d_trans_matrix(o_y_bar_x)
        self.o_y_bar_x = o_y_bar_x
        self.o0b0 = o_y_bar_x[0, 0]
        self.o0b1 = o_y_bar_x[0, 1] 
        self.o1b0 = o_y_bar_x[1, 0]
        self.o1b1 = o_y_bar_x[1, 1]
        self.o00 = o_y_bar_x[0, 0]*px[0]
        self.o01 = o_y_bar_x[1, 0]*px[0]   # oxy and oybx  so flip x,y
        self.o10 = o_y_bar_x[0, 1]*px[1]   # oxy and oybx  so flip x,y
        self.o11 = o_y_bar_x[1, 1]*px[1]

        self.e_y_bar_x = None
        self.e0b0 = None
        self.e0b1 = None
        self.e1b0 = None
        self.e1b1 = None
        if e_y_bar_x is not None:
            self.set_exp_probs(e_y_bar_x)

        self.exogeneity = exogeneity
        self.monotonicity = monotonicity
        self.strong_exo = strong_exo
        if strong_exo:
            self.exogeneity = True

        np.set_printoptions(precision=6, floatmode="fixed")

    def set_obs_probs(self, o_y_bar_x, px):
        """
        This method refreshes the class attributes with new observational
        probabilities. It checks the consistency of the input.

        Parameters
        ----------
        o_y_bar_x : np.array[shape=(2, 2)]
            O_{y|x}
        px : np.array[shape=(2, )]
            P(x)

        Returns
        -------
        None

        """
        Bounder.check_prob_vec(px)
        self.px = px
        self.px0 = px[0]
        self.px1 = px[1]

        Bounder.check_2d_trans_matrix(o_y_bar_x)
        self.o_y_bar_x = o_y_bar_x
        self.o0b0 = o_y_bar_x[0, 0]
        self.o0b1 = o_y_bar_x[0, 1]
        self.o1b0 = o_y_bar_x[1, 0]
        self.o1b1 = o_y_bar_x[1, 1]
        self.o00 = o_y_bar_x[0, 0]*px[0]
        self.o01 = o_y_bar_x[1, 0]*px[0]
        self.o10 = o_y_bar_x[0, 1]*px[1]
        self.o11 = o_y_bar_x[1, 1]*px[1]

    def set_exp_probs(self, e_y_bar_x):
        """
        This method refreshes the class attributes with new experimental
        probabilities. It checks the consistency of the input.

        Parameters
        ----------
        e_y_bar_x : np.array[shape=(2, 2)]
            E_{y|x}

        Returns
        -------
        None

        """
        Bounder.check_2d_trans_matrix(e_y_bar_x)
        self.e_y_bar_x = e_y_bar_x
        self.e0b0 = e_y_bar_x[0, 0]
        self.e0b1 = e_y_bar_x[0, 1]
        self.e1b0 = e_y_bar_x[1, 0]
        self.e1b1 = e_y_bar_x[1, 1]

    @staticmethod
    def check_2d_trans_matrix(mat):
        """
        Checks the 2x2 transition probability mat is well defined.

        Parameters
        ----------
        mat : np.array[shape=(2, 2)]

        Returns
        -------
        None

        """
        assert mat.shape == (2, 2)
        assert (0 <= mat).all()
        assert (mat <= 1).all()  # can't check 0<=mat<=1 at once
        assert np.abs(sum(mat[:, 0]) - 1) < 1e-5
        assert np.abs(sum(mat[:, 1]) - 1) < 1e-5

    @staticmethod
    def check_prob_vec(vec):
        """
        Checks that the probability vector vec is well defined.

        Parameters
        ----------
        vec : np.array[shape=(2, )]

        Returns
        -------
        None

        """
        assert vec.shape == (2, )
        assert (0 <= vec).all()
        assert (vec <= 1).all()
        assert np.abs(sum(vec) - 1) < 1e-5

    def get_py(self):
        """
        Returns P(y) for y=0,1.

        Returns
        -------
        (float, float)
            P(y=0), P(y=1)

        """
        py0 = self.o00 + self.o10
        py1 = self.o01 + self.o11
        return py0, py1

    def get_e_star_bar_star(self):
        """
        Returns E_{*|*} = E_{0|0} + E_{1|1}.

        Returns
        -------
        float
            E_{*|*} = E_{0|0} + E_{1|1}

        """
        return self.e0b0 + self.e1b1

    def get_o_star_bar_star(self):
        """
        Returns O_{*|*} = O_{0|0} + O_{1|1}.

        Returns
        -------
        float
            O_{*|*} = O_{0|0} + O_{1|1}

        """
        return self.o0b0 + self.o1b1

    def get_o_star_star(self):
        """
        Returns O_{*,*} = O_{0,0} + O_{1,1}.

        Returns
        -------
        float
            O_{*,*} = O_{0,0} + O_{1,1}

        """
        return self.o00 + self.o11

    def get_pns3_bds(self):
        """
        Returns bounds for PNS3 = (PNS, N, PS).

        Returns
        -------
        np.array of shape =(3,2)
            [[PNS_low, PNS_high],
            [PN_low, PN_high],
            [PS_low, PS_high]]

        """

        if self.e_y_bar_x is None:         # no experimental data
            pns_bds = [0, self.get_o_star_star()]
            pn_bds = [0, 1]
            ps_bds = [0, 1]
            return np.array([pns_bds, pn_bds, ps_bds])
        else:
            py0, py1 = self.get_py()
            e_star_bar_star = self.get_e_star_bar_star()
            o_star_bar_star = self.get_o_star_bar_star()
            o_star_star = self.get_o_star_star()

            if not self.exogeneity and not self.monotonicity:
                # pns bounds
                pns_left = max(
                    0,
                    e_star_bar_star - 1,
                    self.e0b0 - py0,
                    self.e1b1 - py1)
                pns_right = min(
                    self.e1b1,
                    self.e0b0,
                    o_star_star,
                    e_star_bar_star - o_star_star)

                # pn bounds
                if self.o11 <= 0:
                    pn_left = 0
                    pn_right = 1
                else:
                    pn_left = max(
                        0,
                        (self.e0b0 - py0)/self.o11)
                    pn_right = min(
                        1,
                        (self.e0b0 - self.o00)/self.o11)

                # ps bounds
                if self.o00 <= 0:
                    ps_left = 0
                    ps_right = 1
                else:
                    ps_left = max(
                        0,
                        (self.e1b1 - py1)/self.o00)
                    ps_right = min(
                        1,
                        (self.e1b1 - self.o11)/self.o00)

            elif self.exogeneity and not self.monotonicity:
                # pns bounds
                pns_left = max(
                    0,
                    o_star_bar_star - 1)
                pns_right = min(
                    self.o1b1,
                    self.o0b0)
                # pn bounds
                if self.o1b1 <= 0:
                    pn_left = 0
                    pn_right = 1
                else:
                    err = (self.o1b1 - self.o1b0)/self.o1b1
                    pn_left = max(0, err)
                    pn_right = min(1, self.o0b0/self.o1b1)

                # ps bounds
                if self.o0b0 <= 0:
                    ps_left = 0
                    ps_right = 1
                else:
                    err_tilde = (self.o0b0 - self.o0b1)/self.o0b0
                    ps_left = max(0, err_tilde)
                    ps_right = min(1, self.o1b1/self.o0b0)
            elif not self.exogeneity and self.monotonicity:
                # pns bounds
                pns_left = e_star_bar_star - 1
                pns_right = pns_left
                # pn bounds
                if self.o11 <= 0:
                    pn_left = 1
                else:
                    pn_left = (self.e0b0 - py0)/self.o11
                pn_right = pn_left
                # ps bounds
                if self.o00 <= 0:
                    ps_left = 1
                else:
                    ps_left = (self.e1b1 - py1)/self.o00
                ps_right = ps_left
            elif self.exogeneity and self.monotonicity:
                # pns bounds
                pns_left = o_star_bar_star - 1
                pns_right = pns_left
                # pn bounds
                if self.o11 <= 0:
                    pn_left = 1
                else:
                    pn_left = (self.o0b0 - py0)/self.o11
                pn_right = pn_left
                # ps bounds
                if self.o00 <= 0:
                    ps_left = 1
                else:
                    ps_left = (self.o1b1 - py1)/self.o00
                ps_right = ps_left
            else:
                assert False
            if self.strong_exo:
                if self.o1b1 > 0:
                    pn_left = pns_left/self.o1b1
                    pn_right = pn_left
                if self.o0b0 > 0:
                    ps_left = pns_left/self.o0b0
                    ps_right = ps_left
            pns_bds = [pns_left, pns_right]
            pn_bds = [pn_left, pn_right]
            ps_bds = [ps_left, ps_right]
            return np.array([pns_bds, pn_bds, ps_bds])

    def get_exp_probs_bds(self):
        """
        Returns elementwise bounds on the transition probability matrix E_{
        y|x}.

        Returns
        -------
        np.array[shape=(2, 2)], np.array[shape=(2, 2)]

            E_{y|x} lower bound, E_{y|x} upper bound

        """
        left_bds_e_y_bar_x = np.zeros((2, 2))
        right_bds_e_y_bar_x = np.zeros((2, 2))
        if not self.monotonicity:
            left_bds_e_y_bar_x[1, 1] = self.o11
            right_bds_e_y_bar_x[1, 1] = 1 - self.o10
            left_bds_e_y_bar_x[1, 0] = self.o01
            right_bds_e_y_bar_x[1, 0] = 1 - self.o00

            # use if a <= x <= b then 1-b <= 1-x <= 1-a
            left_bds_e_y_bar_x[0, 1] = self.o10
            right_bds_e_y_bar_x[0, 1] = 1 - self.o11
            left_bds_e_y_bar_x[0, 0] = self.o00
            right_bds_e_y_bar_x[0, 0] = 1 - self.o01
        else:
            py0, py1 = self.get_py()

            left_bds_e_y_bar_x[1, 1] = py1
            right_bds_e_y_bar_x[1, 1] = 1 - self.o10
            left_bds_e_y_bar_x[1, 0] = self.o01
            right_bds_e_y_bar_x[1, 0] = py1

            # use if a <= x <= b then 1-b <= 1-x <= 1-a
            left_bds_e_y_bar_x[0, 1] = self.o10
            right_bds_e_y_bar_x[0, 1] = py0
            left_bds_e_y_bar_x[0, 0] = py0
            right_bds_e_y_bar_x[0, 0] = 1 - self.o01

        return left_bds_e_y_bar_x, right_bds_e_y_bar_x

    @staticmethod
    def get_joint_exp_probs_bds(bounder_m, bounder_f):
        """
        This method merges the bounds obtained from calling
        bounder_m.get_exp_probs_bds( ) and bounder_f.get_exp_probs_bds(). It
        returns the merged joint bounds. It doesn't check whether the joint
        bounds are consistent.

        Parameters
        ----------
        bounder_m : Bounder
            Bounder object for males.
        bounder_f : Bounder
            Bounder object for females.

        Returns
        -------
        np.array[shape=(2, 2)], np.array[shape=(2, 2)]

            max(E_{y|x,m} lower bound, E_{y|x,f} lower bound)
            min(E_{y|x,f} upper bound, E_{y|x,f} upper bound)

        """
        left_m, right_m = bounder_m.get_exp_probs_bds()
        left_f, right_f = bounder_f.get_exp_probs_bds()

        left = np.maximum(left_m, left_f)  # element-wise max
        assert left.shape == (2, 2)
        right = np.minimum(right_m, right_f)  # element-wise min
        assert right.shape == (2, 2)
        if (right-left >= 0).all():
            return left, right
        else:
            return None, None

    def print_exp_probs(self):
        """
        Prints the Experimental probabilities E_{y|x}.

        Returns
        -------
        None

        """
        print("E_{y|x}=\n", self.e_y_bar_x)

    def print_obs_probs(self):
        """
        Prints the Observational Probabilities O_{y|x} and P(x).

        Returns
        -------
        None

        """
        print("O_{y|x}=\n", self.o_y_bar_x)
        print("P_x=\n", self.px)

    def print_all_probs(self):
        """
        Calls print_obs_probs() and print_exp_probs()

        Returns
        -------
        None

        """
        self.print_obs_probs()
        self.print_exp_probs()

    def print_exp_probs_bds(self,
                            left_bds_e_y_bar_x,
                            right_bds_e_y_bar_x):
        """
        Prints left (low) and right (high) bounds for each element of E_{y|x}.

        Parameters
        ----------
        left_bds_e_y_bar_x : np.array[shape=(2, 2)]
            left (low) bounds for each element of E_{y|x}
        right_bds_e_y_bar_x : np.array[shape=(2, 2)]
            right (high) bounds for each element of E_{y|x}

        Returns
        -------

        """
        print("E_{0|0}: %.6f <= %.6f <= %.6f"
            % (left_bds_e_y_bar_x[0, 0],
            0 if self.e0b0 is None else self.e0b0,
            right_bds_e_y_bar_x[0, 0]))
        print("E_{0|1}: %.6f <= %.6f <= %.6f"
            % (left_bds_e_y_bar_x[0, 1],
            0 if self.e0b1 is None else self.e0b1,
            right_bds_e_y_bar_x[0, 1]))
        print("E_{1|0}: %.6f <= %.6f <= %.6f"
            % (left_bds_e_y_bar_x[1, 0],
            0 if self.e1b0 is None else self.e1b0,
            right_bds_e_y_bar_x[1, 0]))
        print("E_{1|1}: %.6f <= %.6f <= %.6f"
            % (left_bds_e_y_bar_x[1, 1],
            0 if self.e1b1 is None else self.e1b1,
            right_bds_e_y_bar_x[1, 1]))
        
    @staticmethod
    def print_pns3_bds(pns3_bds):
        """
        Prints bounds on PNS3 = (PNS, PN, PS).

        Parameters
        ----------
        pns3_bds : np.array[shape=(3, 2)]

        Returns
        -------
        None

        """
        
        print("%.6f <= PNS <= %.6f" % (pns3_bds[0, 0], pns3_bds[0, 1]))
        print("%.6f <= PN  <= %.6f" % (pns3_bds[1, 0], pns3_bds[1, 1]))
        print("%.6f <= PS  <= %.6f" % (pns3_bds[2, 0], pns3_bds[2, 1]))


if __name__ == "__main__":
    def main():
        print("FEMALE-----------------------")
        print("input probabilities obtained from exp. and obs. data:")
        e_y_bar_x_f = np.array([[.79, .52],
                                [.21, .48]])
        o_y_bar_x_f = np.array([[.3, .73],
                               [.7, .27]])
        px_f = np.array([.3, .7])
        f = Bounder(o_y_bar_x_f, px_f,
            e_y_bar_x=e_y_bar_x_f)
        f.print_all_probs()
        print("---------------------------")
        print("Check exp. data is within bds imposed by obs. data:")
        left_bds_e_y_bar_x_f, right_bds_e_y_bar_x_f =\
            f.get_exp_probs_bds()
        f.print_exp_probs_bds(left_bds_e_y_bar_x_f, right_bds_e_y_bar_x_f)
        print("---------------------------")
        pns3_bds_f = f.get_pns3_bds()
        Bounder.print_pns3_bds(pns3_bds_f)

        print("MALE--------------------------")
        print("input probabilities obtained from exp. and obs. data:")
        e_y_bar_x_m = np.array([[.79, .51],
                                [.21, .49]])
        o_y_bar_x_m = np.array([[.3, .3],
                                [.7, .7]])
        px_m = np.array([.3, .7])

        m = Bounder(o_y_bar_x_m, px_m,
            e_y_bar_x=e_y_bar_x_m)
        m.print_all_probs()
        print("---------------------------")
        print("Check exp. data is within bds imposed by obs. data:")
        left_bds_e_y_bar_x_m, right_bds_e_y_bar_x_m =\
            m.get_exp_probs_bds()
        m.print_exp_probs_bds(left_bds_e_y_bar_x_m, right_bds_e_y_bar_x_m)
        print("---------------------------")
        pns3_bds_m = m.get_pns3_bds()
        Bounder.print_pns3_bds(pns3_bds_m)

    main()

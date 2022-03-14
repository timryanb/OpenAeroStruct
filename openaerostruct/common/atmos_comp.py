from collections import namedtuple

import numpy as np
from scipy.interpolate import Akima1DInterpolator as Akima

import openmdao.api as om


"""United States standard atmosphere 1976 tables, data obtained from http://www.digitaldutch.com/atmoscalc/index.htm"""
USatm1976Data = namedtuple("USatm1976Data", ["alt", "T", "P", "rho", "speed_of_sound", "viscosity"])


USatm1976Data.alt = np.array(
    [
        -1000,
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
        11000,
        12000,
        13000,
        14000,
        15000,
        16000,
        17000,
        18000,
        19000,
        20000,
        21000,
        22000,
        23000,
        24000,
        25000,
        26000,
        27000,
        28000,
        29000,
        30000,
        31000,
        32000,
        33000,
        34000,
        35000,
        36000,
        37000,
        38000,
        39000,
        40000,
        41000,
        42000,
        43000,
        44000,
        45000,
        46000,
        47000,
        48000,
        49000,
        50000,
        51000,
        52000,
        53000,
        54000,
        55000,
        56000,
        57000,
        58000,
        59000,
        60000,
        61000,
        62000,
        63000,
        64000,
        65000,
        66000,
        67000,
        68000,
        69000,
        70000,
        71000,
        72000,
        73000,
        74000,
        75000,
        76000,
        77000,
        78000,
        79000,
        80000,
        81000,
        82000,
        83000,
        84000,
        85000,
        86000,
        87000,
        88000,
        89000,
        90000,
        91000,
        92000,
        93000,
        94000,
        95000,
        96000,
        97000,
        98000,
        99000,
        100000,
        105000,
        110000,
        115000,
        120000,
        125000,
        130000,
        135000,
        140000,
        145000,
        150000,
    ]
)  # units='ft'

USatm1976Data.T = np.array(
    [
        522.236,
        518.67,
        515.104,
        511.538,
        507.972,
        504.405,
        500.839,
        497.273,
        493.707,
        490.141,
        486.575,
        483.008,
        479.442,
        475.876,
        472.31,
        468.744,
        465.178,
        461.611,
        458.045,
        454.479,
        450.913,
        447.347,
        443.781,
        440.214,
        436.648,
        433.082,
        429.516,
        425.95,
        422.384,
        418.818,
        415.251,
        411.685,
        408.119,
        404.553,
        400.987,
        397.421,
        393.854,
        390.288,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        389.97,
        390.18,
        390.729,
        391.278,
        391.826,
        392.375,
        392.923,
        393.472,
        394.021,
        394.569,
        395.118,
        395.667,
        396.215,
        396.764,
        397.313,
        397.861,
        398.41,
        398.958,
        399.507,
        400.056,
        400.604,
        401.153,
        401.702,
        402.25,
        402.799,
        403.348,
        403.896,
        404.445,
        404.994,
        405.542,
        406.091,
        406.639,
        407.188,
        407.737,
        408.285,
        408.834,
        411.59,
        419.271,
        426.952,
        434.633,
        442.314,
        449.995,
        457.676,
        465.357,
        473.038,
        480.719,
    ]
)  # units='degR'

USatm1976Data.P = np.array(
    [
        15.2348,
        14.6959,
        14.1726,
        13.6644,
        13.1711,
        12.6923,
        12.2277,
        11.777,
        11.3398,
        10.9159,
        10.5049,
        10.1065,
        9.7204,
        9.34636,
        8.98405,
        8.63321,
        8.29354,
        7.96478,
        7.64665,
        7.33889,
        7.4123,
        6.75343,
        6.47523,
        6.20638,
        5.94664,
        5.69578,
        5.45355,
        5.21974,
        4.9941,
        4.77644,
        4.56651,
        4.36413,
        4.16906,
        3.98112,
        3.8001,
        3.6258,
        3.45803,
        3.29661,
        3.14191,
        2.99447,
        2.85395,
        2.72003,
        2.59239,
        2.47073,
        2.35479,
        2.24429,
        2.13897,
        2.0386,
        1.94293,
        1.85176,
        1.76486,
        1.68204,
        1.60311,
        1.52788,
        1.45618,
        1.38785,
        1.32272,
        1.26065,
        1.20149,
        1.14511,
        1.09137,
        1.04016,
        0.991347,
        0.944827,
        0.900489,
        0.858232,
        0.817958,
        0.779578,
        0.743039,
        0.708261,
        0.675156,
        0.643641,
        0.613638,
        0.585073,
        0.557875,
        0.531976,
        0.507313,
        0.483825,
        0.461455,
        0.440148,
        0.419853,
        0.400519,
        0.382101,
        0.364553,
        0.347833,
        0.331902,
        0.31672,
        0.302253,
        0.288464,
        0.275323,
        0.262796,
        0.250856,
        0.239473,
        0.228621,
        0.218275,
        0.20841,
        0.199003,
        0.190032,
        0.181478,
        0.173319,
        0.165537,
        0.158114,
        0.12582,
        0.10041,
        0.08046,
        0.064729,
        0.0522725,
        0.0423688,
        0.0344637,
        0.0281301,
        0.0230369,
        0.0189267,
    ]
)  # units='psi'

USatm1976Data.rho = np.array(
    [
        0.00244752,
        0.00237717,
        0.00230839,
        0.00224114,
        0.00217539,
        0.00211114,
        0.00204834,
        0.00198698,
        0.00192704,
        0.0018685,
        0.00181132,
        0.00175549,
        0.00170099,
        0.00164779,
        0.00159588,
        0.00154522,
        0.00149581,
        0.00144761,
        0.00140061,
        0.00135479,
        0.00131012,
        0.00126659,
        0.00122417,
        0.00118285,
        0.0011426,
        0.00110341,
        0.00106526,
        0.00102812,
        0.000991984,
        0.000956827,
        0.000922631,
        0.000889378,
        0.00085705,
        0.000825628,
        0.000795096,
        0.000765434,
        0.000736627,
        0.000708657,
        0.000675954,
        0.000644234,
        0.000614002,
        0.000585189,
        0.000557728,
        0.000531556,
        0.000506612,
        0.000482838,
        0.00046018,
        0.000438586,
        0.000418004,
        0.000398389,
        0.000379694,
        0.000361876,
        0.000344894,
        0.000328709,
        0.000313284,
        0.000298583,
        0.000284571,
        0.000271217,
        0.00025849,
        0.00024636,
        0.000234799,
        0.000223781,
        0.000213279,
        0.000203271,
        0.000193732,
        0.000184641,
        0.000175976,
        0.000167629,
        0.000159548,
        0.000151867,
        0.000144566,
        0.000137625,
        0.000131026,
        0.000124753,
        0.000118788,
        0.000113116,
        0.000107722,
        0.000102592,
        9.77131e-05,
        9.30725e-05,
        8.86582e-05,
        0.000084459,
        8.04641e-05,
        7.66632e-05,
        7.30467e-05,
        6.96054e-05,
        6.63307e-05,
        6.32142e-05,
        6.02481e-05,
        5.74249e-05,
        5.47376e-05,
        5.21794e-05,
        4.97441e-05,
        4.74254e-05,
        4.52178e-05,
        4.31158e-05,
        0.000041114,
        3.92078e-05,
        3.73923e-05,
        3.56632e-05,
        3.40162e-05,
        3.24473e-05,
        2.56472e-05,
        2.00926e-05,
        1.58108e-05,
        1.24948e-05,
        9.9151e-06,
        7.89937e-06,
        6.3177e-06,
        5.07154e-06,
        4.08586e-06,
        3.30323e-06,
    ]
)  # units='slug/ft**3'

USatm1976Data.a = np.array(
    [
        1120.28,
        1116.45,
        1112.61,
        1108.75,
        1104.88,
        1100.99,
        1097.09,
        1093.18,
        1089.25,
        1085.31,
        1081.36,
        1077.39,
        1073.4,
        1069.4,
        1065.39,
        1061.36,
        1057.31,
        1053.25,
        1049.18,
        1045.08,
        1040.97,
        1036.85,
        1032.71,
        1028.55,
        1024.38,
        1020.19,
        1015.98,
        1011.75,
        1007.51,
        1003.24,
        998.963,
        994.664,
        990.347,
        986.01,
        981.655,
        977.28,
        972.885,
        968.471,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.076,
        968.337,
        969.017,
        969.698,
        970.377,
        971.056,
        971.735,
        972.413,
        973.091,
        973.768,
        974.445,
        975.121,
        975.797,
        976.472,
        977.147,
        977.822,
        978.496,
        979.169,
        979.842,
        980.515,
        981.187,
        981.858,
        982.53,
        983.2,
        983.871,
        984.541,
        985.21,
        985.879,
        986.547,
        987.215,
        987.883,
        988.55,
        989.217,
        989.883,
        990.549,
        991.214,
        994.549,
        1003.79,
        1012.94,
        1022.01,
        1031,
        1039.91,
        1048.75,
        1057.52,
        1066.21,
        1074.83,
    ]
)  # units='ft/s'

USatm1976Data.viscosity = np.array(
    [
        3.81e-07,
        3.78e-07,
        3.76e-07,
        3.74e-07,
        3.72e-07,
        3.70e-07,
        3.68e-07,
        3.66e-07,
        3.64e-07,
        3.62e-07,
        3.60e-07,
        3.57e-07,
        3.55e-07,
        3.53e-07,
        3.51e-07,
        3.49e-07,
        3.47e-07,
        3.45e-07,
        3.42e-07,
        3.40e-07,
        3.38e-07,
        3.36e-07,
        3.34e-07,
        3.31e-07,
        3.29e-07,
        3.27e-07,
        3.25e-07,
        3.22e-07,
        3.20e-07,
        3.18e-07,
        3.16e-07,
        3.13e-07,
        3.11e-07,
        3.09e-07,
        3.06e-07,
        3.04e-07,
        3.02e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        2.99e-07,
        3.00e-07,
        3.00e-07,
        3.00e-07,
        3.01e-07,
        3.01e-07,
        3.01e-07,
        3.02e-07,
        3.02e-07,
        3.03e-07,
        3.03e-07,
        3.03e-07,
        3.04e-07,
        3.04e-07,
        3.04e-07,
        3.05e-07,
        3.05e-07,
        3.05e-07,
        3.06e-07,
        3.06e-07,
        3.06e-07,
        3.07e-07,
        3.07e-07,
        3.08e-07,
        3.08e-07,
        3.08e-07,
        3.09e-07,
        3.09e-07,
        3.09e-07,
        3.10e-07,
        3.10e-07,
        3.10e-07,
        3.11e-07,
        3.11e-07,
        3.11e-07,
        3.13e-07,
        3.18e-07,
        3.23e-07,
        3.28e-07,
        3.33e-07,
        3.37e-07,
        3.42e-07,
        3.47e-07,
        3.51e-07,
        3.56e-07,
    ]
)  # units='lbf*s/ft**2'

T_interp = Akima(USatm1976Data.alt, USatm1976Data.T)
P_interp = Akima(USatm1976Data.alt, USatm1976Data.P)
rho_interp = Akima(USatm1976Data.alt, USatm1976Data.rho)
a_interp = Akima(USatm1976Data.alt, USatm1976Data.a)
viscosity_interp = Akima(USatm1976Data.alt, USatm1976Data.viscosity)

T_interp_deriv = T_interp.derivative(1)
P_interp_deriv = P_interp.derivative(1)
rho_interp_deriv = rho_interp.derivative(1)
a_interp_deriv = a_interp.derivative(1)
viscosity_interp_deriv = viscosity_interp.derivative(1)


class AtmosComp(om.ExplicitComponent):
    def setup(self):
        self.add_input("altitude", val=1.0, units="ft")
        self.add_input("Mach_number", val=1.0)

        self.add_output("T", val=1.0, units="degR")
        self.add_output("P", val=1.0, units="psi")
        self.add_output("rho", val=1.0, units="slug/ft**3")
        self.add_output("speed_of_sound", val=1.0, units="ft/s")
        self.add_output("mu", val=1.0, units="lbf*s/ft**2")
        self.add_output("v", val=1.0, units="ft/s")

        self.declare_partials(["T", "P", "rho", "speed_of_sound", "mu", "v"], "altitude")
        self.declare_partials("v", "Mach_number")

    def compute(self, inputs, outputs):

        outputs["T"] = T_interp(inputs["altitude"])
        outputs["P"] = P_interp(inputs["altitude"])
        outputs["rho"] = rho_interp(inputs["altitude"])
        outputs["speed_of_sound"] = a_interp(inputs["altitude"])
        outputs["mu"] = viscosity_interp(inputs["altitude"])

        outputs["v"] = outputs["speed_of_sound"] * inputs["Mach_number"]

    def compute_partials(self, inputs, partials):

        partials["T", "altitude"] = T_interp_deriv(inputs["altitude"])
        partials["P", "altitude"] = P_interp_deriv(inputs["altitude"])
        partials["rho", "altitude"] = rho_interp_deriv(inputs["altitude"])
        partials["speed_of_sound", "altitude"] = a_interp_deriv(inputs["altitude"])
        partials["mu", "altitude"] = viscosity_interp_deriv(inputs["altitude"])

        partials["v", "altitude"] = a_interp_deriv(inputs["altitude"]) * inputs["Mach_number"]
        partials["v", "Mach_number"] = a_interp(inputs["altitude"])
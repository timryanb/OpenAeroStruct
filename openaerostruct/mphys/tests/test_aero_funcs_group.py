import unittest

from openaerostruct.mphys.aero_funcs_group import AeroFuncsGroup
from openaerostruct.utils.testing import run_test, get_default_surfaces


class Test(unittest.TestCase):
    def test(self):
        surfaces = get_default_surfaces()
        surfaces[0]["with_viscous"] = False

        comp = AeroFuncsGroup(surfaces=[surfaces[0]], write_solution=False)
        comp.set_input_defaults("wing.widths", val=[1.0, 1.0, 1.0])
        comp.set_input_defaults("v", val=1.0)
        comp.set_input_defaults("rho", val=1.0)

        run_test(self, comp, complex_flag=True, method="cs")


if __name__ == "__main__":
    unittest.main()

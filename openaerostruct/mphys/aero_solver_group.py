import openmdao.api as om

from openaerostruct.aerodynamics.compressible_states import CompressibleVLMStates
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates


class AeroSolverGroup(om.Group):
    """
    Group that contains the states for a incompresible/compressible aerodynamic analysis.
    """

    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)
        self.options.declare("compressible", default=True, desc="prandtl glauert compressibiity flag", recordable=True)

    def setup(self):
        self.surfaces = self.options["surfaces"]

        # Loop through each surface and promote relevant parameters
        proms_in = [("alpha", "aoa"), ("beta", "yaw")]
        proms_out = []
        for surface in self.surfaces:
            name = surface["name"]

            proms_in.append((name + "_normals", name + ".normals"))
            proms_out.append((name + "_sec_forces", name + ".sec_forces"))

            self.add_subsystem(name, VLMGeometry(surface=surface), promotes_inputs=[("def_mesh", name + "_def_mesh")])

        if self.options["compressible"]:
            proms_in.append(("Mach_number", "mach"))
            aero_states = CompressibleVLMStates(surfaces=self.surfaces)
        else:
            aero_states = VLMStates(surfaces=self.surfaces)

        self.add_subsystem(
            "solver",
            aero_states,
            promotes_inputs=proms_in + ["*"],
            promotes_outputs=proms_out + ["circulations", "*_mesh_point_forces"],
        )

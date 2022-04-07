import copy
import numpy as np
import openmdao.api as om
from mphys.builder import Builder
from openaerostruct.aerodynamics.compressible_states import CompressibleVLMStates
from openaerostruct.aerodynamics.geometry import VLMGeometry
from openaerostruct.aerodynamics.states import VLMStates
from openaerostruct.aerodynamics.functionals import VLMFunctionals
from openaerostruct.functionals.total_aero_performance import TotalAeroPerformance
from openaerostruct.mphys.surface_contours import SurfaceContour


class AeroMesh(om.IndepVarComp):
    """
    Component to read the initial mesh coordinates with OAS
    """

    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)

    def setup(self):
        self.surfaces = self.options["surfaces"]
        nnodes = get_number_of_nodes(self.surfaces)
        src_indices = get_src_indices(self.surfaces)
        xpts = np.zeros(nnodes * 3)
        for surface in self.surfaces:
            surf_name = surface["name"]
            xpts[src_indices[surf_name]] = surface["mesh"]
        self.add_output(
            "x_aero0",
            distributed=False,
            val=xpts,
            shape=xpts.size,
            units="m",
            desc="aero node coordinates",
            tags=["mphys_coordinates"],
        )


class DemuxSurfaceMesh(om.ExplicitComponent):
    """
    Demux surface coordinates from flattened array.
    """

    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)

    def setup(self):
        self.surfaces = self.options["surfaces"]

        self.nnodes = get_number_of_nodes(self.surfaces)
        self.src_indices = get_src_indices(self.surfaces)

        # OpenMDAO part of setup
        self.add_input(
            "x_aero",
            distributed=False,
            shape=self.nnodes * 3,
            units="m",
            desc="flattened aero mesh coordinates " "for all oas surfaces",
            tags=["mphys_coupling"],
        )
        for surface in self.surfaces:
            surf_name = surface["name"]
            mesh = surface["mesh"]
            self.add_output(
                f"{surf_name}_def_mesh",
                distributed=False,
                shape=mesh.shape,
                units="m",
                desc="Array defining the nodal coordinates " "of the lifting surface.",
                tags=["mphys_coupling"],
            )

    def compute(self, inputs, outputs):
        for surface in self.surfaces:
            surf_name = surface["name"]
            outputs[surf_name + "_def_mesh"] = inputs["x_aero"][self.src_indices[surf_name]]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for surface in self.surfaces:
                surf_name = surface["name"]
                if "x_aero" in d_inputs and surf_name + "_def_mesh" in d_outputs:
                    d_outputs[surf_name + "_def_mesh"] += d_inputs["x_aero"][self.src_indices[surf_name]]
        if mode == "rev":
            for surface in self.surfaces:
                surf_name = surface["name"]
                if "x_aero" in d_inputs and surf_name + "_def_mesh" in d_outputs:
                    d_inputs["x_aero"][self.src_indices[surf_name]] += d_outputs[surf_name + "_def_mesh"]


class AeroSolverGroup(om.Group):
    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)
        self.options.declare("compressible", default=True, desc="prandtl glauert compressibiity flag", recordable=True)

    def setup(self):
        self.surfaces = self.options["surfaces"]

        # Loop through each surface and promote relevant parameters
        proms_in = []
        proms_out = []
        for surface in self.surfaces:
            name = surface["name"]

            proms_in.append((name + "_normals", name + ".normals"))
            proms_out.append((name + "_sec_forces", name + ".sec_forces"))

            self.add_subsystem(name, VLMGeometry(surface=surface), promotes_inputs=[("def_mesh", name + "_def_mesh")])

        if self.options["compressible"]:
            aero_states = CompressibleVLMStates(surfaces=self.surfaces)
        else:
            aero_states = VLMStates(surfaces=self.surfaces)

        self.add_subsystem(
            "solver",
            aero_states,
            promotes_inputs=proms_in + ["*"],
            promotes_outputs=proms_out + ["circulations", "*_mesh_point_forces"],
        )


class MuxSurfaceForces(om.ExplicitComponent):
    """
    Demux surface coordinates from flattened array.
    """

    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)

    def setup(self):
        self.surfaces = self.options["surfaces"]

        self.nnodes = get_number_of_nodes(self.surfaces)
        self.src_indices = get_src_indices(self.surfaces)

        # OpenMDAO part of setup
        for surface in self.surfaces:
            surf_name = surface["name"]
            mesh = surface["mesh"]
            self.add_input(
                f"{surf_name}_mesh_point_forces",
                distributed=False,
                shape=mesh.shape,
                units="N",
                desc="Array defining the aero forces " "on mesh nodes of the lifting surface.",
                tags=["mphys_coupling"],
            )

        self.add_output(
            "f_aero",
            distributed=False,
            shape=self.nnodes * 3,
            val=0.0,
            units="N",
            desc="flattened aero forces for all oas surfaces",
            tags=["mphys_coupling"],
        )

    def compute(self, inputs, outputs):
        for surface in self.surfaces:
            surf_name = surface["name"]
            outputs["f_aero"][self.src_indices[surf_name]] = inputs[surf_name + "_mesh_point_forces"]

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        if mode == "fwd":
            for surface in self.surfaces:
                surf_name = surface["name"]
                if "f_aero" in d_outputs and surf_name + "_mesh_point_forces" in d_inputs:
                    d_outputs["f_aero"][self.src_indices[surf_name]] += d_inputs[surf_name + "_mesh_point_forces"]

        if mode == "rev":
            for surface in self.surfaces:
                surf_name = surface["name"]
                if "f_aero" in d_outputs and surf_name + "_mesh_point_forces" in d_inputs:
                    d_inputs[surf_name + "_mesh_point_forces"] += d_outputs["f_aero"][self.src_indices[surf_name]]


class AeroCouplingGroup(om.Group):
    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)
        self.options.declare("compressible", default=True, desc="prandtl glauert compressibiity flag", recordable=True)

    def setup(self):
        self.surfaces = self.options["surfaces"]
        self.compressible = self.options["compressible"]

        self.add_subsystem(
            "demuxer",
            DemuxSurfaceMesh(surfaces=self.surfaces),
            promotes_inputs=["x_aero"],
            promotes_outputs=["*_def_mesh"],
        )

        self.add_subsystem(
            "states",
            AeroSolverGroup(surfaces=self.surfaces, compressible=self.compressible),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem(
            "muxer",
            MuxSurfaceForces(surfaces=self.surfaces),
            promotes_inputs=["*_mesh_point_forces"],
            promotes_outputs=["f_aero"],
        )


class AeroFuncsGroup(om.Group):
    def initialize(self):
        self.options.declare("surfaces", default=None, desc="oas surface dicts", recordable=False)
        self.options.declare("user_specified_Sref", types=bool)
        self.options.declare("write_solution", default=True)
        self.options.declare("output_dir")
        self.options.declare("scenario_name", default=None)

    def setup(self):
        self.surfaces = self.options["surfaces"]
        self.user_specified_Sref = self.options["user_specified_Sref"]

        proms_in = []
        for surface in self.surfaces:
            surf_name = surface["name"]
            self.add_subsystem(
                surf_name,
                VLMFunctionals(surface=surface),
                promotes_inputs=["v", "alpha", "beta", "Mach_number", "re", "rho"],
            )

            proms_in.append((surf_name + "_S_ref", surf_name + ".S_ref"))
            proms_in.append((surf_name + "_b_pts", surf_name + ".b_pts"))
            proms_in.append((surf_name + "_widths", surf_name + ".widths"))
            proms_in.append((surf_name + "_chords", surf_name + ".chords"))
            proms_in.append((surf_name + "_sec_forces", surf_name + ".sec_forces"))
            proms_in.append((surf_name + "_CL", surf_name + ".CL"))
            proms_in.append((surf_name + "_CD", surf_name + ".CD"))

        proms_out = ["CM", "CL", "CD"]
        if self.options["user_specified_Sref"]:
            proms_in.append("S_ref_total")
        else:
            proms_out.append("S_ref_total")

        # Add the total aero performance group to compute the CL, CD, and CM
        # of the total aircraft. This accounts for all lifting surfaces.
        self.add_subsystem(
            "total_perf",
            TotalAeroPerformance(surfaces=self.surfaces, user_specified_Sref=self.user_specified_Sref),
            promotes_inputs=proms_in + ["v", "rho", "cg"],
            promotes_outputs=proms_out,
        )

        proms_in = []
        for surface in self.surfaces:
            surf_name = surface["name"]
            proms_in.append((surf_name + "_sec_forces", surf_name + ".sec_forces"))

        if self.options["write_solution"]:
            self.add_subsystem(
                "solution_writer",
                SurfaceContour(
                    surfaces=self.surfaces,
                    base_name=self.options["scenario_name"],
                    output_dir=self.options["output_dir"],
                ),
                promotes_inputs=proms_in + ["*"],
            )


class AeroBuilder(Builder):

    def_options = {"user_specified_Sref": False, "compressible": True, "output_dir": "./", "write_solution": True}

    def __init__(self, surfaces, options=None):
        self.surfaces = surfaces
        # Copy default options
        self.options = copy.deepcopy(self.def_options)
        # Update with user-defined options
        if options:
            self.options.update(options)

    def initialize(self, comm):
        self.nnodes = get_number_of_nodes(self.surfaces)

    def get_coupling_group_subsystem(self, scenario_name=None):
        return AeroCouplingGroup(surfaces=self.surfaces, compressible=self.options["compressible"])

    def get_mesh_coordinate_subsystem(self, scenario_name=None):
        return AeroMesh(surfaces=self.surfaces)

    def get_post_coupling_subsystem(self, scenario_name=None):
        user_specified_Sref = self.options["user_specified_Sref"]
        return AeroFuncsGroup(
            surfaces=self.surfaces,
            write_solution=self.options["write_solution"],
            output_dir=self.options["output_dir"],
            user_specified_Sref=user_specified_Sref,
            scenario_name=scenario_name,
        )

    def get_ndof(self):
        return 3

    def get_number_of_nodes(self):
        """
        Get the number of nodes
        """
        return self.nnodes


def get_number_of_nodes(surfaces):
    """
    Get the total number of nodes over all surfaces.

    Parameters
    ----------
    surfaces : list[dict]
        List of all surfaces.

    Returns
    -------
    nnodes : int
        Total number of nodes across all surfaces.

    """
    nnodes = 0
    for surface in surfaces:
        nnodes += surface["mesh"].size // 3
    return nnodes


def get_src_indices(surfaces):
    """
    Get src indices for each surface that will project each mesh into a single flattened array.

    Parameters
    ----------
    surfaces : list[dict]
        List of all surfaces.

    Returns
    -------
    src_indices : dict
        Dictionary holding source indices sorted by surface name.

    """
    src_indices = {}
    nnodes = 0
    for surface in surfaces:
        surf_name = surface["name"]
        mesh = surface["mesh"]
        nx, ny, _ = mesh.shape
        surf_indices = np.arange(mesh.size) + nnodes
        src_indices[surf_name] = surf_indices.reshape(nx, ny, 3)
        nnodes += mesh.size
    return src_indices

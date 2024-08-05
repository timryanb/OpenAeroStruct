import numpy as np
import jax.numpy as jnp

import openmdao.api as om
from openaerostruct.aerodynamics.eval_mtx_prod import EvalVelMtx


class EvalVelocities(om.ExplicitComponent):
    """
    Compute the total velocities at each of the evaluation points for every
    panel in the entire system. This is the sum of the freestream and induced
    velocities caused by the circulations.

    Parameters
    ----------
    freestream_velocities[system_size, 3] : numpy array
        The rotated freestream velocities at each evaluation point for all
        lifting surfaces. system_size is the sum of the count of all panels
        for all lifting surfaces.
    circulations[system_size] : numpy array
        The vortex ring circulations obtained from solving the AIC linear
        system.
    vel_mtx[num_eval_points, nx - 1, ny - 1, 3] : numpy array
        The AIC matrix for the all lifting surfaces representing the aircraft.
        This has some sparsity pattern, but it is more dense than the FEM matrix
        and the entries have a wide range of magnitudes. One exists for each
        combination of surface name and evaluation points name.

    Returns
    -------
    velocities[num_eval_points, 3] : numpy array
        The actual velocities experienced at the evaluation points for each
        lifting surface in the system. This is the summation of the freestream
        velocities and the induced velocities caused by the circulations.

    """

    def initialize(self):
        self.options.declare("surfaces", types=list)
        self.options.declare("eval_name", types=str)
        self.options.declare("num_eval_points", types=int)

    def setup(self):
        surfaces = self.options["surfaces"]
        eval_name = self.options["eval_name"]
        num_eval_points = self.options["num_eval_points"]

        system_size = 0

        # Determine system_size by looping through each surface and summing
        # the number of panels.
        for surface in surfaces:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]
            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.add_input("freestream_velocities", shape=(system_size, 3), units="m/s")
        self.add_input("circulations", shape=system_size, units="m**2/s", tags=["mphys_coupling"])

        # Get the correct output name; the velocities output depends on which
        # set of evaluation points we use, either collocation or force.
        velocities_name = "{}_velocities".format(eval_name)
        self.add_output(velocities_name, shape=(num_eval_points, 3), units="m/s")

        # Set up indices to create the sparsity pattern for the derivatives.
        circulations_indices = np.arange(system_size)
        velocities_indices = np.arange(num_eval_points * 3).reshape((num_eval_points, 3))

        self.declare_partials(
            velocities_name,
            "circulations",
            rows=np.einsum("ik,j->ijk", velocities_indices, np.ones(system_size, int)).flatten(),
            cols=np.einsum("ik,j->ijk", np.ones((num_eval_points, 3), int), circulations_indices).flatten(),
        )

        # These derivatives are linear and don't change so we set the val here
        self.declare_partials(
            velocities_name,
            "freestream_velocities",
            val=1.0,
            rows=np.arange(3 * num_eval_points),
            cols=np.arange(3 * num_eval_points),
        )

        self.vector_names = []

        for surface in self.options["surfaces"]:
            name = surface["name"]
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            # The logic differs if the surface is symmetric or not, due to the
            # existence of the "ghost" surface; the reflection of the actual.
            ground_effect = surface.get("groundplane", False)
            if ground_effect:
                nx_actual = 2 * nx
            else:
                nx_actual = nx
            if surface["symmetry"]:
                ny_actual = 2 * ny - 1
            else:
                ny_actual = ny

            system_size += (nx - 1) * (ny - 1)
            vectors_name = "{}_{}_vectors".format(name, eval_name)
            self.add_input(vectors_name, shape=(self.system_size, nx_actual, ny_actual, 3), units="m")
            self.vector_names.append(vectors_name)

        self.add_input("alpha", val=1.0, units="deg", tags=["mphys_input"])

        self.aic_mtx = EvalVelMtx(self.options["surfaces"], eval_name)

    def compute(self, inputs, outputs):
        eval_name = self.options["eval_name"]
        num_eval_points = self.options["num_eval_points"]

        velocities_name = "{}_velocities".format(eval_name)
        alpha = jnp.asarray(inputs["alpha"])
        circulations = jnp.asarray(inputs["circulations"])
        vectors = {vector_name: jnp.asarray(inputs[vector_name]) for vector_name in self.vector_names}

        # Start with just the freestream velocities as the base for the output
        # velocities.
        outputs[velocities_name] = inputs["freestream_velocities"] + self.aic_mtx.compute_velocity(alpha, vectors, circulations)
    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        eval_name = self.options["eval_name"]
        num_eval_points = self.options["num_eval_points"]
        velocities_name = "{}_velocities".format(eval_name)

        if mode == "fwd":
            pass
        if mode == "rev":
            if velocities_name in d_outputs:
                d_func = jnp.asarray(d_outputs[velocities_name])

                alpha = jnp.asarray(inputs["alpha"])
                circulations = jnp.asarray(inputs["circulations"])
                vectors = {vector_name: jnp.asarray(inputs[vector_name]) for vector_name in self.vector_names}
                d_alpha, d_vectors, d_circulations = self.aic_mtx.compute_velocity_vjp(alpha, vectors, circulations, d_func)

                if "alpha" in d_inputs:
                    d_inputs["alpha"] += d_alpha

                for vec_name in d_vectors:
                    if vec_name in d_inputs:
                        d_inputs[vec_name] += d_vectors[vec_name]

                if "circulations" in d_inputs:
                    d_inputs["circulations"] += d_circulations


    # def compute_partials(self, inputs, partials):
    #     surfaces = self.options["surfaces"]
    #     eval_name = self.options["eval_name"]
    #     num_eval_points = self.options["num_eval_points"]
    #
    #     system_size = self.system_size
    #
    #     velocities_name = "{}_velocities".format(eval_name)
    #
    #     dv_dcirc = np.zeros((num_eval_points, system_size, 3))
    #
    #     ind_1 = 0
    #     ind_2 = 0
    #     for surface in surfaces:
    #         mesh = surface["mesh"]
    #         nx = mesh.shape[0]
    #         ny = mesh.shape[1]
    #         name = surface["name"]
    #         num = (nx - 1) * (ny - 1)
    #
    #         ind_2 += num
    #
    #         vel_mtx_name = "{}_{}_vel_mtx".format(name, eval_name)
    #
    #         partials[velocities_name, vel_mtx_name] = np.einsum(
    #             "ijk,j->ijk",
    #             np.ones((num_eval_points, num, 3)),
    #             inputs["circulations"][ind_1:ind_2],
    #         ).flatten()
    #
    #         dv_dcirc[:, ind_1:ind_2, :] = inputs[vel_mtx_name].reshape((num_eval_points, num, 3))
    #
    #         ind_1 += num
    #
    #     partials[velocities_name, "circulations"] = dv_dcirc.flatten()

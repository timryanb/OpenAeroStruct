import numpy as np
import jax.numpy as jnp
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.linalg import lu_factor, lu_solve
from openaerostruct.aerodynamics.eval_mtx_prod import EvalVelMtx

import openmdao.api as om


class SolveMatrix(om.ImplicitComponent):
    """
    Solve the AIC linear system to obtain the vortex ring circulations.

    Parameters
    ----------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.

    Returns
    -------
    circulations[system_size] : numpy array
        The vortex ring circulations obtained by solving the AIC linear system.

    """

    def initialize(self):
        self.options.declare("surfaces", types=list)

    def setup(self):
        system_size = 0
        self.eval_name = "coll_pts"

        for surface in self.options["surfaces"]:
            mesh = surface["mesh"]
            nx = mesh.shape[0]
            ny = mesh.shape[1]

            system_size += (nx - 1) * (ny - 1)

        self.system_size = system_size

        self.vector_names = []
        self.normal_names = []

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
            vectors_name = "{}_{}_vectors".format(name, self.eval_name)
            self.add_input(vectors_name, shape=(self.system_size, nx_actual, ny_actual, 3), units="m")
            self.vector_names.append(vectors_name)
            normals_name = "{}_normals".format(name)
            self.add_input(normals_name, shape=(nx - 1, ny - 1, 3))
            self.normal_names.append(normals_name)

        self.add_input("alpha", val=1.0, units="deg", tags=["mphys_input"])
        self.add_input("rhs", shape=self.system_size, units="m/s")
        self.add_output("circulations", shape=self.system_size, units="m**2/s", tags=["mphys_coupling"])

        self.aic_mtx = EvalVelMtx(self.options["surfaces"], self.eval_name)

    def apply_nonlinear(self, inputs, outputs, residuals):
        alpha = jnp.asarray(inputs["alpha"])
        circulations = jnp.asarray(outputs["circulations"])
        vectors = {vector_name: jnp.asarray(inputs[vector_name]) for vector_name in self.vector_names}
        normals = {normal_name: jnp.asarray(inputs[normal_name]) for normal_name in self.normal_names}
        residuals["circulations"] = self.aic_mtx.compute_residual(alpha, vectors, normals, circulations) - inputs["rhs"]

    def solve_nonlinear(self, inputs, outputs):
        alpha = jnp.asarray(inputs["alpha"])
        vectors = {vector_name: jnp.asarray(inputs[vector_name]) for vector_name in self.vector_names}
        normals = {normal_name: jnp.asarray(inputs[normal_name]) for normal_name in self.normal_names}

        mvp = lambda circulations: self.aic_mtx.compute_residual(alpha, vectors, normals, jnp.asarray(circulations))
        def vmp(psi_vec):
            circulations = jnp.zeros_like(psi_vec)
            _, _, _, d_circulations = self.aic_mtx.compute_residual_vjp(alpha, vectors, normals, circulations, jnp.asarray(psi_vec))
            return d_circulations

        self.A = LinearOperator((self.system_size, self.system_size), mvp, vmp)
        self.d = self.aic_mtx.get_diags(alpha, vectors, normals)
        outputs["circulations"], exit_code = gmres(self.A, inputs["rhs"], x0=outputs["circulations"], tol=1e-10, M=self.d)

    def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        alpha = jnp.asarray(inputs["alpha"])
        vectors = {vector_name: jnp.asarray(inputs[vector_name]) for vector_name in self.vector_names}
        normals = {normal_name: jnp.asarray(inputs[normal_name]) for normal_name in self.normal_names}
        circulations = jnp.asarray(outputs["circulations"])
        if mode == "fwd":
            pass

        if mode == "rev":
            if "circulations" in d_residuals:
                d_alpha, d_vectors, d_normals, d_circulations = self.aic_mtx.compute_residual_vjp(alpha, vectors, normals, circulations,
                                                                                                  jnp.asarray(d_residuals["circulations"]))
                if "circulations" in d_outputs:
                    d_outputs["circulations"] += d_circulations

                if "rhs" in d_inputs:
                    d_inputs["rhs"] -= d_residuals["circulations"]

                if "alpha" in d_inputs:
                    d_inputs["alpha"] += d_alpha

                for vec_name in d_vectors:
                    if vec_name in d_inputs:
                        d_inputs[vec_name] += d_vectors[vec_name]

                for normal_name in d_normals:
                    if normal_name in d_inputs:
                        d_inputs[normal_name] += d_normals[normal_name]

    def solve_linear(self, d_outputs, d_residuals, mode):
        if mode == "fwd":
            pass
        if mode == "rev":
            d_residuals["circulations"], exit_code = gmres(self.A.T, d_outputs["circulations"], x0=d_residuals["circulations"], tol=1e-10, M=self.d)

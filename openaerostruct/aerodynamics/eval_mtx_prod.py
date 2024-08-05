from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse import diags

import openmdao.api as om

from openaerostruct.utils.vector_algebra import add_ones_axis
from openaerostruct.utils.vector_algebra import compute_dot_jnp
from openaerostruct.utils.vector_algebra import compute_cross_jnp
from openaerostruct.utils.vector_algebra import compute_norm_jnp


tol = 1e-10


def _compute_finite_vortex(r1, r2):
    r1_norm = compute_norm_jnp(r1)
    r2_norm = compute_norm_jnp(r2)

    r1_x_r2 = compute_cross_jnp(r1, r2)
    r1_d_r2 = compute_dot_jnp(r1, r2)

    num = (1.0 / r1_norm + 1.0 / r2_norm) * r1_x_r2
    den = r1_norm * r2_norm + r1_d_r2

    result = jnp.where(jnp.abs(den) > tol, num/(den * 4 * jnp.pi), 0.0)

    return result


def _compute_semi_infinite_vortex(u, r):
    r_norm = compute_norm_jnp(r)
    u_x_r = compute_cross_jnp(u, r)
    u_d_r = compute_dot_jnp(u, r)

    num = u_x_r
    den = r_norm * (r_norm - u_d_r)
    return num / den / 4 / jnp.pi


class EvalVelMtx:
    """
    Computes the aerodynamic influence coefficient (AIC) matrix for the VLM
    analysis.

    This component is used in two places a given model, first to
    construct the AIC matrix using the collocation points as evaluation points,
    then to construct the AIC matrix where the force points are the evaluation
    points. The first matrix is used to solve for the circulations, while
    the second matrix is used to compute the forces acting on each panel.

    These calculations are rather complicated for a few reasons.
    Each surface interacts with every other surface, including itself.
    Also, in the general case, we have panel in both the spanwise and chordwise
    directions for all surfaces.
    Because of that, we need to compute the influence of each panel on every
    other panel, which results in rather large arrays for the
    intermediate calculations. Accordingly, the derivatives are complicated.

    The actual calcuations done here vary a fair bit in the case of symmetry.
    Not because the physics change, but because we need to account for a
    "ghost" version of the lifting surface, where we want to add the effects
    from the panels across the symmetry plane, but we don't want to actually
    use any of the evaluation points since we're not interested in the
    performance of this "ghost" version, since it's exactly symmetrical.
    This basically results in us looping through more calculations as if the
    panels were actually there.

    The calculations also vary when we consider ground effect.
    This is accomplished by mirroring a second copy of the mesh across
    the ground plane. The documentation has more detailed explanations.
    The ground effect is only implemented for symmetric wings.

    Parameters
    ----------
    alpha : float
        The angle of attack for the aircraft (all lifting surfaces) in degrees.
    vectors[num_eval_points, nx, ny, 3] : numpy array
        The vectors from the aerodynamic meshes to the evaluation points for
        every surface to every surface. For the symmetric case, the third
        dimension is length (2 * ny - 1). There is one of these arrays
        for each lifting surface in the problem.

    Returns
    -------
    vel_mtx[num_eval_points, nx - 1, ny - 1, 3] : numpy array
        The AIC matrix for the all lifting surfaces representing the aircraft.
        This has some sparsity pattern, but it is more dense than the FEM matrix
        and the entries have a wide range of magnitudes. One exists for each
        combination of surface name and evaluation points name.
    """

    def __init__(self, surfaces, eval_name):
        self.surfaces = surfaces
        self.eval_name = eval_name
        self.num_eval_points = 0
        for surface in surfaces:
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
            num_surf_panels = (ny - 1) * (nx - 1)
            surf_idx = jnp.arange(nx_actual * ny_actual).reshape(nx_actual, ny_actual)
            self.num_eval_points += num_surf_panels

    @partial(jax.jit, static_argnames=['self', 'jacobi'])
    def compute_velocity(self, alpha, vectors, circulations, jacobi=False):
        velocities = jnp.zeros((self.num_eval_points, 3))
        ind_1 = 0
        ind_2 = 0
        
        for surface in self.surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            ground_effect = surface.get("groundplane", False)

            num = (nx - 1) * (ny - 1)

            ind_2 += num

            if jnp.ndim(alpha) != 0:
                alpha = alpha.at[0].get()
            cosa = jnp.cos(alpha * jnp.pi / 180.0)
            sina = jnp.sin(alpha * jnp.pi / 180.0)

            surf_gamma = circulations[ind_1:ind_2].reshape(nx - 1, ny - 1)
            if surface["symmetry"]:
                u = jnp.einsum("ijk,l->ijkl", jnp.ones((self.num_eval_points, 1, 2 * (ny - 1))), jnp.array([cosa, 0, sina]))
                # If this is a right-hand symmetrical wing, we need to flip the "y" indexing
                right_wing = abs(surface["mesh"][0, 0, 1]) < abs(surface["mesh"][0, -1, 1])
                if right_wing:
                    surf_gamma = surf_gamma[:, ::-1]
            else:
                u = jnp.einsum("ijk,l->ijkl", jnp.ones((self.num_eval_points, 1, ny - 1)), jnp.array([cosa, 0, sina]))

            vectors_name = "{}_{}_vectors".format(name, self.eval_name)

            # Here, we loop through each of the vectors and compute the AIC
            # terms from the four filaments that make up a ring around a single
            # panel. Thus, we are using vortex rings to construct the AIC
            # matrix. Later, we will convert these to horseshoe vortices
            # to compute the panel forces.

            if ground_effect:
                # mirrored surface along the x mesh direction
                surfaces_to_compute = [vectors[vectors_name][:, :nx, :, :], vectors[vectors_name][:, nx:, :, :]]
                vortex_mults = [1.0, -1.0]
            else:
                surfaces_to_compute = [vectors[vectors_name]]
                vortex_mults = [1.0]

            for i_surf, surface_to_compute in enumerate(surfaces_to_compute):
                # vortex vertices:
                #         A ----- B
                #         |       |
                #         |       |
                #         D-------C
                #
                vortex_mult = vortex_mults[i_surf]
                vert_A = surface_to_compute[:, 0:-1, 1:, :]
                vert_B = surface_to_compute[:, 0:-1, 0:-1, :]
                vert_C = surface_to_compute[:, 1:, 0:-1, :]
                vert_D = surface_to_compute[:, 1:, 1:, :]
                # front vortex
                result1 = _compute_finite_vortex(vert_A, vert_B)
                # right vortex
                result2 = _compute_finite_vortex(vert_B, vert_C)
                # rear vortex
                result3 = _compute_finite_vortex(vert_C, vert_D)
                # left vortex
                result4 = _compute_finite_vortex(vert_D, vert_A)

                # If the surface is symmetric, mirror the results and add them
                # to the vel_mtx.
                result = vortex_mult * (result1 + result2 + result3 + result4)
                if not jacobi:
                    if surface["symmetry"]:
                        velocities = velocities.at[:,:].add(jnp.einsum("ijkl,jk->il", result[:, :, : ny - 1, :], surf_gamma[:, :]))
                        velocities = velocities.at[:,:].add(jnp.einsum("ijkl,jk->il", result[:, :, ny - 1 :, :], surf_gamma[:, ::-1]))
                    else:
                        velocities = velocities.at[:,:].add(jnp.einsum("ijkl,jk->il", result[:, :, :, :], surf_gamma[:, :]))
                else:
                    if surface["symmetry"]:
                        velocities = velocities.at[ind_1:ind_2, :].add(jnp.einsum("iil,i->il", result[ind_1:ind_2, :, : ny - 1, :].reshape(num, -1, 3), surf_gamma[:, :].flatten()))
                        velocities = velocities.at[ind_1:ind_2, :].add(jnp.einsum("iil,i->il", result[ind_1:ind_2, :, ny - 1 :, :].reshape(num, -1, 3), surf_gamma[:, ::-1].flatten()))
                    else:
                        velocities = velocities.at[ind_1:ind_2, :].add(jnp.einsum("iil,i->il", result[ind_1:ind_2, :, :, :].reshape(num, -1, 3), surf_gamma[:, :].flatten()))

                # ----------------- last row -----------------

                vert_D_last = vert_D[:, -1:, :, :]
                vert_C_last = vert_C[:, -1:, :, :]
                result1 = _compute_finite_vortex(vert_D_last, vert_C_last)
                result2 = _compute_semi_infinite_vortex(u, vert_D_last)
                result3 = _compute_semi_infinite_vortex(u, vert_C_last)

                if surface["symmetry"]:
                    res1 = result1[:, :, : ny - 1, :]
                    res1 = res1.at[:].add(result1[:, :, ny - 1 :, :][:, :, ::-1, :])
                    res2 = result2[:, :, : ny - 1, :]
                    res2 = res2.at[:].add(result2[:, :, ny - 1 :, :][:, :, ::-1, :])
                    res3 = result3[:, :, : ny - 1, :]
                    res3 = res3.at[:].add(result3[:, :, ny - 1 :, :][:, :, ::-1, :])
                    result1 = res1
                    result2 = res2
                    result3 = res3
                    
                if not jacobi:
                    velocities = velocities.at[:, :].add(vortex_mult * jnp.einsum("ijkl,jk->il", result1, surf_gamma[-1:, :]))
                    velocities = velocities.at[:, :].add(-vortex_mult * jnp.einsum("ijkl,jk->il", result2, surf_gamma[-1:, :]))
                    velocities = velocities.at[:, :].add(vortex_mult * jnp.einsum("ijkl,jk->il", result3, surf_gamma[-1:, :]))
                else:
                    result = result1 + result2 + result3
                    global_indices = jnp.arange(ind_1, ind_2, dtype=int).reshape(nx -1, ny -1)
                    last_global_indices = global_indices[-1, :]
                    velocities = velocities.at[last_global_indices, :].add(vortex_mult * jnp.einsum("iil,i->il", result[last_global_indices, :, :].reshape(ny-1, ny-1, 3), surf_gamma[-1:, :].flatten()))

            ind_1 += num
        return velocities

    @partial(jax.jit, static_argnames=['self'])
    def compute_residual(self, alpha, vectors, normals, circulations):
        velocity = self.compute_velocity(alpha, vectors, circulations)
        res = jnp.zeros([self.num_eval_points])
        surfaces = self.surfaces

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            normals_name = "{}_normals".format(name)

            res = res.at[ind_1:ind_2].add(jnp.einsum("ij,ij->i", velocity[ind_1:ind_2,:], normals[normals_name].reshape(-1, 3)))

            ind_1 += num

        return res

    @partial(jax.jit, static_argnums=(0,))
    def compute_velocity_vjp(self, alpha, vectors, circulations, d_output):
        # VJP is a good choice here since the jacbian matrix is a row vector.
        # We can compute the jacobian with a single call of the VJP function.

        # vjp always returns the primal and the vjp
        _, vjp_fun = jax.vjp(self.compute_velocity, alpha, vectors, circulations)

        return vjp_fun(d_output)

    @partial(jax.jit, static_argnums=(0,))
    def compute_residual_vjp(self, alpha, vectors, normals, circulations, d_residual):
        # VJP is a good choice here since the jacbian matrix is a row vector.
        # We can compute the jacobian with a single call of the VJP function.

        # vjp always returns the primal and the vjp
        _, vjp_fun = jax.vjp(self.compute_residual, alpha, vectors, normals, circulations)

        return vjp_fun(d_residual)

    def get_diags(self, alpha, vectors, normals):
        circulations = jnp.ones(self.num_eval_points)
        velocity = self.compute_velocity(alpha, vectors, circulations, jacobi=True)
        diag_vals = jnp.zeros([self.num_eval_points])
        surfaces = self.surfaces

        ind_1 = 0
        ind_2 = 0
        for surface in surfaces:
            nx = surface["mesh"].shape[0]
            ny = surface["mesh"].shape[1]
            name = surface["name"]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            normals_name = "{}_normals".format(name)

            diag_vals = diag_vals.at[ind_1:ind_2].add(jnp.einsum("ij,ij->i", velocity[ind_1:ind_2, :], normals[normals_name].reshape(-1, 3)))

            ind_1 += num

        return diags(1.0/np.array(diag_vals))

    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux_data = {'surfaces': self.surfaces, 'eval_name': self.eval_name, 'num_eval_points': self.num_eval_points, "options": self.options}  # static values
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

jax.tree_util.register_pytree_node(EvalVelMtx,
                               EvalVelMtx._tree_flatten,
                               EvalVelMtx._tree_unflatten)
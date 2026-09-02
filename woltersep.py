from math import floor
import gurobipy as gp
import numpy as np
from utils import (
    simple_bound,
    variable_bound,
    is_simple_bound,
    is_var_bound,
    get_var_bound,
    get_simple_bound,
    get_lb_star,
    get_ub_star,
    big_f,
    f_bar,
    is_integer,
)


class Wolter:
    def __init__(self, ip, frac_sol, duals, lp_slacks, eps=1e-4):
        self.ip = ip.copy()
        self.eps = eps
        self.solution = frac_sol
        self.b = self.ip.getAttr("rhs")
        self.vars = self.ip.getVars()
        self.constraints = self.ip.getConstrs()
        self.ncons = self.ip.NumConstrs
        self.senses = [c.Sense for c in self.constraints]
        self.types = [var.vtype for var in self.vars]
        self.cont_idx = [
            idx
            for idx, var in enumerate(self.vars)
            if var.vtype == gp.GRB.CONTINUOUS
        ]
        self.int_idx = [
            idx
            for idx, var in enumerate(self.vars)
            if idx not in self.cont_idx
        ]
        self.matrix = self.ip.getA().toarray()
        self.baseA = np.copy(self.matrix)

        self.validate()
        self.getbounds()
        self.standardize()

        self.fill_solution()

        self.row = np.zeros(self.newmatrix.shape[1] + 1)
        self.duals = duals
        self.slacks = lp_slacks
        self.uses = [0 for _ in range(self.newmatrix.shape[0])]
        self.scores = [0 for _ in range(self.newmatrix.shape[0])]
        self.base_scores = [0 for _ in range(self.newmatrix.shape[0])]

        self.create_scores(duals, lp_slacks)
        self.compute_scores()

    def create_scores(self, duals, lp_slacks):
        idx = 0
        obj = np.array([var.Obj for var in self.vars])
        obj_norm = max(np.linalg.norm(obj), 1)
        for i in range(self.ncons):
            if i in self.rows_to_delete:
                continue
            dens = np.linalg.norm(self.baseA[i, :], ord=0) / len(
                self.baseA[i, :]
            )
            dist = max(np.linalg.norm(self.baseA[i, :]), 0.1)
            dist = lp_slacks[i] / dist
            self.base_scores[idx] = max(duals[i] / obj_norm, 0.0001)
            self.base_scores[idx] += 0.0001 * (1 - dens)
            self.base_scores[idx] += 0.001 * (1 - dist)
            idx += 1

    def compute_scores(self):
        for idx in range(len(self.scores)):
            self.scores[idx] = (0.9 ** self.uses[idx]) * self.base_scores[idx]

    def validate(self):
        neg_vars = [var for var in self.vars if var.LB < 0]
        assert len(neg_vars) == 0, "Negative lower bounds are not supported"
        neg_vars = [var for var in self.vars if var.UB < 0]
        assert len(neg_vars) == 0, "Negative upper bounds are not supported"
        neg_vars = [
            var for var in self.vars if var.LB > 0 and var.VType != "C"
        ]
        if len(neg_vars) > 0:
            print(
                "Warning: some integer variables have positive lower bounds\n"
            )

    def getbounds(self):
        self.var_bound_constr = []
        self.simple_bounds_constr = []

        self.variable_up_bounds = {}
        self.variable_lo_bounds = {}
        self.simple_up_bounds = {}
        self.simple_lo_bounds = {}
        for idx in self.cont_idx:
            self.variable_up_bounds[idx] = []
            self.variable_lo_bounds[idx] = []
            self.simple_up_bounds[idx] = []
            self.simple_lo_bounds[idx] = []

        for idx in self.cont_idx:
            self.simple_lo_bounds[idx].append(
                simple_bound(idx, self.vars[idx].LB, "lower")
            )
        for idx in self.cont_idx:
            self.simple_up_bounds[idx].append(
                simple_bound(idx, self.vars[idx].UB, "upper")
            )

        for i in range(self.ncons):
            if is_var_bound(self.matrix[i], self.types, self.senses[i]):
                self.var_bound_constr.append(i)
                idx, var_bound = get_var_bound(
                    self.matrix[i], self.types, self.senses[i], self.b[i]
                )
                if var_bound.sense == "upper":
                    self.variable_up_bounds[idx].append(var_bound)
                else:
                    self.variable_lo_bounds[idx].append(var_bound)
            if is_simple_bound(self.matrix[i], self.senses[i], self.types):
                self.simple_bounds_constr.append(i)
                idx, this_simple_bound = get_simple_bound(
                    self.matrix[i], self.senses[i], self.b[i]
                )
                if this_simple_bound.sense == "upper":
                    self.simple_up_bounds[idx].append(this_simple_bound)
                else:
                    self.simple_lo_bounds[idx].append(this_simple_bound)

        self.rows_to_delete = self.var_bound_constr + self.simple_bounds_constr

    def standardize(self):
        for idx, constraint in enumerate(self.constraints):
            if constraint.sense == "<" and idx not in self.rows_to_delete:
                ones = [1 if i == idx else 0 for i in range(len(self.b))]
                self.matrix = np.c_[self.matrix, ones]
            elif constraint.sense == ">" and idx not in self.rows_to_delete:
                ones = [-1 if i == idx else 0 for i in range(len(self.b))]
                self.matrix = np.c_[self.matrix, ones]

        self.newmatrix = np.delete(self.matrix, self.rows_to_delete, axis=0)
        self.newb = np.delete(self.b, self.rows_to_delete, axis=0)
        self.all_cont_idx = list(range(self.newmatrix.shape[1]))
        self.all_cont_idx = [
            idx for idx in self.all_cont_idx if idx not in self.int_idx
        ]

    def fill_solution(self):
        self.slacks = [
            abs(np.dot(self.baseA[i, :], np.array(self.solution)) - self.b[i])
            for i in range(self.ncons)
        ]
        eq_idx = [
            idx for idx, c in enumerate(self.constraints) if c.Sense == "="
        ]
        slacks_to_delete = list(set(self.rows_to_delete + eq_idx))
        self.slacks = list(np.delete(self.slacks, slacks_to_delete, axis=0))
        self.slacks = [abs(s) for s in self.slacks]
        self.x_all = list(self.solution) + list(self.slacks)
        # assert (
            # len(self.x_all) == self.newmatrix.shape[1]
        # ), "Solution length mismatch"

    def bound_substitution(self):
        self.substitutions = set()
        self.sub_row = self.row.copy()
        self.sub_solution = self.x_all.copy()
        lb_star = {}
        ub_star = {}

        for j in self.cont_idx:
            lb_star[j], a_lbstar = get_lb_star(
                self.simple_lo_bounds[j],
                self.variable_lo_bounds[j],
                self.x_all,
            )
            ub_star[j], a_ubstar = get_ub_star(
                self.simple_up_bounds[j],
                self.variable_up_bounds[j],
                self.x_all,
            )

            u_dist = ub_star[j] - self.x_all[j]
            l_dist = self.x_all[j] - lb_star[j]

            if l_dist <= u_dist:
                self.substitutions.add(a_lbstar)
                if isinstance(a_lbstar, variable_bound):
                    self.sub_row[a_lbstar.int_var] += (
                        self.row[j] * a_lbstar.bound
                    )
                else:
                    self.sub_row[-1] += -self.row[j] * a_lbstar.bound
                self.sub_solution[j] = self.x_all[j] - lb_star[j]
            else:
                self.substitutions.add(a_ubstar)
                if isinstance(a_ubstar, variable_bound):
                    self.sub_row[j] = -self.sub_row[j]
                    self.sub_row[a_ubstar.int_var] += (
                        self.row[j] * a_ubstar.bound
                    )
                else:
                    self.sub_row[j] = -self.sub_row[j]
                    self.sub_row[-1] += -self.row[j] * a_ubstar.bound
                self.sub_solution[j] = ub_star[j] - self.x_all[j]

        self.s_star = 0
        for j in self.all_cont_idx:
            if self.sub_row[j] > 0:
                self.sub_row[j] = 0
            if self.sub_row[j] < 0:
                self.s_star += self.sub_row[j] * self.sub_solution[j]
        self.s_star = -self.s_star

    def cut_generation(self, eps=1e-4):
        big_u = [
            j for j in self.int_idx if self.x_all[j] >= (self.vars[j].UB / 2)
        ]
        big_t = [
            (j, self.x_all[j] - (self.vars[j].UB / 2))
            for j in self.int_idx
            if j not in big_u
        ]
        t = len(big_t)
        big_t = sorted(big_t, key=lambda x: x[1], reverse=True)
        n_star = [
            abs(self.sub_row[j])
            for j in self.int_idx
            if abs(self.sub_row[j]) > eps
            and eps < self.x_all[j] < (self.vars[j].UB - eps)
        ]
        extended_n_star = 1 + max([abs(self.sub_row[j]) for j in self.int_idx])
        n_star.append(extended_n_star)
        delta_found = False
        best_violation = -float("inf")
        L = 0

        local_big_t = big_t.copy()

        while not delta_found and L < t:

            if L >= 0 and abs(self.x_all[local_big_t[L][0]]) > self.eps:
                if self.vars[local_big_t[L][0]].UB < 1e99:
                    big_u.append(local_big_t[L][0])
                    big_t.remove(local_big_t[L])
            elif L > 0:
                L += 1
                continue

            for delta in n_star:
                beta = 0
                beta += self.sub_row[-1]
                for j in big_u:
                    beta -= self.sub_row[j] * self.vars[j].UB
                beta = beta / delta
                if is_integer(beta):
                    continue
                else:
                    delta_found = True
                    violation = 0
                    f_beta = beta - floor(beta)
                    for j, _ in big_t:
                        violation += (
                            big_f(f_beta, self.sub_row[j] / delta)
                            * self.x_all[j]
                        )
                    for j in big_u:
                        violation += big_f(
                            f_beta, -self.sub_row[j] / delta
                        ) * (self.vars[j].UB * self.x_all[j])
                    violation -= floor(beta)
                    violation -= self.s_star / (delta * (1 - f_beta))
                    if violation > best_violation:
                        best_violation = violation
                        best_delta = delta

            L += 1

        if not delta_found:
            return False

        delta_bar = best_delta
        for delta in [delta_bar / 2, delta_bar / 4, delta_bar / 8]:
            beta = 0
            beta += self.sub_row[-1]
            for j in big_u:
                beta -= self.sub_row[j] * self.vars[j].UB
            beta = beta / delta
            violation = 0
            f_beta = beta - floor(beta)
            for j, _ in big_t:
                violation += (
                    big_f(f_beta, self.sub_row[j] / delta) * self.x_all[j]
                )
            for j in big_u:
                violation += big_f(f_beta, -self.sub_row[j] / delta) * (
                    self.vars[j].UB * self.x_all[j]
                )
            violation -= floor(beta)
            violation -= self.s_star / (delta * (1 - f_beta))
            if violation > best_violation:
                best_violation = violation
                best_delta = delta

        delta = best_delta
        best_big_u = big_u.copy()
        best_big_t = big_t.copy()
        t = len(big_t)
        big_t = sorted(big_t, key=lambda x: x[1], reverse=True)

        local_big_t = big_t.copy()

        for L in range(t):
            if self.x_all[local_big_t[L][0]] > eps:
                if self.vars[local_big_t[L][0]].UB < 1e99:
                    big_u.append(local_big_t[L][0])
                    big_t.remove(local_big_t[L])
                beta = 0
                beta += self.sub_row[-1]
                for j in big_u:
                    beta -= self.sub_row[j] * self.vars[j].UB
                beta = beta / delta
                violation = 0
                f_beta = beta - floor(beta)
                for j, _ in big_t:
                    violation += (
                        big_f(f_beta, self.sub_row[j] / delta) * self.x_all[j]
                    )
                for j in big_u:
                    violation += big_f(f_beta, -self.sub_row[j] / delta) * (
                        self.vars[j].UB * self.x_all[j]
                    )
                violation -= floor(beta)
                violation -= self.s_star / (delta * (1 - f_beta))

                if violation > best_violation:
                    best_violation = violation
                    best_big_u = big_u.copy()
                    best_big_t = big_t.copy()

        big_u = best_big_u
        big_t = best_big_t

        beta = 0
        beta += self.sub_row[-1]
        for j in big_u:
            beta -= self.sub_row[j] * self.vars[j].UB
        beta = beta / delta
        f_beta = beta - floor(beta)

        self.generated_cut = [0 for _ in range(len(self.sub_row))]

        for j, _ in big_t:
            self.generated_cut[j] = big_f(f_beta, self.sub_row[j] / delta)
        for j in big_u:
            self.generated_cut[j] = -big_f(f_beta, -self.sub_row[j] / delta)
        for j in self.all_cont_idx:
            self.generated_cut[j] = f_bar(f_beta, self.sub_row[j] / delta)
        self.generated_cut[-1] = floor(beta)
        for j in big_u:
            self.generated_cut[-1] -= (
                big_f(f_beta, -self.sub_row[j] / delta) * self.vars[j].UB
            )

        return True

    def starting_row(self, best_idx):
        for j in range(self.newmatrix.shape[1]):
            self.row[j] = self.newmatrix[best_idx, j]
            self.row[-1] = self.newb[best_idx]
        # self.uses[best_idx] += 1
        # print(
        #     f"Starting with row {best_idx} and score {self.scores[best_idx]}"
        # )
        # print(f"Row: {self.row}\n")
        # print(self.newmatrix[best_idx, :])
        # print(self.newb[best_idx])

    def aggregation(self):
        lb_star = {}
        ub_star = {}
        d_j = {}

        for j in self.cont_idx:
            lb_star[j], a_lbstar = get_lb_star(
                self.simple_lo_bounds[j],
                self.variable_lo_bounds[j],
                self.x_all,
            )
            ub_star[j], a_ubstar = get_ub_star(
                self.simple_up_bounds[j],
                self.variable_up_bounds[j],
                self.x_all,
            )

            u_dist = ub_star[j] - self.x_all[j]
            l_dist = self.x_all[j] - lb_star[j]

            d_j[j] = min(u_dist, l_dist)

        big_m_star = [
            j
            for j in range(self.baseA.shape[1])
            if j in self.all_cont_idx and abs(self.row[j]) > self.eps
        ]
        if len(big_m_star) == 0:
            return False

        bestaggscore = -float("inf")
        bestdist = 0

        k = None
        for j in big_m_star:
            if d_j[j] < bestdist:
                continue
            candidate_const = [
                i
                for i in range(self.newmatrix.shape[0])
                if abs(self.newmatrix[i][j]) > self.eps and i not in self.big_q
            ]
            for i in candidate_const:
                agg_score = self.scores[i]
                if d_j[j] > bestdist or agg_score > bestaggscore:
                    bestaggscore = agg_score
                    bestdist = d_j[j]
                    k = j
                    r = i

        if k is None:
            return False

        w_r = -self.row[k] / self.newmatrix[r][k]
        for j in range(len(self.row) - 1):
            self.row[j] += w_r * self.newmatrix[r][j]
        self.row[-1] += w_r * self.newb[r]
        self.big_q.add(r)
        # self.uses[r] += 1

        return True

    def coumpute_violation(self):
        self.presub_row = self.generated_cut.copy()
        for sub in self.substitutions:
            if sub.sense == "lower":
                if isinstance(sub, variable_bound):
                    self.presub_row[sub.int_var] -= (
                        self.row[sub.real_var] * sub.bound
                    )
                else:
                    self.presub_row[-1] += self.row[sub.real_var] * sub.bound
            else:
                if isinstance(sub, variable_bound):
                    self.presub_row[sub.real_var] = -self.presub_row[
                        sub.real_var
                    ]
                    self.presub_row[sub.int_var] -= (
                        self.row[sub.real_var] * sub.bound
                    )
                else:
                    self.presub_row[sub.real_var] = -self.presub_row[
                        sub.real_var
                    ]
                    self.presub_row[-1] += self.row[sub.real_var] * sub.bound

        lhs = np.array(self.presub_row[:-1])
        rhs = self.presub_row[-1]
        lhs = np.dot(lhs, self.x_all)
        violation = lhs - rhs

        return violation > self.eps

    def generate_cuts(self, maxaggr=6):
        self.cuts = []
        indices = [
            i
            for i, v in sorted(
                enumerate(self.scores), key=lambda x: x[1], reverse=True
            )
        ]

        maxfails = 150
        fails = 0
        for i in range(len(self.scores)):
            if fails >= maxfails:
                return self.cuts
            best_idx = indices[i]
            self.big_q = set()
            self.big_q.add(best_idx)
            self.starting_row(best_idx)
            self.bound_substitution()
            if self.cut_generation() and self.coumpute_violation():
                fails = 0
                self.cuts.append(self.presub_row)
                for idx in self.big_q:
                    self.uses[idx] += 1
                self.compute_scores()
                continue
            else:
                fails += 1

            while len(self.big_q) < maxaggr:
                if not self.aggregation():
                    fails += 1
                    break
                self.bound_substitution()
                if self.cut_generation() and self.coumpute_violation():
                    fails = 0
                    self.cuts.append(self.presub_row)
                    for idx in self.big_q:
                        self.uses[idx] += 1
                    self.compute_scores()
                    break
                else:
                    fails += 1

        return self.cuts

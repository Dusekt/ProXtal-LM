"""
ProXtal-LM Validation — Crystal Structure Search
=================================================

Pure PyTorch batched Differential Evolution optimizer with dynamic
Matthews coefficient for finding the best space group / unit cell
given predicted crystogram contacts.
"""

import argparse
import numpy as np
import torch
import gemmi
from scipy.spatial.transform import Rotation
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@dataclass
class CrystalContact:
    residue_i: int
    residue_j: int
    crystogram_dist: float
    distogram_dist: float
    gap: float
    lattice_probability: float

def get_chiral_space_groups() -> List[gemmi.SpaceGroup]:
    chiral_sgs = []
    for i in range(1, 231):
        sg = gemmi.SpaceGroup(i)
        if all(np.linalg.det(np.array(op.rot).reshape(3,3) / 24.0) > 0 for op in sg.operations()):
            chiral_sgs.append(sg)
    return chiral_sgs

def get_crystal_system_params(sg_num: int):
    """ Returns: (num_free_lengths, [alpha, beta, gamma]) """
    if sg_num <= 2: return 2, [None, None, None]      # Triclinic
    if sg_num <= 14: return 2, [90.0, None, 90.0]     # Monoclinic
    if sg_num <= 74: return 2, [90.0, 90.0, 90.0]     # Orthorhombic
    if sg_num <= 142: return 1, [90.0, 90.0, 90.0]    # Tetragonal
    if sg_num <= 194: return 1, [90.0, 90.0, 120.0]   # Trigonal/Hexagonal
    return 0, [90.0, 90.0, 90.0]                      # Cubic


class PyTorchDifferentialEvolution:
    """ Batched Differential Evolution running entirely on PyTorch tensors. """
    def __init__(self, objective_func, bounds, popsize=15, maxiter=50, 
                 F=(0.5, 1.0), CR=0.7, strategy='best1bin', device='cuda', tol=0.01):
        self.objective_func = objective_func
        self.bounds = torch.tensor(bounds, dtype=torch.float32, device=device)
        self.popsize = popsize
        self.maxiter = maxiter
        self.F = F
        self.CR = CR
        self.strategy = strategy
        self.device = device
        self.tol = tol
        
        self.n_params = len(bounds)
        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.range = self.upper_bounds - self.lower_bounds
        
        normalized_pop = torch.rand((self.popsize, self.n_params), device=self.device)
        self.population = self.lower_bounds + normalized_pop * self.range
        
        self.fitness = self.objective_func(self.population)
        self.best_idx = torch.argmin(self.fitness)
        self.best_vector = self.population[self.best_idx].clone()
        self.best_score = self.fitness[self.best_idx].item()

    def _mutate(self):
        idx = torch.argsort(torch.rand((self.popsize, self.popsize), device=self.device), dim=-1)
        arange = torch.arange(self.popsize, device=self.device).unsqueeze(1)
        valid_idx = idx[idx != arange].view(self.popsize, self.popsize - 1)
        r1, r2, r3 = valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]
        
        if isinstance(self.F, tuple):
            F_batch = self.F[0] + torch.rand((self.popsize, 1), device=self.device) * (self.F[1] - self.F[0])
        else:
            F_batch = self.F

        if self.strategy == 'best1bin':
            mutant = self.best_vector + F_batch * (self.population[r1] - self.population[r2])
        else:
            mutant = self.population[r1] + F_batch * (self.population[r2] - self.population[r3])
        return mutant

    def optimize(self):
        for generation in range(self.maxiter):
            mutant = self._mutate()
            cross_points = torch.rand((self.popsize, self.n_params), device=self.device) <= self.CR
            force_cross = torch.randint(0, self.n_params, (self.popsize,), device=self.device)
            cross_points.scatter_(1, force_cross.unsqueeze(1), True)
            
            trial_population = torch.where(cross_points, mutant, self.population)
            trial_population = torch.max(torch.min(trial_population, self.upper_bounds), self.lower_bounds)
            
            trial_fitness = self.objective_func(trial_population)
            improved = trial_fitness < self.fitness
            
            self.population = torch.where(improved.unsqueeze(1), trial_population, self.population)
            self.fitness = torch.where(improved, trial_fitness, self.fitness)
            
            current_best_idx = torch.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.best_score:
                self.best_idx = current_best_idx
                self.best_vector = self.population[self.best_idx].clone()
                self.best_score = self.fitness[self.best_idx].item()
                
            if torch.std(self.fitness).item() < self.tol:
                break
                
        return type('OptimizeResult', (), {'x': self.best_vector.cpu().numpy(), 'fun': self.best_score})


def extract_crystal_contacts(crystogram, distogram, bin_edges, n_residues, max_crysto_dist=8.0, min_lattice_prob=0.25, min_dist_gap=3.0, min_seq_sep=20, threshold=8.0):
    print("\n[V13-Dynamic] Extracting crystal contacts using probability mass...")
    bins = bin_edges
    exp_crysto = np.sum(bins[None,None,:] * crystogram, axis=-1) / (crystogram.sum(axis=-1) + 1e-10)
    exp_disto = np.sum(bins[None,None,:] * distogram, axis=-1) / (distogram.sum(axis=-1) + 1e-10)
    gap_matrix = exp_disto - exp_crysto

    contact_bins = bins < threshold
    prob_crysto = np.sum(crystogram[:, :, contact_bins], axis=-1)
    prob_disto = np.sum(distogram[:, :, contact_bins], axis=-1)
    prob_lattice = np.clip(prob_crysto - prob_disto, 0, 1)

    n = crystogram.shape[0]
    seq_sep = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    prob_lattice[seq_sep < min_seq_sep] = 0
    np.fill_diagonal(prob_lattice, 0)
    prob_lattice = np.triu(prob_lattice, k=1)

    contacts = []
    for i in range(min(n, n_residues)):
        for j in range(i+1, min(n, n_residues)):
            if prob_lattice[i, j] > min_lattice_prob and gap_matrix[i, j] > min_dist_gap and exp_crysto[i, j] < max_crysto_dist:
                contacts.append(CrystalContact(i, j, exp_crysto[i,j], exp_disto[i,j], gap_matrix[i,j], prob_lattice[i,j]))

    contacts.sort(key=lambda x: x.lattice_probability * x.gap, reverse=True)
    return contacts[:]

class CrystalSearchDynamic:
    def __init__(self, crystogram_path, distogram_path, pdb_path, device='cuda', output_dir='positioned_structures'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.crystogram, self.distogram = np.load(crystogram_path), np.load(distogram_path)
        self.bin_edges = np.linspace(2.0, 22.0, 64)

        self.coords, self.mw, self.centroid, self.full_structure = self._load_protein(pdb_path)
        n_res = self.coords.shape[0]
        
        self.crystal_contacts = extract_crystal_contacts(self.crystogram, self.distogram, self.bin_edges, n_res)

        if self.crystal_contacts:
            self.contact_i = torch.tensor([c.residue_i for c in self.crystal_contacts], dtype=torch.long, device=self.device)
            self.contact_j = torch.tensor([c.residue_j for c in self.crystal_contacts], dtype=torch.long, device=self.device)
            self.target_dists = torch.tensor([c.crystogram_dist for c in self.crystal_contacts], dtype=torch.float32, device=self.device)
            self.opt_weights = torch.tensor([c.lattice_probability * c.gap for c in self.crystal_contacts], dtype=torch.float32, device=self.device)

        self.space_groups = get_chiral_space_groups()

    def _load_protein(self, pdb_path):
        st = gemmi.read_structure(pdb_path)
        st.remove_alternative_conformations()
        st.remove_hydrogens()
        st.remove_waters()
        coords, mw = [], 0.0
        for res in st[0][0]:
            cb, ca = res.find_atom('CB', '*'), res.find_atom('CA', '*')
            if cb: coords.append([cb.pos.x, cb.pos.y, cb.pos.z])
            elif ca: coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
            for atom in res: mw += atom.element.weight
        coords = torch.tensor(coords, dtype=torch.float32, device=self.device)
        centroid = coords.mean(dim=0).cpu().numpy()
        return coords - torch.tensor(centroid, device=self.device), mw, centroid, st

    def parse_symops(self, space_group):
        rots, trans = [], []
        for op in space_group.operations():
            rots.append((np.array(op.rot).reshape(3, 3) / 24.0).astype(np.float32))
            trans.append((np.array(op.tran) / 24.0).astype(np.float32))
        return (torch.tensor(np.stack(rots), dtype=torch.float32, device=self.device),
                torch.tensor(np.stack(trans), dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def batched_objective(self, params_batch, sym_rot, sym_tran, sg_num, Z):
        """ Evaluates the entire DE population simultaneously on the GPU """
        popsize = params_batch.shape[0]
        
        target_vms = params_batch[:, 0].cpu().numpy()
        euler_cpu = params_batch[:, 1:4].cpu().numpy()
        frac_pos = params_batch[:, 4:7]
        params_cpu = params_batch.cpu().numpy()
        
        n_ratios, fixed_angles = get_crystal_system_params(sg_num)
        
        # 1. Resolve Cell geometries per population member
        orth_mats, frac_mats = [], []
        for i in range(popsize):
            p_idx = 7
            ratios = [1.0, 1.0, 1.0]
            if n_ratios == 1: 
                ratios[2] = params_cpu[i, p_idx]; p_idx += 1
            elif n_ratios == 2: 
                ratios[1] = params_cpu[i, p_idx]; ratios[2] = params_cpu[i, p_idx+1]; p_idx += 2
                
            angles = []
            for fa in fixed_angles:
                if fa is None: angles.append(params_cpu[i, p_idx]); p_idx += 1
                else: angles.append(fa)
                
            temp_cell = gemmi.UnitCell(ratios[0], ratios[1], ratios[2], angles[0], angles[1], angles[2])
            target_volume = target_vms[i] * self.mw * Z
            scale = (target_volume / temp_cell.volume) ** (1/3)
            cell = gemmi.UnitCell(ratios[0]*scale, ratios[1]*scale, ratios[2]*scale, angles[0], angles[1], angles[2])
            
            orth_mats.append(cell.orth.mat.tolist())
            frac_mats.append(cell.frac.mat.tolist())
            
        orth_mat = torch.tensor(orth_mats, dtype=torch.float32, device=self.device) # (popsize, 3, 3)
        frac_mat = torch.tensor(frac_mats, dtype=torch.float32, device=self.device) # (popsize, 3, 3)
        
        # 2. Batched Rotations & Translations
        rot_batch = Rotation.from_euler('xyz', euler_cpu).as_matrix()
        rot = torch.tensor(rot_batch, dtype=torch.float32, device=self.device) # (popsize, 3, 3)
        
        # Apply Rotations
        rotated = torch.einsum('pij,nj->pni', rot, self.coords.float()) # (popsize, N, 3)
        frac_rotated = torch.einsum('pij,pnj->pni', frac_mat, rotated)
        frac_positioned = frac_rotated + frac_pos.unsqueeze(1)
        
        # Apply Symops
        frac_mates = []
        for idx in range(1, sym_rot.shape[0]):
            mate = torch.einsum('ij,pnj->pni', sym_rot[idx], frac_positioned) + sym_tran[idx].view(1, 1, 3)
            frac_mates.append(mate)
            
        # 3. Batched Scoring
        min_dist = torch.full((popsize,), float('inf'), device=self.device)
        clash_count = torch.zeros((popsize,), dtype=torch.float32, device=self.device)
        
        # Sterics checking
        for f_mate in frac_mates:
            diff_frac = frac_positioned.unsqueeze(2) - f_mate.unsqueeze(1) # (popsize, N, N, 3)
            diff_frac -= torch.round(diff_frac)
            cart_diff = torch.einsum('pabc,pcd->pabd', diff_frac, orth_mat)
            d_mat = torch.norm(cart_diff, dim=-1) # (popsize, N, N)
            
            min_dist = torch.minimum(min_dist, d_mat.view(popsize, -1).min(dim=1).values)
            clash_count += (d_mat < 2.5).sum(dim=(1, 2)).float()
            
        # Contact Scoring
        if not self.crystal_contacts:
            weighted_score = torch.zeros((popsize,), dtype=torch.float32, device=self.device)
        else:
            f_coords_i = frac_positioned[:, self.contact_i, :]
            f_coords_j = frac_positioned[:, self.contact_j, :]
            c_dists = torch.full((popsize, len(self.crystal_contacts)), float('inf'), device=self.device)
            
            # NEW: Track how many distinct mates are being used
            active_mates_count = torch.zeros((popsize,), dtype=torch.float32, device=self.device)
            margin = 0.20 * self.target_dists.unsqueeze(0)
            
            for f_mate in frac_mates:
                diff_ij = f_coords_i - f_mate[:, self.contact_j, :]
                diff_ij -= torch.round(diff_ij)
                d_ij = torch.norm(torch.einsum('pbc,pcd->pbd', diff_ij, orth_mat), dim=-1)
                
                diff_ji = f_coords_j - f_mate[:, self.contact_i, :]
                diff_ji -= torch.round(diff_ji)
                d_ji = torch.norm(torch.einsum('pbc,pcd->pbd', diff_ji, orth_mat), dim=-1)
                
                mate_min = torch.minimum(d_ij, d_ji)
                c_dists = torch.minimum(c_dists, mate_min)
                
                # If this mate is within margin (+ 0.5A slack) for ANY contact, count it as "active"
                is_active = (torch.abs(mate_min - self.target_dists.unsqueeze(0)) <= margin + 0.5).any(dim=1)
                active_mates_count += is_active.float()
                
            residuals = torch.abs(c_dists - self.target_dists.unsqueeze(0))
            active_residuals = torch.clamp(residuals - margin, min=0)
            weighted_score = (active_residuals ** 2 * self.opt_weights.unsqueeze(0)).sum(dim=1)
            
            # THE MATE TAX: Penalize heavily if using more than 4 distinct symmetry mates
            mate_tax = torch.clamp(active_mates_count - 4, min=0) * 10.0
            weighted_score += mate_tax
            
        # Combine Penalties
        steric_penalty = torch.zeros((popsize,), dtype=torch.float32, device=self.device)
        has_clash = clash_count > 0
        close_contact = min_dist < 3.5
        
        dist_penalty = 50.0 * ((3.5 - min_dist) ** 2)
        steric_penalty[has_clash] = clash_count[has_clash] * 50.0 + dist_penalty[has_clash]
        
        mask2 = ~has_clash & close_contact
        steric_penalty[mask2] = dist_penalty[mask2]
        
        return weighted_score + steric_penalty

    @torch.no_grad()
    def _single_check_and_score_mic(self, frac_positioned, frac_mates, orth_mat, track_assignments=False):
        """ Single-state evaluation purely for final PDB generation and assignment tracking """
        min_dist = float('inf')
        clash_count = 0
        for f_mate in frac_mates:
            diff_frac = frac_positioned[:, None, :] - f_mate[None, :, :]
            diff_frac -= torch.round(diff_frac)
            d_mat = torch.norm(torch.matmul(diff_frac, orth_mat), dim=-1)
            min_dist = min(min_dist, d_mat.min().item())
            clash_count += (d_mat < 2.5).sum().item()

        if not self.crystal_contacts: 
            return clash_count, min_dist, 0.0, {'pct_satisfied': 0.0, 'mean_dist': float('inf'), 'mse': float('inf'), 'satisfied': 0, 'assignments': []}
            
        f_coords_i = frac_positioned[self.contact_i]
        f_coords_j = frac_positioned[self.contact_j]
        c_dists = torch.full((len(self.crystal_contacts),), float('inf'), device=self.device)
        best_mate_idx = torch.zeros(len(self.crystal_contacts), dtype=torch.long, device=self.device)

        for sym_idx, f_mate in enumerate(frac_mates):
            diff_ij = f_coords_i - f_mate[self.contact_j]
            diff_ij -= torch.round(diff_ij)
            d_ij = torch.norm(torch.matmul(diff_ij, orth_mat), dim=-1)
            
            diff_ji = f_coords_j - f_mate[self.contact_i]
            diff_ji -= torch.round(diff_ji)
            d_ji = torch.norm(torch.matmul(diff_ji, orth_mat), dim=-1)
            
            mate_min = torch.minimum(d_ij, d_ji)
            mask = mate_min < c_dists
            c_dists[mask] = mate_min[mask]
            if track_assignments: best_mate_idx[mask] = sym_idx

        residuals = torch.abs(c_dists - self.target_dists)
        margin = 0.20 * self.target_dists
        satisfied = (residuals <= margin).sum().item()
        active_residuals = torch.clamp(residuals - margin, min=0)
        
        # Calculate unique mates used for the final assignment
        unique_mates_used = len(set(best_mate_idx.cpu().numpy().tolist()))
        mate_tax = max(0, unique_mates_used - 4) * 10.0
        
        mse = (residuals ** 2).mean().item()
        weighted_score = (active_residuals ** 2 * self.opt_weights).sum().item() + mate_tax
        
        details = {
            'mean_dist': c_dists.mean().item(), 'satisfied': satisfied,
            'pct_satisfied': 100.0 * satisfied / len(self.crystal_contacts), 
            'residual': weighted_score, 'mse': mse,
            'unique_mates': unique_mates_used
        }
        if track_assignments: details['assignments'] = best_mate_idx.cpu().numpy().tolist()
        return clash_count, min_dist, weighted_score, details

    def search_space_group(self, space_group, maxiter, popsize):
        sym_rot, sym_tran = self.parse_symops(space_group)
        if sym_rot.shape[0] <= 1: return None

        Z = len(list(space_group.operations()))
        n_ratios, fixed_angles = get_crystal_system_params(space_group.number)
        
        bounds = [(2.0, 3)]            # Matthews Coeff (Vm)
        bounds += [(0, 2*np.pi)] * 3     # Euler XYZ
        bounds += [(0.0, 1.0)] * 3       # Fractional Position
        bounds += [(0.3, 3.0)] * n_ratios # Cell aspect ratios
        for fa in fixed_angles:
            if fa is None: bounds.append((60.0, 120.0))

        optimizer = PyTorchDifferentialEvolution(
            objective_func=lambda params: self.batched_objective(params, sym_rot, sym_tran, space_group.number, Z),
            bounds=bounds, strategy='best1bin', maxiter=maxiter, popsize=popsize, device=self.device
        )

        result = optimizer.optimize()

        params = result.x
        target_vm, euler, frac_pos = params[0], params[1:4], params[4:7]
        p_idx = 7
        ratios = [1.0, 1.0, 1.0]
        if n_ratios == 1: ratios[2] = params[p_idx]; p_idx += 1
        elif n_ratios == 2: ratios[1] = params[p_idx]; ratios[2] = params[p_idx+1]; p_idx += 2
            
        angles = []
        for fa in fixed_angles:
            if fa is None: angles.append(params[p_idx]); p_idx += 1
            else: angles.append(fa)

        temp_cell = gemmi.UnitCell(ratios[0], ratios[1], ratios[2], angles[0], angles[1], angles[2])
        target_volume = target_vm * self.mw * Z
        scale = (target_volume / temp_cell.volume) ** (1/3)
        cell = gemmi.UnitCell(ratios[0]*scale, ratios[1]*scale, ratios[2]*scale, angles[0], angles[1], angles[2])
        
        orth_mat = torch.tensor(cell.orth.mat.tolist(), dtype=torch.float32, device=self.device)
        frac_mat = torch.tensor(cell.frac.mat.tolist(), dtype=torch.float32, device=self.device)
        
        rot = torch.tensor(Rotation.from_euler('xyz', euler).as_matrix(), dtype=torch.float32, device=self.device)
        rotated = torch.matmul(self.coords.float(), rot.T)
        frac_rotated = torch.matmul(rotated, frac_mat.T)
        frac_positioned = frac_rotated + torch.tensor(frac_pos, dtype=torch.float32, device=self.device)
        
        frac_mates = []
        for idx in range(1, sym_rot.shape[0]):
            frac_mates.append(torch.matmul(frac_positioned, sym_rot[idx].T) + sym_tran[idx])

        clash_count, min_geom, _, details = self._single_check_and_score_mic(frac_positioned, frac_mates, orth_mat, track_assignments=True)
        
        st = self.full_structure.clone()
        rot_np = rot.cpu().numpy()
        trans_orth_np = torch.matmul(torch.tensor(frac_pos, dtype=torch.float32, device=self.device), orth_mat).cpu().numpy()
        
        for model in st:
            for chain in model:
                for res in chain:
                    for atom in res:
                        c_vec = np.array([atom.pos.x - self.centroid[0], atom.pos.y - self.centroid[1], atom.pos.z - self.centroid[2]])
                        n_vec = rot_np @ c_vec + trans_orth_np
                        atom.pos = gemmi.Position(n_vec[0], n_vec[1], n_vec[2])

        st.cell = cell
        st.spacegroup_hm = space_group.hm
        safe_sg = space_group.hm.replace(' ', '_')
        pdb_path = self.output_dir / f"positioned_{safe_sg}.pdb"
        st.write_pdb(str(pdb_path))

        assignments_out = [{'res_i': self.crystal_contacts[idx].residue_i, 'res_j': self.crystal_contacts[idx].residue_j, 'mate': mate_idx + 1} 
                           for idx, mate_idx in enumerate(details.get('assignments', []))]

        return {
            'space_group': space_group.hm, 'z_value': Z, 'pdb_path': str(pdb_path),
            'score': details['pct_satisfied'], 'weighted_residual': result.fun, 'mse': details.get('mse', float('inf')),
            'mean_contact_dist': details['mean_dist'], 'geometric_score': min_geom,
            'matthews_coeff': float(target_vm),
            'cell': [cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
            'contact_assignments': assignments_out
        }

    def run_full_search(self, maxiter=50, popsize=15, out="v13_dynamic.json"):
        print("\n" + "="*70)
        print("CRYSTAL SEARCH V13 - PURE PYTORCH DE BATCHING + DYNAMIC VM")
        print("="*70)
        results = {}
        for i, sg in enumerate(self.space_groups):
            print(f"[{i+1}/{len(self.space_groups)}] Evaluating: {sg.hm}")
            config = self.search_space_group(sg, maxiter, popsize)
            if config: results[sg.hm] = config

        # Changed to sort primarily by the minimized loss function (weighted residual), then by raw MSE (ascending)
        sorted_res = sorted(results.items(), key=lambda x: (x[1]['weighted_residual'], x[1]['mse']))
        
        print(f"\n{'Rank':<5} {'Space Group':<14} {'Z':<3} {'Loss':<8} {'MSE':<7} {'%Sat':<6} {'Geom':<6} {'Vm':<5}")
        print("-"*70)
        for rank, (sg_hm, cfg) in enumerate(sorted_res[:20], 1):
            print(f"{rank:<5} {sg_hm:<14} {cfg['z_value']:<3} {cfg['weighted_residual']:<8.2f} {cfg['mse']:<7.2f} {cfg['score']:<6.1f} {cfg['geometric_score']:<6.1f} {cfg['matthews_coeff']:<5.2f}")

        c_list = []
        for rank, (sg_hm, cfg) in enumerate(sorted_res, 1):
            cfg['rank'], cfg['id'] = rank, sg_hm.replace(' ', '_')
            c_list.append(cfg)

        with open(out, 'w') as f:
            json.dump({'candidates': c_list}, f, indent=2, cls=NumpyEncoder)
        print(f"Saved to {out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crystogram", required=True)
    parser.add_argument("--distogram", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--de_maxiter", type=int, default=50)
    parser.add_argument("--de_popsize", type=int, default=15)
    parser.add_argument("--output", default="v13_dynamic.json")
    args = parser.parse_args()

    engine = CrystalSearchDynamic(args.crystogram, args.distogram, args.pdb)
    engine.run_full_search(args.de_maxiter, args.de_popsize, args.output)

if __name__ == "__main__":
    main()
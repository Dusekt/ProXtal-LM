"""
ProXtal-LM Validation — MLIP Sidechain Resolver
================================================

Phase 2 of validation: final relaxation using MACE-OFF MLIP calculator.

Takes Stage 1 crystal search outputs (already at Vm~2.5) and performs
a 2-phase relaxation:
1. Fixed Cell (Untangle Sidechains)
2. Variable Cell (Final Box Optimization)
"""

import json
import numpy as np
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import gemmi

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
    from ase.optimize import FIRE
    from ase.constraints import FixSymmetry
    from ase.geometry import cell_to_cellpar
    from ase.calculators.calculator import Calculator
    try:
        from ase.filters import FrechetCellFilter
        CELL_FILTER_CLASS = FrechetCellFilter
    except ImportError:
        from ase.filters import ExpCellFilter
        CELL_FILTER_CLASS = ExpCellFilter
    ASE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: ASE not available: {e}"); sys.exit(1)

try:
    from mace.calculators import mace_off
    MACE_AVAILABLE = True
except ImportError:
    print("ERROR: MACE not available. pip install mace-torch"); sys.exit(1)


class SurvivalStatus(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    SURVIVED = "survived"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class CandidateResult:
    candidate_id: str
    space_group: str
    z_value: int
    pdb_path: str
    status: SurvivalStatus = SurvivalStatus.FAILED
    
    initial_max_force: float = float('inf')
    initial_capped_pct: float = 100.0
    
    mid_max_force: float = float('inf')  # After Phase 1
    
    final_max_force: float = float('inf') # After Phase 2
    final_capped_pct: float = 100.0
    final_energy: float = float('inf')
    
    detanglement_delta: float = 0.0
    total_runtime_seconds: float = 0.0

    final_volume: float = 0.0
    final_cell_dims: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    final_angles: Tuple[float, float, float] = (90.0, 90.0, 90.0)
    error_message: Optional[str] = None

class ForceCappedCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    def __init__(self, base_calc: Calculator, max_force: float = 20.0):
        super().__init__()
        self.base_calc = base_calc
        self.max_force = max_force
        self._stats = {'max_original': 0.0, 'n_capped': 0, 'capped_pct': 0.0}

    def calculate(self, atoms: Atoms, properties: List[str], system_changes: List):
        self.base_calc.calculate(atoms, properties, system_changes)
        self.results['energy'] = self.base_calc.results.get('energy', 0.0)
        if 'stress' in self.base_calc.results:
            self.results['stress'] = self.base_calc.results['stress'].copy()

        forces = self.base_calc.results['forces'].copy()
        original_norms = np.linalg.norm(forces, axis=1)

        self._stats['max_original'] = float(np.max(original_norms))
        self._stats['n_capped'] = int(np.sum(original_norms > self.max_force))
        self._stats['capped_pct'] = 100.0 * self._stats['n_capped'] / len(forces)

        for i in range(len(forces)):
            norm = original_norms[i]
            if norm > self.max_force:
                forces[i] = forces[i] * (self.max_force / norm)

        self.results['forces'] = forces

    def get_stats(self) -> Dict:
        return self._stats.copy()

def get_crystal_system(sg: str) -> str:
    sg = sg.strip().upper().replace(' ', '').replace('-', '')
    if any(p in sg for p in ['F23','I23','P23','F432','I432','P432','FM3M','IM3M','PM3M','PA3','IA3']): return 'cubic'
    if 'P6' in sg or sg.startswith('H'): return 'hexagonal'
    if 'P3' in sg or 'R3' in sg: return 'trigonal'
    if 'P4' in sg or 'I4' in sg: return 'tetragonal'
    if '222' in sg or '221' in sg or sg in ['P212121', 'P21212', 'C2221', 'C222']: return 'orthorhombic'
    if 'P2' in sg or 'C2' in sg or 'P121' in sg: return 'monoclinic'
    return 'orthorhombic' # Fallback safely locks angles

def get_strain_mask(crystal_system: str) -> List[bool]:
    masks = {
        'triclinic': [True, True, True, True, True, True],
        'monoclinic': [True, True, True, False, True, False],
        'orthorhombic': [True, True, True, False, False, False],
        'tetragonal': [True, False, True, False, False, False],
        'trigonal': [True, False, True, False, False, False],
        'hexagonal': [True, False, True, False, False, False],
        'cubic': [True, False, False, False, False, False],
    }
    return masks.get(crystal_system, [True, True, True, False, False, False])

import gemmi
from ase.io import read as ase_read

def get_chain_id(n):
    """Converts an integer (1, 2, 3...) to a PDB chain ID (A, B... Z, AA, AB...)"""
    result = ""
    while n > 0:
        n -= 1
        result = chr(65 + (n % 26)) + result
        n //= 26
    return result

def build_crystal_and_squeeze(pdb_path, sg_name, target_vm=2.5):
    """Squeezes the cell to biological density and mathematically clones the symmetry mates."""
    
    # 1. Auto-Calculate exact Molecular Weight
    temp_atoms = ase_read(pdb_path)
    exact_mw = sum(temp_atoms.get_masses())
    print(f"  Auto-detected exact Molecular Weight: {exact_mw:.1f} Da")
    
    st = gemmi.read_structure(pdb_path)
    sg = gemmi.SpaceGroup(sg_name)
    cell = st.cell
    
    # 2. The Squeeze
    current_vol = cell.volume
    z_value = len(list(sg.operations()))
    target_vol = target_vm * exact_mw * z_value
    scale = (target_vol / current_vol) ** (1/3)
    
    new_cell = gemmi.UnitCell(cell.a * scale, cell.b * scale, cell.c * scale, cell.alpha, cell.beta, cell.gamma)
    
    # 3. The Cloning Machine (Fixed for whole-chain integrity)
    new_st = gemmi.Structure()
    new_model = gemmi.Model("1") 
    
    chain_counter = 0
    for op in sg.operations():
        for original_chain in st[0]: 
            new_chain = original_chain.clone()
            new_chain.name = get_chain_id(chain_counter + 1)
            chain_counter += 1
            
            # --- THE FIX: Corrected gemmi indexing ---
            # original_chain[0] is the first residue, [0][0] is the first atom
            anchor_pos = original_chain[0][0].pos
            anchor_frac = cell.fractionalize(anchor_pos)
            anchor_sym_frac = op.apply_to_xyz(anchor_frac.tolist())
            
            # Find out how many periodic boxes away the anchor ended up
            # (using integer floor division)
            shift_x = int(anchor_sym_frac[0] // 1)
            shift_y = int(anchor_sym_frac[1] // 1)
            shift_z = int(anchor_sym_frac[2] // 1)
            
            for res in new_chain:
                for atom in res:
                    frac = cell.fractionalize(atom.pos)
                    sym_frac = op.apply_to_xyz(frac.tolist())
                    
                    # Shift the whole molecule back by the same amount!
                    # This keeps all bonds perfectly intact.
                    sym_frac[0] -= shift_x
                    sym_frac[1] -= shift_y
                    sym_frac[2] -= shift_z
                    
                    new_pos = new_cell.orthogonalize(gemmi.Fractional(*sym_frac))
                    atom.pos = gemmi.Position(*new_pos)
            # ---------------------------------------------------------------------
            
            new_model.add_chain(new_chain)
            
    # NOW we add the fully-loaded model to the structure
    new_st.add_model(new_model)
    new_st.cell = new_cell
    new_st.spacegroup_hm = sg_name
            
    full_cell_path = pdb_path.replace(".pdb", "_fullcell.pdb")
    new_st.write_pdb(full_cell_path)
    return full_cell_path

class MLIPResolver:
    def __init__(self, mace_model="medium", device="cuda", max_force_cap=20.0, fire_steps=150, fire_maxstep=0.05):
        self.device = device
        self.max_force_cap = max_force_cap
        self.fire_steps = fire_steps
        self.fire_maxstep = fire_maxstep
        print(f"Loading MACE-OFF-{mace_model.upper()} on {device}...")
        self.mace_calc = mace_off(model=mace_model, device=device, default_dtype="float32", batch_size=16)

    def run_candidate(self, candidate: Dict) -> CandidateResult:
        start_time = time.time()
        cand_id = candidate.get('id', candidate.get('space_group', 'unknown'))
        sg = candidate['space_group']
        z = candidate.get('z_value', candidate.get('z', 1)) # FIXED KEY ERROR
        pdb_path = candidate['pdb_path']

        result = CandidateResult(candidate_id=cand_id, space_group=sg, z_value=z, pdb_path=pdb_path)
        print(f"\n[{cand_id}] Starting 2-Phase Relaxation...")

        try:
            # Build the full, squeezed unit cell! (Adjust mw to your protein's exact mass if known)
            full_cell_pdb = build_crystal_and_squeeze(pdb_path, sg, target_vm=2.5)
            
            # Now ASE loads the FULL crystal, not just a monomer
            atoms = ase_read(full_cell_pdb)
            atoms.pbc = True
            crystal_sys = get_crystal_system(sg)
            strain_mask = get_strain_mask(crystal_sys)
            try:
                atoms.set_constraint(FixSymmetry(atoms, symprec=0.01))
            except: pass

            # Attach Calculator
            atoms.calc = ForceCappedCalculator(self.mace_calc, max_force=self.max_force_cap)
            
            # Record initial state
            init_stats = atoms.calc.get_stats() if hasattr(atoms.calc, 'get_stats') else {}
            forces = atoms.get_forces()
            result.initial_max_force = float(np.max(np.linalg.norm(forces, axis=1))) if len(forces) > 0 else 0.0
            result.initial_capped_pct = init_stats.get('capped_pct', 100.0)
            print(f"  Init -> Max Force: {result.initial_max_force:.1e} eV/A | Capped: {result.initial_capped_pct:.1f}%")

            # ========================================================
            # PHASE 1: Fixed Cell (Untangle Sidechains)
            # ========================================================
            print("  Phase 1: Fixed Cell Optimization...")
            opt1 = FIRE(atoms, maxstep=self.fire_maxstep, logfile=None)
            opt1.run(fmax=0.5, steps=self.fire_steps)
            
            mid_stats = atoms.calc.get_stats()
            result.mid_max_force = mid_stats.get('max_original', float('inf'))
            print(f"  Phase 1 Complete -> Max Force: {result.mid_max_force:.1e} eV/A")

            # ========================================================
            # PHASE 2: Variable Cell (Final Box Optimization)
            # ========================================================
            print("  Phase 2: Variable Cell Optimization...")
            ecf = CELL_FILTER_CLASS(atoms, mask=strain_mask)
            opt2 = FIRE(ecf, maxstep=self.fire_maxstep, logfile=None)
            opt2.run(fmax=0.05, steps=self.fire_steps)
            
            final_stats = atoms.calc.get_stats()
            result.final_max_force = final_stats.get('max_original', float('inf'))
            result.final_capped_pct = final_stats.get('capped_pct', 0.0)
            result.final_energy = atoms.get_potential_energy() / len(atoms) # Energy per atom
            
            # ========================================================
            # Save Relaxed Structure (Using Gemmi to preserve metadata!)
            # ========================================================
            out_pdb = pdb_path.replace(".pdb", "_relaxed.pdb")
            
            # 1. Load the pristine full cell we made earlier
            final_st = gemmi.read_structure(full_cell_pdb)
            
            # 2. Update the Unit Cell with the new relaxed dimensions
            new_cell_params = atoms.get_cell().cellpar() # [a, b, c, alpha, beta, gamma]
            final_st.cell = gemmi.UnitCell(*new_cell_params)
            
            # 3. Inject the relaxed ASE coordinates back into the Gemmi structure
            relaxed_positions = atoms.get_positions()
            atom_idx = 0
            for model in final_st:
                for chain in model:
                    for res in chain:
                        for atom in res:
                            # Update to the newly relaxed [x, y, z] Cartesian coordinates
                            atom.pos = gemmi.Position(*relaxed_positions[atom_idx])
                            atom_idx += 1
                            
            # 4. Save the perfect PDB
            final_st.write_pdb(out_pdb)
            print(f"  Phase 2 Complete -> Relaxed PDB saved to {out_pdb}")

            # Capture Dimensions
            cell = atoms.get_cell()
            result.final_volume = atoms.get_volume()
            result.final_cell_dims = tuple(np.linalg.norm(cell, axis=1))
            result.final_angles = tuple(cell_to_cellpar(cell)[3:])
            result.detanglement_delta = result.initial_max_force - result.final_max_force
            
            if result.final_capped_pct == 0 and result.final_max_force < 5.0: result.status = SurvivalStatus.EXCELLENT
            elif result.final_capped_pct < 10: result.status = SurvivalStatus.GOOD
            else: result.status = SurvivalStatus.SURVIVED

        except Exception as e:
            result.error_message = str(e)
            result.status = SurvivalStatus.FAILED
            print(f"  Failed: {e}")

        result.total_runtime_seconds = time.time() - start_time
        return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--candidates", required=True)
    parser.add_argument("-o", "--output", default="v6_relaxed_scoreboard.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_force", type=float, default=20.0)
    args = parser.parse_args()

    with open(args.candidates) as f:
        data = json.load(f)
    candidates = data.get('candidates', data.get('ranking', data))

    resolver = MLIPResolver(device=args.device, max_force_cap=args.max_force)
    
    results = []
    for cand in candidates[:15]: # Process top 15 candidates
        results.append(resolver.run_candidate(cand))

    # Rank by Energy per Atom! (Lower is more stable)
    results.sort(key=lambda x: x.final_energy if x.status != SurvivalStatus.FAILED else float('inf'))

    out_data = []
    print("\n" + "="*80)
    print(f"{'Rank':<5} {'Space Group':<14} {'Status':<10} {'Energy/Atom':<12} {'Max Force':<10}")
    print("-" * 80)
    for rank, r in enumerate(results, 1):
        print(f"{rank:<5} {r.space_group:<14} {r.status.value:<10} {r.final_energy:<12.3f} {r.final_max_force:<10.1f}")
        out_data.append(r.__dict__)

    with open(args.output, 'w') as f:
        # Simple string enum serialization
        for item in out_data: item['status'] = item['status'].value
        json.dump(out_data, f, indent=2)

if __name__ == "__main__":
    main()
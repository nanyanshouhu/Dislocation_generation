import numpy as np
import re
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure

#######################################
# 1. Elastic Constants and Derived Properties
#######################################

def read_elastic_constants(filename):
    """
    Reads the elastic constant matrix from a file.
    The file can either contain a 6x6 matrix or key-value pairs (e.g., "C11 = value").
    If a full 6x6 matrix is provided, it is assumed to be in Voigt notation.
    For the full 21-component case, it expects the following keys:
      'C11','C12','C13','C14','C15','C16',
      'C22','C23','C24','C25','C26',
      'C33','C34','C35','C36',
      'C44','C45','C46',
      'C55','C56',
      'C66'
    """
    constants = {}
    try:
        # Attempt to read as a 6x6 matrix (9 independent values)
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                row = [float(p) for p in parts]
                data.append(row)
        data = np.array(data)
        if data.shape == (6, 6):
            constants["C11"] = data[0, 0]
            constants["C22"] = data[1, 1]
            constants["C33"] = data[2, 2]
            constants["C12"] = data[0, 1]  # assuming symmetry
            constants["C13"] = data[0, 2]
            constants["C23"] = data[1, 2]
            constants["C44"] = data[3, 3]
            constants["C55"] = data[4, 4]
            constants["C66"] = data[5, 5]
            # For a 6x6 matrix, extra components are set to zero.
            extra_keys = ["C14", "C15", "C16", "C24", "C25", "C26",
                          "C34", "C35", "C36", "C45", "C46", "C56"]
            for key in extra_keys:
                constants[key] = 0.0
            print("Elastic constants read as a 6x6 matrix (9 independent values).")
            return constants
        elif data.size == 21:
            # Assume file contains 21 independent values in a single row or column.
            keys = ["C11", "C12", "C13", "C14", "C15", "C16",
                    "C22", "C23", "C24", "C25", "C26",
                    "C33", "C34", "C35", "C36",
                    "C44", "C45", "C46",
                    "C55", "C56",
                    "C66"]
            data = data.flatten()
            for key, val in zip(keys, data):
                constants[key] = float(val)
            print("Elastic constants read as 21 independent components.")
            return constants
        else:
            raise ValueError("File format for elastic constants is not recognized.")
    except Exception as e:
        raise RuntimeError("Error reading elastic constants: " + str(e))
    
    # Fallback: key-value reading
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(r'(C\d+)\s*=\s*([\d\.Ee+-]+)', line)
            if match:
                key = match.group(1)
                value = float(match.group(2))
                constants[key] = value
    return constants

def calculate_poisson_direct(crystal_system, Cij):
    """
    Calculates the Poisson ratio based on the crystal system.
    For cubic: nu = C12 / (C11 + C12).
    For hexagonal, returns two Poisson ratios: nu_12 and nu_13.
    For other systems, the effective Poisson ratio is computed via the Hill average.
    Returns a dictionary.
    """
    system = crystal_system.lower()
    if system == "cubic":
        try:
            nu = Cij["C12"] / (Cij["C11"] + Cij["C12"])
            return {"nu": nu}
        except KeyError:
            raise ValueError("Missing C11 or C12 elastic constants for cubic system.")
    elif system == "hexagonal":
        try:
            nu_12 = Cij["C12"] / (Cij["C11"] + Cij["C12"])
            nu_13 = Cij["C13"] / (Cij["C33"] + Cij["C13"])
            return {"nu_12": nu_12, "nu_13": nu_13}
        except KeyError:
            raise ValueError("For a hexagonal system, C11, C12, C13, and C33 must be provided.")
    else:
        nu_eff = calculate_effective_poisson_ratio(crystal_system, Cij)
        return {"nu": nu_eff}

def calculate_K_G(crystal_system, Cij):
    """
    Calculates the bulk (K_V) and shear (G_V) moduli.
    For cubic:
         K_V = (C11 + 2 * C12) / 3
         G_V = (C11 - C12 + 3 * C44) / 5
    For hexagonal:
         K_V = (2 * C11 + C33 + 2 * C12 + 4 * C13) / 9
         G_V = (C11 - C12 + C33 - C13 + 3 * C44) / 5
    For other systems (using Voigt averages):
         K_V = (C11 + C22 + C33 + 2*(C12 + C13 + C23)) / 9
         G_V = (C11 + C22 + C33 - (C12 + C13 + C23) + 3*(C44 + C55 + C66)) / 15
    """
    system = crystal_system.lower()
    if system == "cubic":
        try:
            K_V = (Cij["C11"] + 2 * Cij["C12"]) / 3.0
            G_V = (Cij["C11"] - Cij["C12"] + 3 * Cij["C44"]) / 5.0
            return K_V, G_V
        except KeyError:
            raise ValueError("For cubic systems, C11, C12, and C44 must be provided.")
    elif system == "hexagonal":
        try:
            K_V = (2 * Cij["C11"] + Cij["C33"] + 2 * Cij["C12"] + 4 * Cij["C13"]) / 9.0
            G_V = (Cij["C11"] - Cij["C12"] + Cij["C33"] - Cij["C13"] + 3 * Cij["C44"]) / 5.0
            return K_V, G_V
        except KeyError:
            raise ValueError("For hexagonal systems, C11, C12, C13, C33, and C44 must be provided.")
    else:
        try:
            # Use get() with defaults if keys are not available.
            K_V = (Cij["C11"] + Cij.get("C22", Cij["C11"]) + Cij.get("C33", Cij["C11"]) +
                   2*(Cij["C12"] + Cij.get("C13", Cij["C12"]) + Cij.get("C23", Cij["C12"]))) / 9.0
            G_V = (Cij["C11"] + Cij.get("C22", Cij["C11"]) + Cij.get("C33", Cij["C11"]) -
                   (Cij["C12"] + Cij.get("C13", Cij["C12"]) + Cij.get("C23", Cij["C12"])) +
                   3*(Cij["C44"] + Cij.get("C55", Cij["C44"]) + Cij.get("C66", Cij["C44"]))) / 15.0
            return K_V, G_V
        except KeyError:
            raise ValueError("Generic formulas require keys: C11, and at least C12 and C44.")

def calculate_effective_poisson_ratio(crystal_system, Cij):
    """
    Calculates an effective (Hill average) Poisson ratio:
         nu_eff = (3*K - 2*G) / (2*(3*K + G))
    """
    K, G = calculate_K_G(crystal_system, Cij)
    nu_eff = (3 * K - 2 * G) / (2 * (3 * K + G))
    return nu_eff

def determine_elastic_mode(Cij, crystal_system, tol=0.1):
    """
    Determines whether to use an isotropic or anisotropic Green's function correction.
    For cubic systems, calculates the Zener anisotropy factor:
         A = 2 * C44 / (C11 - C12)
    If |A - 1| < tol, the material is nearly isotropic;
    otherwise, it defaults to "anisotropic".
    """
    system = crystal_system.lower()
    if system == "cubic":
        try:
            C11 = Cij["C11"]
            C12 = Cij["C12"]
            C44 = Cij["C44"]
        except KeyError:
            raise ValueError("Cij must contain C11, C12, and C44 for a cubic system.")
        A = 2 * C44 / (C11 - C12)
        print(f"Computed Zener anisotropy factor A = {A:.3f}")
        if abs(A - 1) < tol:
            return "isotropic"
        else:
            return "anisotropic"
    else:
        print(f"Non-cubic system ({crystal_system}) detected. Defaulting to anisotropic Green's function.")
        return "anisotropic"

#######################################
# 2. Displacement Field Functions for Dislocation Types
#######################################

def displacement_field_edge(dx, dy, b, nu):
    """
    Returns the displacement field (ux, uy, 0) for an edge dislocation
    at point (dx, dy) with Burgers vector magnitude b and Poisson ratio nu.
    """
    r2 = dx**2 + dy**2
    if r2 < 1e-12:
        r2 = 1e-12
    ux = (b / (2 * np.pi)) * (np.arctan2(dy, dx) + (dx * dy) / (2 * (1 - nu) * r2))
    uy = -(b / (2 * np.pi)) * (((1 - 2 * nu) / (4 * (1 - nu))) * np.log(r2) + (dx**2 - dy**2) / (4 * (1 - nu) * r2))
    return np.array([ux, uy, 0.0])

def displacement_field_screw(dx, dy, b):
    """
    Returns the displacement field (0, 0, uz) for a screw dislocation,
    where uz = (b / (2π)) * theta.
    """
    theta = np.arctan2(dy, dx)
    uz = (b / (2 * np.pi)) * theta
    return np.array([0.0, 0.0, uz])

def displacement_field_mixed(dx, dy, b, nu, mix_angle):
    """
    Returns the displacement field for a mixed dislocation.
    The mix_angle (in degrees) sets the weight of the edge vs. screw component.
    """
    rad = np.deg2rad(mix_angle)
    w_edge = np.cos(rad)
    w_screw = np.sin(rad)
    disp_edge = displacement_field_edge(dx, dy, b, nu)
    disp_screw = displacement_field_screw(dx, dy, b)
    return w_edge * disp_edge + w_screw * disp_screw

#######################################
# 2.2. Lattice Green's Function Correction
#######################################

def lattice_green_function(r, mu, nu):
    """
    Computes the lattice Green's function using an isotropic approximation.
    G_ij(r) = 1 / (16π * μ * (1 - ν)) * [((3 - 4ν) δ_ij) / r + (r_i r_j) / r^3]
    """
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        r_norm = 1e-12
    identity = np.eye(3)
    rr = np.outer(r, r)
    G = (1.0 / (16 * np.pi * mu * (1 - nu))) * ((3 - 4 * nu) * identity / r_norm + rr / (r_norm**3))
    return G

def compute_green_function_correction(r, mu, nu, mode="isotropic", Cij=None):
    """
    Computes the Green's function correction matrix for displacement.
    
    Parameters:
       r      : displacement vector
       mu, nu : shear modulus and Poisson ratio for the isotropic case
       mode   : "isotropic" uses the analytical formula; "anisotropic" is a placeholder.
       Cij    : elastic constants (for the anisotropic method)
    
    Returns:
       G : a 3x3 Green's function correction matrix.
    """
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        r_norm = 1e-12
    if mode == "isotropic":
        identity = np.eye(3)
        rr = np.outer(r, r)
        G = (1.0 / (16 * np.pi * mu * (1 - nu))) * ((3 - 4 * nu) * identity / r_norm + rr / (r_norm**3))
    elif mode == "anisotropic":
        G = np.zeros((3, 3))  # Placeholder: anisotropic method not implemented
    else:
        raise ValueError("Unknown mode for Green's function computation")
    return G

def periodic_green_correction(pos, core, b, nu, mu, Lx, Ly, nmax=1,
                              F_effective=1.0, mode="isotropic", Cij=None):
    """
    Computes the correction displacement for an atom at position 'pos' due to periodic images
    of the dislocation core using the lattice Green's function method.
    """
    correction = np.zeros(3)
    F = F_effective * np.array([1.0, 0.0, 0.0])  # Effective force along the x-direction.
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            if i == 0 and j == 0:
                continue
            image_core = core + np.array([i * Lx, j * Ly, 0])
            r = pos - image_core
            G = compute_green_function_correction(r, mu, nu, mode=mode, Cij=Cij)
            correction += np.dot(G, F)
    return correction

#######################################
# 3. Rotation and Dislocation Dipole Structure Generation
#######################################

def rotate_vector(vec, R):
    """
    Rotates the vector 'vec' using the rotation matrix R.
    """
    return np.dot(R, vec)

def generate_rotation_matrix(burgers_direction, slip_plane_normal):
    """
    Generates a rotation matrix that maps the standard coordinate system to one defined by the
    given Burgers vector direction and slip plane normal.
    The first column aligns with the Burgers vector; the third with the slip plane normal.
    """
    e1 = burgers_direction / np.linalg.norm(burgers_direction)
    e3 = slip_plane_normal / np.linalg.norm(slip_plane_normal)
    e2 = np.cross(e3, e1)
    e2 = e2 / np.linalg.norm(e2)
    return np.array([e1, e2, e3]).T

#######################################
# Schmid Factor and Slip System Selection
#######################################

def compute_schmid_factor(load_dir, slip_direction, slip_plane_normal):
    """
    Computes the Schmid factor for a candidate slip system.
    load_dir: external load unit vector (assumed uniaxial)
    slip_direction: unit vector along the slip direction
    slip_plane_normal: unit vector normal to the slip plane
    Schmid factor m = |(load_dir · slip_direction)| * |(load_dir · slip_plane_normal)|
    """
    load_dir = load_dir / np.linalg.norm(load_dir)
    slip_direction = slip_direction / np.linalg.norm(slip_direction)
    slip_plane_normal = slip_plane_normal / np.linalg.norm(slip_plane_normal)
    m = abs(np.dot(load_dir, slip_direction)) * abs(np.dot(load_dir, slip_plane_normal))
    return m

def auto_select_slip_system(structure, load_dir=np.array([0, 0, 1]), cubic_subtype="fcc", override_crystal_system=None):
    """
    Automatically selects the slip system based on the crystal system and Schmid factor.
    
    Parameters:
      structure: pymatgen Structure object (original POSCAR structure is used)
      load_dir: external load direction (default [0, 0, 1])
      cubic_subtype: For cubic systems, specify "fcc" or "bcc"
      override_crystal_system: if provided, use it directly.
    
    Returns:
      (burgers_direction, slip_plane_normal) from the candidate slip system with the highest Schmid factor.
    """
    # Use the overridden crystal system if provided; otherwise, set as "unknown"
    crystal_system = override_crystal_system if override_crystal_system is not None else "unknown"
    
    # Define candidate slip systems. For non-cubic systems, we use a default candidate.
    if crystal_system.lower() == "cubic" or crystal_system.lower() == "unknown":
        if cubic_subtype.lower() == "fcc":
            candidates = [
                {"slip_plane": np.array([1, 1, 1]), "slip_direction": np.array([1, -1, 0])},
                {"slip_plane": np.array([1, 1, 1]), "slip_direction": np.array([-1, 0, 1])},
                {"slip_plane": np.array([1, 1, 1]), "slip_direction": np.array([0, 1, -1])}
            ]
        elif cubic_subtype.lower() == "bcc":
            candidates = [
                {"slip_plane": np.array([1, 1, 0]), "slip_direction": np.array([1, 1, 1])},
                {"slip_plane": np.array([1, 1, 0]), "slip_direction": np.array([-1, 1, 1])},
                {"slip_plane": np.array([1, 1, 0]), "slip_direction": np.array([1, -1, 1])}
            ]
        else:
            candidates = [
                {"slip_plane": np.array([1, 1, 1]), "slip_direction": np.array([1, -1, 0])}
            ]
    else:
        candidates = [
            {"slip_plane": np.array([0, 0, 1]), "slip_direction": np.array([1, 0, 0])}
        ]
    
    best_m = -1.0
    best_candidate = None
    for candidate in candidates:
        slip_plane_normal = candidate["slip_plane"] / np.linalg.norm(candidate["slip_plane"])
        burgers_direction = candidate["slip_direction"] / np.linalg.norm(candidate["slip_direction"])
        m = compute_schmid_factor(load_dir, burgers_direction, slip_plane_normal)
        print(f"Candidate slip system: plane normal = {slip_plane_normal}, slip direction = {burgers_direction}, Schmid factor = {m:.4f}")
        if m > best_m:
            best_m = m
            best_candidate = {"burgers_direction": burgers_direction, "slip_plane_normal": slip_plane_normal}
    print(f"Automatically selected slip system for {crystal_system.capitalize()} system (subtype {cubic_subtype.upper() if crystal_system.lower()=='cubic' else ''}) with Schmid factor = {best_m:.4f}:")
    print(f"  Slip plane normal: {best_candidate['slip_plane_normal']}")
    print(f"  Slip direction (Burgers direction): {best_candidate['burgers_direction']}")
    return best_candidate["burgers_direction"], best_candidate["slip_plane_normal"]

#######################################
# Identification of Crystal System from Elastic Constants
#######################################

def identify_crystal_system_from_elastic_constants(Cij, tol=0.1):
    """
    Heuristically identifies the likely crystal system based on 21 independent elastic constants.
    
    Expected relationships (approximately, within the given tolerance):
    
      Cubic:
        C11 ≈ C22 ≈ C33,
        C12 ≈ C13 ≈ C23,
        C44 ≈ C55 ≈ C66,
        and all extra coefficients (C14, C15, ..., C56) ≈ 0.
      
      Hexagonal:
        C11 ≈ C22, C13 ≈ C23, and ideally C66 ≈ (C11 - C12)/2, with negligible extra coefficients.
      
      Tetragonal:
        C11 ≈ C22, extra coefficients negligible, but C66 deviates from (C11 - C12)/2.
      
      Orthorhombic:
        Extra coefficients negligible but the diagonal constants differ.
      
      Monoclinic:
        A few extra coefficients are nonzero (e.g., 4 or fewer).
      
      Triclinic:
        Many extra coefficients are nonzero.
    
    Returns one of: "cubic", "hexagonal", "tetragonal", "orthorhombic", "monoclinic", or "triclinic".
    """
    def rel_diff(a, b):
        return abs(a - b) / abs(a) if abs(a) > 1e-8 else abs(b)
    
    try:
        C11 = Cij["C11"]
        C22 = Cij["C22"]
        C33 = Cij["C33"]
        C12 = Cij["C12"]
        C13 = Cij["C13"]
        C23 = Cij["C23"]
        C44 = Cij["C44"]
        C55 = Cij["C55"]
        C66 = Cij["C66"]
    except KeyError as e:
        raise ValueError("Missing key in elastic constants: " + str(e))
    
    threshold = 1e-3 * abs(C11) if abs(C11) > 1e-8 else 1e-3
    extra_keys = ["C14", "C15", "C16", "C24", "C25", "C26",
                  "C34", "C35", "C36", "C45", "C46", "C56"]
    nonzero_extra = [key for key in extra_keys if abs(Cij.get(key, 0.0)) > threshold]
    nonzero_count = len(nonzero_extra)
    
    # Cubic check
    if (rel_diff(C11, C22) < tol and rel_diff(C11, C33) < tol and rel_diff(C22, C33) < tol and
        rel_diff(C12, C13) < tol and rel_diff(C12, C23) < tol and rel_diff(C13, C23) < tol and
        rel_diff(C44, C55) < tol and rel_diff(C44, C66) < tol and rel_diff(C55, C66) < tol and
        nonzero_count == 0):
        return "cubic"
    
    # Hexagonal check
    if (rel_diff(C11, C22) < tol and rel_diff(C13, C23) < tol):
        ideal_C66 = (C11 - C12) / 2.0
        if rel_diff(C66, ideal_C66) < tol and nonzero_count == 0:
            return "hexagonal"
    
    # Tetragonal check
    if rel_diff(C11, C22) < tol and nonzero_count == 0:
        ideal_C66 = (C11 - C12) / 2.0
        if rel_diff(C66, ideal_C66) >= tol:
            return "tetragonal"
    
    # Orthorhombic check (if extra coefficients are negligible but diagonal elements differ)
    if nonzero_count == 0:
        if rel_diff(C11, C22) >= tol or rel_diff(C11, C33) >= tol or rel_diff(C22, C33) >= tol:
            return "orthorhombic"
    
    # Monoclinic: if only a few extra coefficients are nonzero (4 or fewer)
    if nonzero_count <= 4:
        return "monoclinic"
    
    # Otherwise, classify as triclinic
    return "triclinic"

#######################################
# 4. Dislocation Dipole Structure Generation
#######################################

def generate_dislocation_dipole_structure(structure, b, nu, burgers_dir, slip_plane_normal,
                                           dislocation_type="edge", mix_angle=0, dipole_offset=10.0,
                                           nmax=1, F_effective=1.0, mu=None, mode="isotropic",
                                           Cij=None, crystal_system="cubic"):
    """
    Generates a dislocation dipole structure under periodic boundary conditions,
    including a lattice Green's function correction.
    
    Parameters:
      structure         : pymatgen Structure object
      b                 : Burgers vector magnitude
      nu                : Poisson ratio (effective, if needed)
      burgers_dir       : Burgers vector direction (unit vector)
      slip_plane_normal : Slip plane normal (unit vector)
      dislocation_type  : "edge", "screw", or "mixed"
      mix_angle         : Mixing angle (degrees) for mixed dislocations
      dipole_offset     : Distance between the two dislocation cores
      nmax              : Periodic image summation range
      F_effective       : Effective force magnitude for Green's function correction
      mu                : Shear modulus; if None, computed from Cij and crystal_system
      mode              : "isotropic" or "anisotropic" for the Green's function method
      Cij               : Elastic constant dictionary (if needed)
      crystal_system    : e.g., "cubic", "hexagonal", etc.
    
    Returns:
      A new pymatgen Structure object with the dislocation dipole applied.
    """
    lattice = structure.lattice
    cart_coords = structure.cart_coords.copy()
    
    xs = cart_coords[:, 0]
    ys = cart_coords[:, 1]
    xc = (xs.max() + xs.min()) / 2.0
    yc = (ys.max() + ys.min()) / 2.0
    
    core1 = np.array([xc + dipole_offset / 2.0, yc, 0.0])
    core2 = np.array([xc - dipole_offset / 2.0, yc, 0.0])
    
    Lx, Ly, Lz = lattice.abc
    
    # If shear modulus is not provided, compute it from the elastic constants.
    if mu is None:
        _, G = calculate_K_G(crystal_system, read_elastic_constants("Cij.out"))
        mu = G
    
    R = generate_rotation_matrix(burgers_dir, slip_plane_normal)
    
    new_coords = []
    for pos in cart_coords:
        dx1 = pos[0] - core1[0]
        dy1 = pos[1] - core1[1]
        dx2 = pos[0] - core2[0]
        dy2 = pos[1] - core2[1]
        
        if dislocation_type.lower() == "edge":
            disp1 = displacement_field_edge(dx1, dy1, b, nu)
            disp2 = -displacement_field_edge(dx2, dy2, b, nu)
        elif dislocation_type.lower() == "screw":
            disp1 = displacement_field_screw(dx1, dy1, b)
            disp2 = -displacement_field_screw(dx2, dy2, b)
        elif dislocation_type.lower() == "mixed":
            disp1 = displacement_field_mixed(dx1, dy1, b, nu, mix_angle)
            disp2 = -displacement_field_mixed(dx2, dy2, b, nu, mix_angle)
        else:
            raise ValueError("Unknown dislocation type. Choose 'edge', 'screw', or 'mixed'.")
        
        disp_total = disp1 + disp2
        
        # Add periodic Green's function correction from both cores.
        disp_corr_1 = periodic_green_correction(pos, core1, b, nu, mu, Lx, Ly, nmax, F_effective, mode, Cij)
        disp_corr_2 = periodic_green_correction(pos, core2, -b, nu, mu, Lx, Ly, nmax, F_effective, mode, Cij)
        disp_total += (disp_corr_1 + disp_corr_2)
        
        disp_rotated = rotate_vector(disp_total, R)
        new_coords.append(pos + disp_rotated)
    
    new_structure = Structure(lattice, structure.species, new_coords, coords_are_cartesian=True)
    return new_structure

#######################################
# 5. Main Program
#######################################

if __name__ == "__main__":
    # 1. Read the original POSCAR structure (do not standardize)
    try:
        poscar = Poscar.from_file("POSCAR")
        structure = poscar.structure
    except Exception as e:
        raise RuntimeError("Error reading the POSCAR file: " + str(e))
    
    # 2. Read elastic constants and identify the crystal system from them.
    cij_file = "Cij.out"
    try:
        Cij = read_elastic_constants(cij_file)
        print("Read elastic constants:")
        for key, value in Cij.items():
            print(f"  {key} = {value}")
        crystal_system = identify_crystal_system_from_elastic_constants(Cij, tol=0.1)
        print(f"Identified crystal system from elastic constants: {crystal_system.upper()}")
        if crystal_system in ["cubic", "hexagonal"]:
            poisson_direct = calculate_poisson_direct(crystal_system, Cij)
            print("Directly computed Poisson ratio(s):")
            for k, v in poisson_direct.items():
                print(f"  {k} = {v:.4f}")
        nu_eff = calculate_effective_poisson_ratio(crystal_system, Cij)
        print(f"Effective Poisson ratio (Hill average) for {crystal_system.capitalize()} system = {nu_eff:.4f}")
        nu = nu_eff
    except Exception as e:
        raise RuntimeError("Error reading or calculating elastic constants: " + str(e))
    
    # 3. Automatically select the slip system using the Schmid factor.
    try:
        load_direction = np.array([0, 0, 1])
        burgers_dir, slip_plane_normal = auto_select_slip_system(structure, load_dir=load_direction,
                                                                  override_crystal_system=crystal_system)
    except Exception as e:
        raise RuntimeError("Automatic slip system selection failed: " + str(e))
    
    # 4. Determine the elastic correction mode.
    mode = determine_elastic_mode(Cij, crystal_system, tol=0.1)
    print(f"Using {mode} model for Green's function correction.")
    
    # 5. Set dislocation parameters.
    b = 2.5  # Burgers vector magnitude (Å)
    Lx, Ly, Lz = structure.lattice.abc
    dislocation_type = "mixed"  # Options: "edge (mix_angle =0)", "screw (mix_angle =90)", or "mixed (mix_angle =0~90)"
    mix_angle = 45              # Mixing angle for mixed dislocations (degrees)
    dipole_offset = Lx * 0.2    # Adjust based on cell size.
    F_effective = 1.0
    nmax = 1  # Periodic image summation range
    
    # 6. Generate the dislocation dipole structure with Green's function correction.
    disloc_structure = generate_dislocation_dipole_structure(
        structure, b, nu, burgers_dir, slip_plane_normal,
        dislocation_type, mix_angle, dipole_offset, nmax, F_effective,
        mu=None, mode=mode, Cij=Cij, crystal_system=crystal_system
    )
    
    # 7. Write out the new POSCAR file.
    new_poscar = Poscar(disloc_structure)
    new_poscar.write_file("POSCAR_dislocation")
    print("Dislocation dipole structure generated: POSCAR_dislocation has been saved.")

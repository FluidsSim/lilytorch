import numpy as np

def compute_capsule_inertia(radius, cylinder_length, mass):
    """
    Compute moments of inertia for a capsule (cylinder with hemispherical end caps).

    Parameters:
    -----------
    radius : float
        Radius of the capsule (m)
    cylinder_length : float
        Length of the cylindrical portion (m)
    mass : float
        Total mass of the capsule (kg)

    Returns:
    --------
    dict : Dictionary containing Ixx, Iyy, Izz
        Ixx, Iyy: Moments of inertia about perpendicular axes (transverse)
        Izz: Moment of inertia about axis along capsule length (axial)
    """
    r = radius
    h = cylinder_length
    m = mass

    # Total length including hemispherical caps
    L = h + 2 * r

    # Moment of inertia about axial axis (along capsule length)
    I_axial = 0.5 * m * r**2

    # Moment of inertia about perpendicular axes (transverse)
    I_transverse = m * (r**2 / 4 + h**2 / 12 + 2 * r * h / 5)

    return {
        'Ixx': I_transverse,
        'Iyy': I_transverse,
        'Izz': I_axial,
        'total_length': L
    }


def print_inertia_tensor(inertia_dict, name="Capsule"):
    """Print formatted inertia tensor information."""
    print(f"\n{name} Inertia Properties:")
    print(f"  Total Length: {inertia_dict['total_length']:.6f} m")
    print(f"  Ixx: {inertia_dict['Ixx']:.6e} kg⋅m²")
    print(f"  Iyy: {inertia_dict['Iyy']:.6e} kg⋅m²")
    print(f"  Izz: {inertia_dict['Izz']:.6e} kg⋅m²")


def main():
    """Compute inertias for salamander leg capsules."""

    # Leg capsule parameters from salamander.sdf
    radius = 0.001  # m
    cylinder_length = 0.006  # m
    # Compute mass from volume assuming uniform density
    density = 900  # kg/m³

    # Volume of capsule = volume of cylinder + volume of two hemispheres (= one sphere)
    volume_cylinder = np.pi * radius**2 * cylinder_length
    volume_sphere = (4/3) * np.pi * radius**3
    total_volume = volume_cylinder + volume_sphere

    mass = density * total_volume  # kg

    print("="*60)
    print("SALAMANDER LEG CAPSULE INERTIA COMPUTATION")
    print("="*60)

    print(f"\nInput Parameters:")
    print(f"  Radius: {radius} m")
    print(f"  Cylinder Length: {cylinder_length} m")
    print(f"  Mass: {mass:.6e} kg")

    # Compute inertias
    inertia = compute_capsule_inertia(radius, cylinder_length, mass)
    print_inertia_tensor(inertia, "Leg Capsule")

    # Compare with values from SDF file
    print("\n" + "="*60)
    print("COMPARISON WITH SDF FILE VALUES")
    print("="*60)

    sdf_ixx = 4.642575810304919e-11
    sdf_iyy = 4.642575810304919e-11
    sdf_izz = 9.94837673636768e-12

    print(f"\nSDF File Values:")
    print(f"  Ixx: {sdf_ixx:.6e} kg⋅m²")
    print(f"  Iyy: {sdf_iyy:.6e} kg⋅m²")
    print(f"  Izz: {sdf_izz:.6e} kg⋅m²")

    print(f"\nDifferences:")
    print(f"  ΔIxx: {abs(inertia['Ixx'] - sdf_ixx):.6e} kg⋅m² ({100*abs(inertia['Ixx'] - sdf_ixx)/sdf_ixx:.2f}%)")
    print(f"  ΔIyy: {abs(inertia['Iyy'] - sdf_iyy):.6e} kg⋅m² ({100*abs(inertia['Iyy'] - sdf_iyy)/sdf_iyy:.2f}%)")
    print(f"  ΔIzz: {abs(inertia['Izz'] - sdf_izz):.6e} kg⋅m² ({100*abs(inertia['Izz'] - sdf_izz)/sdf_izz:.2f}%)")

    # Test with different capsule configurations
    print("\n" + "="*60)
    print("OTHER CAPSULE CONFIGURATIONS")
    print("="*60)

    # Example: leg_1_R_2 with length 0.007
    radius_2 = 0.001
    cylinder_length_2 = 0.007
    mass_2 = 1.989675347273536e-05

    inertia_2 = compute_capsule_inertia(radius_2, cylinder_length_2, mass_2)
    print(f"\nCapsule with length {cylinder_length_2} m:")
    print_inertia_tensor(inertia_2, "Alternative Config")


if __name__ == "__main__":
    main()

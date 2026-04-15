/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Deposition implementation:
    - Dynamically add atoms at z = z_fraction * box_length (default 0.75)
    - Sets velocity with vz < 0 (depositing downward), vx = vy = 0
    - Random xy positions within the box
    - Use type index (0, 1, 2...) directly, like RDF does
    
Usage: deposition <interval> <type1> <num1> [<type2> <num2> ...]
Example: deposition 1000 0 5 1 10
------------------------------------------------------------------------------*/

#include "deposition.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <cstring>
#include <random>
#include <map>

static const std::map<std::string, double> MASS_TABLE{
  {"H", 1.0080000000},   {"He", 4.0026020000}, {"Li", 6.9400000000},  {"Be", 9.0121831000},
  {"B", 10.8100000000},  {"C", 12.0110000000}, {"N", 14.0070000000},  {"O", 15.9990000000},
  {"F", 18.9984031630},  {"Ne", 20.1797000000},{"Na", 22.9897692800}, {"Mg", 24.3050000000},
  {"Al", 26.9815385000}, {"Si", 28.0850000000}, {"P", 30.9737619980},  {"S", 32.0600000000},
  {"Cl", 35.4500000000}, {"Ar", 39.9480000000}, {"K", 39.0983000000},  {"Ca", 40.0780000000},
  {"Sc", 44.9559080000}, {"Ti", 47.8670000000}, {"V", 50.9415000000},  {"Cr", 51.9961000000},
  {"Mn", 54.9380440000}, {"Fe", 55.8450000000}, {"Co", 58.9331940000}, {"Ni", 58.6934000000},
  {"Cu", 63.5460000000}, {"Zn", 65.3800000000}, {"Ga", 69.7230000000}, {"Ge", 72.6300000000},
  {"As", 74.9215950000}, {"Se", 78.9710000000}, {"Br", 79.9040000000}, {"Kr", 83.7980000000},
  {"Rb", 85.4678000000}, {"Sr", 87.6200000000}, {"Y", 88.9058400000},  {"Zr", 91.2240000000},
  {"Nb", 92.9063700000}, {"Mo", 95.9500000000}, {"Tc", 98},            {"Ru", 101.0700000000},
  {"Rh", 102.9055000000},{"Pd", 106.4200000000},{"Ag", 107.8682000000},{"Cd", 112.4140000000},
  {"In", 114.8180000000},{"Sn", 118.7100000000},{"Sb", 121.7600000000},{"Te", 127.6000000000},
  {"I", 126.9044700000}, {"Xe", 131.2930000000},{"Cs", 132.9054519600},{"Ba", 137.3270000000},
  {"La", 138.9054700000},{"Ce", 140.1160000000},{"Pr", 140.9076600000},{"Nd", 144.2420000000},
  {"Pm", 145},           {"Sm", 150.3600000000},{"Eu", 151.9640000000},{"Gd", 157.2500000000},
  {"Tb", 158.9253500000},{"Dy", 162.5000000000},{"Ho", 164.9303300000},{"Er", 167.2590000000},
  {"Tm", 168.9342200000},{"Yb", 173.0450000000},{"Lu", 174.9668000000},{"Hf", 178.4900000000},
  {"Ta", 180.9478800000},{"W", 183.8400000000}, {"Re", 186.2070000000},{"Os", 190.2300000000},
  {"Ir", 192.2170000000},{"Pt", 195.0840000000},{"Au", 196.9665690000},{"Hg", 200.5920000000},
  {"Tl", 204.3800000000},{"Pb", 207.2000000000},{"Bi", 208.9804000000},{"Po", 210},
  {"At", 210},           {"Rn", 222},           {"Fr", 223},           {"Ra", 226},
  {"Ac", 227},           {"Th", 232.0377000000},{"Pa", 231.0358800000},{"U", 238.0289100000},
  {"Np", 237},           {"Pu", 244},           {"Am", 243},           {"Cm", 247},
  {"Bk", 247},           {"Cf", 251},           {"Es", 252},           {"Fm", 257},
  {"Md", 258},           {"No", 259},           {"Lr", 262}};

static void add_atoms_to_system(
  Atom& atom,
  int num_deposit,
  const std::vector<std::string>& symbols_to_add,
  const std::vector<int>& types_to_add,
  const std::vector<double>& masses_to_add,
  const std::vector<double>& positions_to_add,
  const std::vector<double>& velocities_to_add)
{
  const int old_N = atom.number_of_atoms;
  const int new_N = old_N + num_deposit;
  atom.number_of_atoms = new_N;

  std::vector<int> new_cpu_type(new_N);
  std::vector<std::string> new_cpu_atom_symbol(new_N);
  std::vector<double> new_cpu_mass(new_N);
  std::vector<float> new_cpu_charge(new_N, 0.0f);
  std::vector<double> new_cpu_pos(new_N * 3);
  std::vector<double> new_cpu_vel(new_N * 3);
  std::vector<double> new_force(new_N * 3, 0.0);
  std::vector<double> new_virial(new_N * 9);
  std::vector<double> new_potential(new_N);
  std::vector<double> old_force(old_N * 3);
  std::vector<double> old_virial(old_N * 9);
  atom.force_per_atom.copy_to_host(old_force.data());
  atom.virial_per_atom.copy_to_host(old_virial.data());
  atom.potential_per_atom.copy_to_host(new_potential.data());

  for (int i = 0; i < old_N; ++i) {
    new_cpu_type[i] = atom.cpu_type[i];
    new_cpu_atom_symbol[i] = atom.cpu_atom_symbol[i];
    new_cpu_mass[i] = atom.cpu_mass[i];
    new_cpu_charge[i] = atom.cpu_charge[i];
    for (int d = 0; d < 3; ++d) {
      new_cpu_pos[i + d * new_N] = atom.cpu_position_per_atom[i + d * old_N];
      new_cpu_vel[i + d * new_N] = atom.cpu_velocity_per_atom[i + d * old_N];
      new_force[i + d * new_N] = old_force[i + d * old_N];
    }
    for (int k = 0; k < 9; ++k) {
      new_virial[i * 9 + k] = old_virial[i * 9 + k];
    }
  }
  
  for (int i = 0; i < num_deposit; ++i) {
    int idx = old_N + i;
    new_cpu_type[idx] = types_to_add[i];
    new_cpu_atom_symbol[idx] = symbols_to_add[i];
    new_cpu_mass[idx] = masses_to_add[i];
    for (int d = 0; d < 3; ++d) {
      new_cpu_pos[idx + d * new_N] = positions_to_add[i * 3 + d];
      new_cpu_vel[idx + d * new_N] = velocities_to_add[i * 3 + d];
    }
  }
  
  atom.cpu_type = std::move(new_cpu_type);
  atom.cpu_atom_symbol = std::move(new_cpu_atom_symbol);
  atom.cpu_mass = std::move(new_cpu_mass);
  atom.cpu_charge = std::move(new_cpu_charge);
  atom.cpu_position_per_atom = std::move(new_cpu_pos);
  atom.cpu_velocity_per_atom = std::move(new_cpu_vel);

  atom.type.resize(new_N);
  atom.mass.resize(new_N);
  atom.charge.resize(new_N);
  atom.potential_per_atom.resize(new_N);
  atom.position_per_atom.resize(new_N * 3);
  atom.velocity_per_atom.resize(new_N * 3);
  atom.force_per_atom.resize(new_N * 3);
  atom.virial_per_atom.resize(new_N * 9);

  atom.type.copy_from_host(atom.cpu_type.data());
  atom.mass.copy_from_host(atom.cpu_mass.data());
  atom.charge.copy_from_host(atom.cpu_charge.data());
  atom.position_per_atom.copy_from_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  atom.force_per_atom.copy_from_host(new_force.data());
  atom.virial_per_atom.copy_from_host(new_virial.data());
  atom.potential_per_atom.copy_from_host(new_potential.data());
}

void Deposition::parse(const char** param, int num_param)
{
  printf("Deposition: Initialize atom deposition (single type).\n");
  
  if (num_param != 5) {
    PRINT_INPUT_ERROR("deposition should have 4 parameters.\n");
  }

  if (!is_valid_int(param[1], &deposit_interval)) {
    PRINT_INPUT_ERROR("interval should be an integer.\n");
  }
  if (deposit_interval <= 0) {
    PRINT_INPUT_ERROR("interval should > 0.\n");
  }
  printf("    Deposition interval every %d steps\n", deposit_interval);

  std::string filename_potential = get_filename_potential();
  atom_symbols = get_atom_symbols(filename_potential);
  int num_types = atom_symbols.size();

  if (!is_valid_int(param[2], &type_deposit)) {
    PRINT_INPUT_ERROR("type index should be an integer.\n");
  }
  if (type_deposit < 0) {
    PRINT_INPUT_ERROR("type index should >= 0.\n");
  }
  if (type_deposit >= num_types) {
    PRINT_INPUT_ERROR("type index should be less than number of types in potential.\n");
  }

  deposit_mass = MASS_TABLE.at(atom_symbols[type_deposit]);

  if (!is_valid_int(param[3], &num_deposit)) {
    PRINT_INPUT_ERROR("number of atoms should be an integer.\n");
  }
  if (num_deposit <= 0) {
    PRINT_INPUT_ERROR("number of atoms should > 0.\n");
  }

  printf("    Will deposit %d atoms of type %d (%s, mass %g) per event\n", 
         num_deposit, type_deposit, atom_symbols[type_deposit].c_str(), deposit_mass);
  printf("    Total atoms per deposition: %d\n", num_deposit);
  printf("    Z-position: %g * Lz \n", z_position_fraction);
  printf("    Z-velocity: %g A/fs \n", vz_deposit);

  if (!is_valid_real(param[4], &vz_deposit)) {
    PRINT_INPUT_ERROR("velocity should be a number.\n");
  }
  if (vz_deposit <= 0) {
    PRINT_INPUT_ERROR("velocity should > 0.\n");
  }

  is_deposition = true;
}

void Deposition::perform_deposition(Atom& atom, Box& box)
{
  double z_pos = z_position_fraction * box.cpu_h[8];
  double lx = box.cpu_h[0];
  double ly = box.cpu_h[4];
  
  std::vector<std::string> deposit_symbols;
  std::vector<int> deposit_types;
  std::vector<double> deposit_masses;
  std::vector<double> deposit_positions;
  std::vector<double> deposit_velocities;
  
  deposit_symbols.resize(num_deposit);
  deposit_types.resize(num_deposit);
  deposit_masses.resize(num_deposit);
  deposit_positions.resize(num_deposit * 3);
  deposit_velocities.resize(num_deposit * 3);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist_x(0.0, lx);
  std::uniform_real_distribution<double> dist_y(0.0, ly);

  for (int n = 0; n < num_deposit; ++n) {
    deposit_symbols[n] = atom_symbols[type_deposit];
    deposit_types[n] = type_deposit;
    deposit_masses[n] = deposit_mass;
    
    deposit_positions[n * 3 + 0] = dist_x(gen);
    deposit_positions[n * 3 + 1] = dist_y(gen);
    deposit_positions[n * 3 + 2] = z_pos;
    
    deposit_velocities[n * 3 + 0] = 0.0;
    deposit_velocities[n * 3 + 1] = 0.0;
    deposit_velocities[n * 3 + 2] = -vz_deposit;
  }
  
  int old_N = atom.number_of_atoms;
  add_atoms_to_system(atom, num_deposit, deposit_symbols, deposit_types, deposit_masses, deposit_positions, deposit_velocities);
  int new_N = atom.number_of_atoms;
  
  num_deposited_total += num_deposit;
  num_deposition_events++;

  printf("    Deposited %d atoms (type %d, N: %d -> %d, event #%d)\n", 
         num_deposit, type_deposit, old_N, new_N, num_deposition_events);
}

void Deposition::compute(int step, Atom& atom, Box& box)
{
  if (step % deposit_interval == 0 && is_deposition){
    perform_deposition(atom, box);
  } else {
    return;
  }
}

void Deposition::finalize(void)
{
  if (is_deposition) {
    printf("Deposition: Finalized. Total events: %d, Total atoms deposited: %d\n", 
           num_deposition_events, num_deposited_total);
  }
  is_deposition = false;
  num_deposited_total = 0;
  num_deposition_events = 0;
}

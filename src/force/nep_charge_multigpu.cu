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
The neuroevolution potential (NEP) with charge (qNEP).
This is the multi-GPU (single-node) version. The short-range NEP part is
spatially decomposed over the GPUs as in nep_multigpu.cu, while the long-range
Coulomb part (PPPM or Ewald) is done on GPU 0 using globally gathered charges.

This is the multi-GPU (single-node) version. It has good parallel efficiency
when there is NVlink, but is also not very bad when there is only PCI-E.
------------------------------------------------------------------------------*/

#include "nep_charge_multigpu.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstddef>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP_Charge_MULTIGPU::check_ewald_pppm()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  use_pppm = true;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "kspace") {
        if (tokens.size() != 2) {
          std::cout << "kspace must have 1 parameter\n";
          exit(1);
        }
        std::string kspace_method = tokens[1];
        if (kspace_method == "ewald") {
          use_pppm = false;
        } else if (kspace_method == "pppm") {
          use_pppm = true;
        } else {
          std::cout << "kspace method can only be ewald or pppm\n";
          exit(1);
        }
      }
    }
  }

  input_run.close();
}

void NEP_Charge_MULTIGPU::initialize_dftd3()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }

  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "dftd3") {
        input_run.close();
        PRINT_INPUT_ERROR("dftd3 has not been implemented for multi-GPU version.");
      }
    }
  }

  input_run.close();
}

NEP_Charge_MULTIGPU::NEP_Charge_MULTIGPU(
  const int num_gpus,
  const char* file_potential,
  const int num_atoms,
  const int partition_direction_input)
{
  printf("Try to use %d GPUs for the NEP-charge part.\n", num_gpus);

  partition_direction = partition_direction_input;
  number_of_atoms = num_atoms;

  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4_charge1") {
    zbl.enabled = false;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_zbl_charge1") {
    zbl.enabled = true;
    paramb.charge_mode = 1;
  } else if (tokens[0] == "nep4_charge2") {
    zbl.enabled = false;
    paramb.charge_mode = 2;
  } else if (tokens[0] == "nep4_zbl_charge2") {
    zbl.enabled = true;
    paramb.charge_mode = 2;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 charge models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("Use the NEP4-Charge%d potential with %d atom type.\n",
      paramb.charge_mode, paramb.num_types);
  } else {
    printf("Use the NEP4-Charge%d potential with %d atom types.\n",
      paramb.charge_mode, paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexibled = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  // cutoff
  tokens = get_tokens(input);
  if (tokens.size() != 5) {
    std::cout << "This line should be cutoff rc_radial rc_angular MN_radial MN_angular.\n";
    exit(1);
  }
  paramb.rc_radial = get_double_from_token(tokens[1], __FILE__, __LINE__);
  paramb.rc_angular = get_double_from_token(tokens[2], __FILE__, __LINE__);
  printf("    radial cutoff = %g A.\n", paramb.rc_radial);
  printf("    angular cutoff = %g A.\n", paramb.rc_angular);

  int MN_radial = get_int_from_token(tokens[3], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[4], __FILE__, __LINE__);
  printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = get_tokens(input);
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);
  if (paramb.has_q_222) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_1111) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_112) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_123) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_233) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_134) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  tokens = get_tokens(input);
  if (tokens.size() != 3) {
    std::cout << "This line should be ANN num_neurons 0." << std::endl;
    exit(1);
  }
  annmb[0].num_neurons1 = get_int_from_token(tokens[1], __FILE__, __LINE__);
  annmb[0].dim = (paramb.n_max_radial + 1) + paramb.dim_angular;
  printf("    ANN = %d-%d-1.\n", annmb[0].dim, annmb[0].num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial; // largest cutoff
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb[0].num_para_ann = (annmb[0].dim + 3) * annmb[0].num_neurons1 * paramb.num_types + 2;

  printf("    number of neural network parameters = %d.\n", annmb[0].num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb[0].num_para = annmb[0].num_para_ann + num_para_descriptor;
  printf("    total number of parameters = %d\n", annmb[0].num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb[0].num_para + annmb[0].dim);
  for (int n = 0; n < annmb[0].num_para + annmb[0].dim; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }

  // flexible zbl potential parameters
  if (zbl.flexibled) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  paramb.num_gpus = num_gpus;
  nep_temp_data.num_atoms_per_gpu = num_atoms;
  if (num_gpus > 1) {
    nep_temp_data.num_atoms_per_gpu = (num_atoms * 1.25) / num_gpus;
  }

  for (int gpu = 0; gpu < num_gpus; ++gpu) {
    annmb[gpu].num_para_ann = annmb[0].num_para_ann;
    annmb[gpu].num_para = annmb[0].num_para;
    annmb[gpu].dim = annmb[0].dim;
    annmb[gpu].num_neurons1 = annmb[0].num_neurons1;
#ifndef ZHEYONG
    CHECK(gpuSetDevice(gpu));
#endif

    nep_data[gpu].parameters.resize(annmb[gpu].num_para + annmb[gpu].dim);
    nep_data[gpu].parameters.copy_from_host(parameters.data());

    update_potential(nep_data[gpu].parameters.data(), annmb[gpu]);
    annmb[gpu].q_scaler = nep_data[gpu].parameters.data() + annmb[gpu].num_para;

    nep_data[gpu].cell_count.resize(num_atoms);
    nep_data[gpu].cell_count_sum.resize(num_atoms);
    nep_data[gpu].cell_contents.resize(num_atoms);

    CHECK(gpuStreamCreate(&nep_data[gpu].stream));
  }

  CHECK(gpuSetDevice(0));

  // charge related parameters and the long-range solver (living on GPU 0)
  charge_para.alpha = float(PI) / paramb.rc_radial; // a good value
  check_ewald_pppm();
  if (use_pppm) {
    pppm.initialize(charge_para.alpha);
  } else {
    ewald.initialize(charge_para.alpha);
  }
  charge_para.two_alpha_over_sqrt_pi = 2.0f * charge_para.alpha / sqrt(float(PI));
  charge_para.A = erfc(float(PI)) / (paramb.rc_radial * paramb.rc_radial);
  charge_para.A += charge_para.two_alpha_over_sqrt_pi * exp(-float(PI * PI)) / paramb.rc_radial;
  charge_para.B = - erfc(float(PI)) / paramb.rc_radial - charge_para.A * paramb.rc_radial;

  nep_temp_data.cell_count_sum_cpu.resize(num_atoms);
  nep_temp_data.cell_count.resize(num_atoms);
  nep_temp_data.cell_count_sum.resize(num_atoms);
  nep_temp_data.cell_contents.resize(num_atoms);
  nep_temp_data.charge_sum.resize(16);
  nep_temp_data.charge_sum_cpu.resize(16);

  allocate_memory();

  initialize_dftd3();
}

void NEP_Charge_MULTIGPU::allocate_memory()
{
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(gpuSetDevice(gpu));
#endif

    nep_data[gpu].f12x.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].f12y.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].f12z.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].NN_radial.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].NL_radial.resize(static_cast<size_t>(nep_temp_data.num_atoms_per_gpu) * paramb.MN_radial);
    nep_data[gpu].NN_angular.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].NL_angular.resize(nep_temp_data.num_atoms_per_gpu * paramb.MN_angular);
    nep_data[gpu].Fp.resize(static_cast<size_t>(nep_temp_data.num_atoms_per_gpu) * annmb[gpu].dim);
    nep_data[gpu].sum_fxyz.resize(static_cast<size_t>(nep_temp_data.num_atoms_per_gpu) * (paramb.n_max_angular + 1) *
      ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
    nep_data[gpu].type.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].position.resize(nep_temp_data.num_atoms_per_gpu * 3);
    nep_data[gpu].potential.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].force.resize(nep_temp_data.num_atoms_per_gpu * 3);
    nep_data[gpu].virial.resize(nep_temp_data.num_atoms_per_gpu * 9);
    nep_data[gpu].D_real.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].charge.resize(nep_temp_data.num_atoms_per_gpu);
    nep_data[gpu].charge_derivative.resize(
      static_cast<size_t>(nep_temp_data.num_atoms_per_gpu) * annmb[gpu].dim);
    nep_data[gpu].bec.resize(nep_temp_data.num_atoms_per_gpu * 9);
    nep_data[gpu].charge_sum.resize(1);
  }

  CHECK(gpuSetDevice(0));

  nep_temp_data.type.resize(nep_temp_data.num_atoms_per_gpu);
  nep_temp_data.position.resize(nep_temp_data.num_atoms_per_gpu * 3);
  nep_temp_data.potential.resize(nep_temp_data.num_atoms_per_gpu);
  nep_temp_data.force.resize(nep_temp_data.num_atoms_per_gpu * 3);
  nep_temp_data.virial.resize(nep_temp_data.num_atoms_per_gpu * 9);
  nep_temp_data.float_temp.resize(nep_temp_data.num_atoms_per_gpu);

  charge_full.resize(number_of_atoms);
  bec_full.resize(number_of_atoms * 9);
  D_real_full.resize(number_of_atoms);
  lr_force.resize(number_of_atoms * 3);
  lr_virial.resize(number_of_atoms * 9);
  lr_potential.resize(number_of_atoms);
}

NEP_Charge_MULTIGPU::~NEP_Charge_MULTIGPU(void)
{
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuStreamDestroy(nep_data[gpu].stream));
  }
}

void NEP_Charge_MULTIGPU::update_potential(float* parameters, ANN& ann)
{
  const int num_outputs = 2;
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1 * num_outputs;
  }
  ann.sqrt_epsilon_inf = pointer;
  pointer += 1;
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  cell_id_x = floor(sx * box.thickness_x * rc_inv);
  cell_id_y = floor(sy * box.thickness_y * rc_inv);
  cell_id_z = floor(sz * box.thickness_z * rc_inv);

  while (cell_id_x < 0)
    cell_id_x += nx;
  while (cell_id_x >= nx)
    cell_id_x -= nx;
  while (cell_id_y < 0)
    cell_id_y += ny;
  while (cell_id_y >= ny)
    cell_id_y -= ny;
  while (cell_id_z < 0)
    cell_id_z += nz;
  while (cell_id_z >= nz)
    cell_id_z -= nz;
  if (partition_direction == 0) {
    cell_id = cell_id_y + ny * (cell_id_z + nz * cell_id_x);
  } else if (partition_direction == 1) {
    cell_id = cell_id_x + nx * (cell_id_z + nz * cell_id_y);
  } else {
    cell_id = cell_id_x + nx * (cell_id_y + ny * cell_id_z);
  }
}

static __device__ void find_cell_id(
  const int partition_direction,
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id)
{
  int cell_id_x, cell_id_y, cell_id_z;
  find_cell_id(
    partition_direction,
    box,
    x,
    y,
    z,
    rc_inv,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);
}

static __global__ void find_cell_counts(
  const int partition_direction,
  const Box box,
  const int N,
  int* cell_count,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    atomicAdd(&cell_count[cell_id], 1);
  }
}

static __global__ void find_cell_contents(
  const int partition_direction,
  const Box box,
  const int N,
  int* cell_count,
  const int* cell_count_sum,
  int* cell_contents,
  const double* x,
  const double* y,
  const double* z,
  const int nx,
  const int ny,
  const int nz,
  const double rc_inv)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int cell_id;
    find_cell_id(partition_direction, box, x[n1], y[n1], z[n1], rc_inv, nx, ny, nz, cell_id);
    const int ind = atomicAdd(&cell_count[cell_id], 1);
    cell_contents[cell_count_sum[cell_id] + ind] = n1;
  }
}

static void __global__ set_to_zero(int size, int* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0;
  }
}

static void __global__ set_to_zero_double(int size, double* data)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    data[n] = 0.0;
  }
}

static void find_cell_list(
  gpuStream_t& stream,
  const int partition_direction,
  const double rc,
  const int* num_bins,
  Box& box,
  const int N,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents)
{
  const int offset = position_per_atom.size() / 3;
  const int block_size = 256;
  const int grid_size = (N - 1) / block_size + 1;
  const double rc_inv = 1.0 / rc;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + offset;
  const double* z = position_per_atom.data() + offset * 2;
  const int N_cells = num_bins[0] * num_bins[1] * num_bins[2];

  // number of cells is allowed to be larger than the number of atoms
  if (N_cells > cell_count.size()) {
    cell_count.resize(N_cells);
    cell_count_sum.resize(N_cells);
  }

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_count_sum.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count_sum.size(), cell_count_sum.data());
  GPU_CHECK_KERNEL

  set_to_zero<<<(cell_contents.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_contents.size(), cell_contents.data());
  GPU_CHECK_KERNEL

  find_cell_counts<<<grid_size, block_size, 0, stream>>>(
    partition_direction,
    box,
    N,
    cell_count.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
#ifdef USE_HIP
    thrust::hip::par.on(stream),
#else
    thrust::cuda::par.on(stream),
#endif
    cell_count.data(),
    cell_count.data() + N_cells,
    cell_count_sum.data());

  set_to_zero<<<(cell_count.size() - 1) / 64 + 1, 64, 0, stream>>>(
    cell_count.size(), cell_count.data());
  GPU_CHECK_KERNEL

  find_cell_contents<<<grid_size, block_size, 0, stream>>>(
    partition_direction,
    box,
    N,
    cell_count.data(),
    cell_count_sum.data(),
    cell_contents.data(),
    x,
    y,
    z,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    rc_inv);
  GPU_CHECK_KERNEL
}

static __global__ void find_neighbor_list_large_box(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  const int partition_direction,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;
  int count_angular = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    partition_direction,
    box,
    x1,
    y1,
    z1,
    2.0f * paramb.rcinv_radial,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int xxx = xx;
        int yyy = yy;
        int zzz = zz;
        if (cell_id_x + xx < 0)
          xxx += nx;
        else if (cell_id_x + xx >= nx)
          xxx -= nx;
        if (cell_id_y + yy < 0)
          yyy += ny;
        else if (cell_id_y + yy >= ny)
          yyy -= ny;
        if (cell_id_z + zz < 0)
          zzz += nz;
        else if (cell_id_z + zz >= nz)
          zzz -= nz;

        int neighbor_cell = cell_id;
        if (partition_direction == 0) {
          neighbor_cell += (xxx * nz + zzz) * ny + yyy;
        } else if (partition_direction == 1) {
          neighbor_cell += (yyy * nz + zzz) * nx + xxx;
        } else {
          neighbor_cell += (zzz * ny + yyy) * nx + xxx;
        }

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;

          float rc_radial = paramb.rc_radial;
          float rc_angular = paramb.rc_angular;

          if (d12_square >= rc_radial * rc_radial) {
            continue;
          }

          g_NL_radial[static_cast<size_t>(N) * count_radial++ + n1] = n2;

          if (d12_square < rc_angular * rc_angular) {
            g_NL_angular[count_angular++ * N + n1] = n2;
          }
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}

static __global__ void find_descriptor(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  NEP_Charge_MULTIGPU::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  float* g_charge,
  float* g_charge_derivative,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += gn12;
      }
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

      float F = 0.0f, Fp[MAX_DIM] = {0.0f};
      float charge = 0.0f;
      float charge_derivative[MAX_DIM] = {0.0f};

      apply_ann_one_layer_charge(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[t1],
        annmb.b0[t1],
        annmb.w1[t1],
        annmb.b1,
        q,
        F,
        Fp,
        charge,
        charge_derivative);

      g_pe[n1] += F;
      g_charge[n1] = charge;

      for (int d = 0; d < annmb.dim; ++d) {
        g_Fp[d * N + n1] = Fp[d] * annmb.q_scaler[d];
        g_charge_derivative[d * N + n1] = charge_derivative[d] * annmb.q_scaler[d];
      }
  }
}

// partial sum of charge over [N1, N2) on one GPU
static __global__ void find_charge_sum(
  const int N1,
  const int N2,
  const float* g_charge,
  double* g_charge_sum)
{
  int tid = threadIdx.x;
  __shared__ double s_charge[1024];
  double charge = 0.0;
  for (int n = N1 + tid; n < N2; n += 1024) {
    charge += (double)g_charge[n];
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_charge_sum[0] = s_charge[0];
  }
}

// enforce global charge neutrality by shifting the charge by a constant
static __global__ void shift_charge(
  const int N1,
  const int N2,
  const float delta,
  float* g_charge)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    g_charge[n1] += delta;
  }
}

static __global__ void find_bec_diagonal(
  const int N,
  const int N1,
  const int N2,
  const float* g_q,
  float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 < N2) {
    g_bec[n1 + N * 0] = g_q[n1];
    g_bec[n1 + N * 1] = 0.0f;
    g_bec[n1 + N * 2] = 0.0f;
    g_bec[n1 + N * 3] = 0.0f;
    g_bec[n1 + N * 4] = g_q[n1];
    g_bec[n1 + N * 5] = 0.0f;
    g_bec[n1 + N * 6] = 0.0f;
    g_bec[n1 + N * 7] = 0.0f;
    g_bec[n1 + N * 8] = g_q[n1];
  }
}

static __global__ void find_bec_radial(
  const NEP_Charge_MULTIGPU::ParaMB paramb,
  const NEP_Charge_MULTIGPU::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_bec_angular(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  NEP_Charge_MULTIGPU::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_charge_derivative,
  const float* g_sum_fxyz,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }

      float bec_xx = 0.5f* (r12[0] * f12[0]);
      float bec_xy = 0.5f* (r12[0] * f12[1]);
      float bec_xz = 0.5f* (r12[0] * f12[2]);
      float bec_yx = 0.5f* (r12[1] * f12[0]);
      float bec_yy = 0.5f* (r12[1] * f12[1]);
      float bec_yz = 0.5f* (r12[1] * f12[2]);
      float bec_zx = 0.5f* (r12[2] * f12[0]);
      float bec_zy = 0.5f* (r12[2] * f12[1]);
      float bec_zz = 0.5f* (r12[2] * f12[2]);

      atomicAdd(&g_bec[n1], bec_xx);
      atomicAdd(&g_bec[n1 + N], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2], -bec_xx);
      atomicAdd(&g_bec[n2 + N], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void scale_bec(
  const int N,
  const int N1,
  const int N2,
  const float* sqrt_epsilon_inf,
  float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x + N1;
  if (n1 < N2) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

// gather a float slice corresponding to local indices [N1, N2)
// into a full-length array in global atom order (on GPU 0)
static __global__ void collect_float(
  const int N1,
  const int N2,
  const int M1,
  const int* cell_contents,
  const float* g_local,
  float* g_global)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N2 - N1) {
    g_global[cell_contents[n + M1]] = g_local[n];
  }
}

// scatter a full-length array in global atom order (on GPU 0)
// into local indices [0, N3), the inverse mapping of collect_float
static __global__ void distribute_float(
  const int N1,
  const int N2,
  const int N3,
  const int M0,
  const int M1,
  const int M2,
  const int* cell_contents,
  const float* g_global,
  float* g_local)
{
  int n_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_local < N3) {
    int n_global;
    if (n_local < N1) { // left
      n_global = cell_contents[n_local + M0];
    } else if (n_local < N2) { // middle
      n_global = cell_contents[n_local - N1 + M1];
    } else { // right
      n_global = cell_contents[n_local - N2 + M2];
    }
    g_local[n_local] = g_global[n_global];
  }
}

// sum of a full-length float array on GPU 0 (for computing the mean of D_real)
static __global__ void find_float_sum(
  const int N,
  const float* g_data,
  double* g_sum)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      sum += (double)g_data[n];
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_sum[0] = s_sum[0];
  }
}

// subtract a constant (the global mean of D_real) over [N1, N2)
static __global__ void subtract_float(
  const int N1,
  const int N2,
  const float value,
  float* g_data)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    g_data[n1] -= value;
  }
}

static __global__ void find_force_charge_real_space(
  const int N,
  const NEP_Charge_MULTIGPU::Charge_Para charge_para,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const float* g_charge,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe,
  float* g_D_real)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q1 = g_charge[n1];
    float s_pe = -charge_para.two_alpha_over_sqrt_pi * 0.5f * q1 * q1; // self energy part
    float D_real = -q1 * charge_para.two_alpha_over_sqrt_pi; // self energy part

    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float q2 = g_charge[n2];
      float qq = q1 * q2;
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float erfc_r = erfc(charge_para.alpha * d12) * d12inv;
      D_real += q2 * erfc_r;
      s_pe += 0.5f * qq * erfc_r;
      float f2 = erfc_r + charge_para.two_alpha_over_sqrt_pi * exp(-charge_para.alpha * charge_para.alpha * d12 * d12);
      f2 *= -0.5f * K_C_SP * qq * d12inv * d12inv;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};

      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_D_real[n1] += K_C_SP * D_real;
    g_pe[n1] += K_C_SP * s_pe;
  }
}

static __global__ void find_force_radial(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  NEP_Charge_MULTIGPU::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
          gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];
        }
        float tmp12 = g_Fp[n1 + n * N] + g_charge_derivative[n1 + n * N] * g_D_real[n1];
        float tmp21 = g_Fp[n2 + n * N] + g_charge_derivative[n2 + n * N] * g_D_real[n2];
        tmp12 *= gnp12 * d12inv;
        tmp21 *= gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

static __global__ void find_partial_force_angular(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  NEP_Charge_MULTIGPU::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* g_charge_derivative,
  const float* g_D_real,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      float tmp = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1]
        + g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      Fp[d] = tmp;
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void gpu_find_force_many_body(
  const int number_of_particles,
  const int N1,
  const int N2,
  const Box box,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N2) {
    int neighbor_number = g_neighbor_number[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * number_of_particles + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];

      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      apply_mic(box, x12double, y12double, z12double);
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + number_of_particles * k]) {
          offset = k;
          break;
        }
      }
      index = offset * number_of_particles + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];

      // per atom force
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;

      // per-atom virial
      s_sxx += x12 * f21x;
      s_sxy += x12 * f21y;
      s_sxz += x12 * f21z;
      s_syx += y12 * f21x;
      s_syy += y12 * f21y;
      s_syz += y12 * f21z;
      s_szx += z12 * f21x;
      s_szy += z12 * f21y;
      s_szz += z12 * f21z;
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * number_of_particles] += s_sxx;
    g_virial[n1 + 1 * number_of_particles] += s_syy;
    g_virial[n1 + 2 * number_of_particles] += s_szz;
    g_virial[n1 + 3 * number_of_particles] += s_sxy;
    g_virial[n1 + 4 * number_of_particles] += s_sxz;
    g_virial[n1 + 5 * number_of_particles] += s_syz;
    g_virial[n1 + 6 * number_of_particles] += s_syx;
    g_virial[n1 + 7 * number_of_particles] += s_szx;
    g_virial[n1 + 8 * number_of_particles] += s_szy;
  }
}

static __global__ void find_force_ZBL(
  NEP_Charge_MULTIGPU::ParaMB paramb,
  const int N,
  const NEP_Charge_MULTIGPU::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

static __global__ void distribute_position(
  const int num_atoms_gobal,
  const int num_atoms_local,
  const int N1,
  const int N2,
  const int N3,
  const int M0,
  const int M1,
  const int M2,
  const int* cell_contents,
  const int* g_type_global,
  const double* g_position_global,
  int* g_type_local,
  double* g_position_local)
{
  int n_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (n_local < N3) {
    int n_global;
    if (n_local < N1) { // left
      n_global = cell_contents[n_local + M0];
    } else if (n_local < N2) { // middle
      n_global = cell_contents[n_local - N1 + M1];
    } else { // right
      n_global = cell_contents[n_local - N2 + M2];
    }

    g_type_local[n_local] = g_type_global[n_global];
    for (int d = 0; d < 3; ++d) {
      g_position_local[n_local + d * num_atoms_local] =
        g_position_global[n_global + d * num_atoms_gobal];
    }
  }
}

static __global__ void collect_properties(
  const int num_atoms_global,
  const int num_atoms_local,
  const int N1,
  const int N2,
  const int M1,
  const int* cell_contents,
  const double* g_force_local,
  const double* g_potential_local,
  const double* g_virial_local,
  double* g_force_global,
  double* g_potential_global,
  double* g_virial_global)
{
  int n_local = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n_local < N2) {
    int n_global = cell_contents[n_local - N1 + M1];
    for (int d = 0; d < 3; ++d) {
      g_force_global[n_global + d * num_atoms_global] =
        g_force_local[n_local + d * num_atoms_local];
    }
    g_potential_global[n_global] = g_potential_local[n_local];
    for (int d = 0; d < 9; ++d) {
      g_virial_global[n_global + d * num_atoms_global] =
        g_virial_local[n_local + d * num_atoms_local];
    }
  }
}

// add the long-range contributions (in global atom order, on GPU 0) to the outputs
static __global__ void add_long_range(
  const int N,
  const double* g_lr_potential,
  const double* g_lr_force,
  const double* g_lr_virial,
  double* g_potential,
  double* g_force,
  double* g_virial)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    g_potential[n] += g_lr_potential[n];
    for (int d = 0; d < 3; ++d) {
      g_force[n + d * N] += g_lr_force[n + d * N];
    }
    for (int d = 0; d < 9; ++d) {
      g_virial[n + d * N] += g_lr_virial[n + d * N];
    }
  }
}

void NEP_Charge_MULTIGPU::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position,
  GPU_Vector<double>& potential,
  GPU_Vector<double>& force,
  GPU_Vector<double>& virial)
{
  if (!box.pbc_x || !box.pbc_y || !box.pbc_z) {
    PRINT_INPUT_ERROR("Cannot use non-periodic boundaries for qNEP models.");
  }

  const int N = type.size();

  // the multi-GPU version only supports the large-box path of NEP_Charge
  {
    double volume = box.get_volume();
    double thickness_x = volume / box.get_area(0);
    double thickness_y = volume / box.get_area(1);
    double thickness_z = volume / box.get_area(2);
    if (
      (box.pbc_x && thickness_x <= 2.5 * (paramb.rc_radial + 1.0)) ||
      (box.pbc_y && thickness_y <= 2.5 * (paramb.rc_radial + 1.0)) ||
      (box.pbc_z && thickness_z <= 2.5 * (paramb.rc_radial + 1.0))) {
      std::cout << "The box has a thickness < 2.5 radial cutoffs in a periodic direction.\n";
      std::cout << "This is not allowed for the multi-GPU version of NEP-charge.\n";
      std::cout << "Please increase the periodic direction(s) or use a single GPU.\n";
      exit(1);
    }
  }

  const double rc_cell_list = 0.5 * rc;
  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  if (
    (box.pbc_x && num_bins[0] < 3) || (box.pbc_y && num_bins[1] < 3) ||
    (box.pbc_z && num_bins[2] < 3)) {
    std::cout << "A periodic direction has less than three times of the NEP cutoff.\n";
    std::cout << "This is not allowed for the multi-GPU version of NEP.\n";
    std::cout << "Please increase the periodic direction(s).\n";
    exit(1);
  }

  if (partition_direction < 0) {
    partition_direction = 2;
    if (num_bins[0] >= num_bins[1] && num_bins[0] >= num_bins[2]) {
      partition_direction = 0;
    }
    if (num_bins[1] >= num_bins[0] && num_bins[1] >= num_bins[2]) {
      partition_direction = 1;
    }
  }
  int num_bins_longitudinal = num_bins[partition_direction] / paramb.num_gpus;
  int num_bins_transverse =
    (num_bins[0] * num_bins[1] * num_bins[2]) / num_bins[partition_direction];

  if (num_bins_longitudinal < 10) {
    printf("The longest direction has less than 5 times of the NEP cutoff per GPU.\n");
    printf("Please reduce the number of GPUs or increase the simulation cell size.\n");
    exit(1);
  }

  find_cell_list(
    nep_data[0].stream,
    partition_direction,
    rc_cell_list,
    num_bins,
    box,
    N,
    position,
    nep_temp_data.cell_count,
    nep_temp_data.cell_count_sum,
    nep_temp_data.cell_contents);

  if (num_bins[0] * num_bins[1] * num_bins[2] > nep_temp_data.cell_count_sum_cpu.size()) {
    nep_temp_data.cell_count_sum_cpu.resize(num_bins[0] * num_bins[1] * num_bins[2]);
  }

  nep_temp_data.cell_count_sum.copy_to_host(
    nep_temp_data.cell_count_sum_cpu.data(), num_bins[0] * num_bins[1] * num_bins[2]);

  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    if (paramb.num_gpus == 1) {
      nep_data[gpu].N1 = 0;
      nep_data[gpu].N4 = 0;
      nep_data[gpu].N2 = N;
      nep_data[gpu].N5 = N;
      nep_data[gpu].N3 = N;
      nep_data[gpu].M0 = 0;
      nep_data[gpu].M1 = 0;
      nep_data[gpu].M2 = N;
    } else {
      if (gpu == 0) {
        nep_data[gpu].M0 =
          nep_temp_data
            .cell_count_sum_cpu[(num_bins[partition_direction] - 4) * num_bins_transverse];
        nep_data[gpu].M1 = 0;
        nep_data[gpu].M2 =
          nep_temp_data.cell_count_sum_cpu[num_bins_longitudinal * num_bins_transverse];
        nep_data[gpu].N1 = N - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data
            .cell_count_sum_cpu[(num_bins[partition_direction] - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = nep_data[gpu].N1 + nep_data[gpu].M2;
        nep_data[gpu].N5 =
          nep_data[gpu].N1 +
          nep_temp_data.cell_count_sum_cpu[(num_bins_longitudinal + 2) * num_bins_transverse];
        nep_data[gpu].N3 =
          nep_data[gpu].N1 +
          nep_temp_data.cell_count_sum_cpu[(num_bins_longitudinal + 4) * num_bins_transverse];
      } else if (gpu == paramb.num_gpus - 1) {
        nep_data[gpu].M0 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal - 4) * num_bins_transverse];
        nep_data[gpu].M1 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].M2 = 0;
        nep_data[gpu].N1 = nep_data[gpu].M1 - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data
            .cell_count_sum_cpu[(gpu * num_bins_longitudinal - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = N - nep_data[gpu].M0;
        nep_data[gpu].N5 =
          nep_data[gpu].N2 + nep_temp_data.cell_count_sum_cpu[2 * num_bins_transverse];
        nep_data[gpu].N3 =
          nep_data[gpu].N2 + nep_temp_data.cell_count_sum_cpu[4 * num_bins_transverse];
      } else {
        nep_data[gpu].M0 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal - 4) * num_bins_transverse];
        nep_data[gpu].M1 =
          nep_temp_data.cell_count_sum_cpu[(gpu * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].M2 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal) * num_bins_transverse];
        nep_data[gpu].N1 = nep_data[gpu].M1 - nep_data[gpu].M0;
        nep_data[gpu].N4 =
          nep_temp_data
            .cell_count_sum_cpu[(gpu * num_bins_longitudinal - 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N2 = nep_data[gpu].M2 - nep_data[gpu].M0;
        nep_data[gpu].N5 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal + 2) * num_bins_transverse] -
          nep_data[gpu].M0;
        nep_data[gpu].N3 =
          nep_temp_data
            .cell_count_sum_cpu[((gpu + 1) * num_bins_longitudinal + 4) * num_bins_transverse] -
          nep_data[gpu].M0;
      }
    }
    if (nep_data[gpu].N3 > nep_temp_data.num_atoms_per_gpu) {
      nep_temp_data.num_atoms_per_gpu = nep_data[gpu].N3 * 1.1;
      allocate_memory();
    }
  }

  // serial
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    distribute_position<<<(nep_data[gpu].N3 - 1) / 64 + 1, 64>>>(
      N,
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      nep_data[gpu].N3,
      nep_data[gpu].M0,
      nep_data[gpu].M1,
      nep_data[gpu].M2,
      nep_temp_data.cell_contents.data(),
      type.data(),
      position.data(),
      nep_temp_data.type.data(),
      nep_temp_data.position.data());
    GPU_CHECK_KERNEL

    CHECK(gpuMemcpy(
      nep_data[gpu].type.data(),
      nep_temp_data.type.data(),
      sizeof(int) * nep_data[gpu].N3,
      gpuMemcpyDeviceToDevice));
    for (int d = 0; d < 3; ++d) {
      CHECK(gpuMemcpy(
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * d,
        nep_temp_data.position.data() + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * nep_data[gpu].N3,
        gpuMemcpyDeviceToDevice));
    }
  }

  // parallel: neighbor list, descriptor, and partial sum of charge
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(gpuSetDevice(gpu));
#endif

    set_to_zero_double<<<(nep_temp_data.num_atoms_per_gpu - 1) / 64 + 1, 64, 0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu, nep_data[gpu].potential.data());
    GPU_CHECK_KERNEL
    set_to_zero_double<<<(nep_temp_data.num_atoms_per_gpu * 3 - 1) / 64 + 1, 64, 0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu * 3, nep_data[gpu].force.data());
    GPU_CHECK_KERNEL
    set_to_zero_double<<<(nep_temp_data.num_atoms_per_gpu * 9 - 1) / 64 + 1, 64, 0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu * 9, nep_data[gpu].virial.data());
    GPU_CHECK_KERNEL

    find_cell_list(
      nep_data[gpu].stream,
      partition_direction,
      rc_cell_list,
      num_bins,
      box,
      nep_data[gpu].N3,
      nep_data[gpu].position,
      nep_data[gpu].cell_count,
      nep_data[gpu].cell_count_sum,
      nep_data[gpu].cell_contents);

    find_neighbor_list_large_box<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      partition_direction,
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      num_bins[0],
      num_bins[1],
      num_bins[2],
      box,
      nep_data[gpu].type.data(),
      nep_data[gpu].cell_count.data(),
      nep_data[gpu].cell_count_sum.data(),
      nep_data[gpu].cell_contents.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].NN_radial.data(),
      nep_data[gpu].NL_radial.data(),
      nep_data[gpu].NN_angular.data(),
      nep_data[gpu].NL_angular.data());
    GPU_CHECK_KERNEL

    find_descriptor<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      annmb[gpu],
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      box,
      nep_data[gpu].NN_radial.data(),
      nep_data[gpu].NL_radial.data(),
      nep_data[gpu].NN_angular.data(),
      nep_data[gpu].NL_angular.data(),
      nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].potential.data(),
      nep_data[gpu].Fp.data(),
      nep_data[gpu].charge.data(),
      nep_data[gpu].charge_derivative.data(),
      nep_data[gpu].sum_fxyz.data());
    GPU_CHECK_KERNEL

    find_charge_sum<<<1, 1024, 0, nep_data[gpu].stream>>>(
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      nep_data[gpu].charge.data(),
      nep_data[gpu].charge_sum.data());
    GPU_CHECK_KERNEL
  }

  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuSetDevice(gpu));
    CHECK(gpuDeviceSynchronize());
  }

  CHECK(gpuSetDevice(0));

  // enforce global charge neutrality
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuMemcpy(
      nep_temp_data.charge_sum.data() + gpu,
      nep_data[gpu].charge_sum.data(),
      sizeof(double),
      gpuMemcpyDeviceToDevice));
  }
  nep_temp_data.charge_sum.copy_to_host(
    nep_temp_data.charge_sum_cpu.data(), paramb.num_gpus);
  double total_charge = 0.0;
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    total_charge += nep_temp_data.charge_sum_cpu[gpu];
  }
  const float charge_shift = float(-total_charge / N);

  // parallel: shift charge and compute BEC
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(gpuSetDevice(gpu));
#endif

    shift_charge<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      charge_shift,
      nep_data[gpu].charge.data());
    GPU_CHECK_KERNEL

    // get BEC (the diagonal part)
    find_bec_diagonal<<<
      (nep_data[gpu].N3 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu,
      0,
      nep_data[gpu].N3,
      nep_data[gpu].charge.data(),
      nep_data[gpu].bec.data());
    GPU_CHECK_KERNEL

    // get BEC (radial descriptor part)
    find_bec_radial<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      annmb[gpu],
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      box,
      nep_data[gpu].NN_radial.data(),
      nep_data[gpu].NL_radial.data(),
      nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].charge_derivative.data(),
      nep_data[gpu].bec.data());
    GPU_CHECK_KERNEL

    // get BEC (angular descriptor part)
    find_bec_angular<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      annmb[gpu],
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      box,
      nep_data[gpu].NN_angular.data(),
      nep_data[gpu].NL_angular.data(),
      nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].charge_derivative.data(),
      nep_data[gpu].sum_fxyz.data(),
      nep_data[gpu].bec.data());
    GPU_CHECK_KERNEL

    // scale q to q * sqrt(epsilon_inf)
    scale_bec<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      annmb[gpu].sqrt_epsilon_inf,
      nep_data[gpu].bec.data());
    GPU_CHECK_KERNEL
  }

  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuSetDevice(gpu));
    CHECK(gpuDeviceSynchronize());
  }

  CHECK(gpuSetDevice(0));

  // gather charge and BEC for [N1, N2) into full arrays in global atom order on GPU 0
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    const int num_local = nep_data[gpu].N2 - nep_data[gpu].N1;

    CHECK(gpuMemcpy(
      nep_temp_data.float_temp.data(),
      nep_data[gpu].charge.data() + nep_data[gpu].N1,
      sizeof(float) * num_local,
      gpuMemcpyDeviceToDevice));
    collect_float<<<(num_local - 1) / 64 + 1, 64>>>(
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      nep_data[gpu].M1,
      nep_temp_data.cell_contents.data(),
      nep_temp_data.float_temp.data(),
      charge_full.data());
    GPU_CHECK_KERNEL

    for (int d = 0; d < 9; ++d) {
      CHECK(gpuMemcpy(
        nep_temp_data.float_temp.data(),
        nep_data[gpu].bec.data() + d * nep_temp_data.num_atoms_per_gpu + nep_data[gpu].N1,
        sizeof(float) * num_local,
        gpuMemcpyDeviceToDevice));
      collect_float<<<(num_local - 1) / 64 + 1, 64>>>(
        nep_data[gpu].N1,
        nep_data[gpu].N2,
        nep_data[gpu].M1,
        nep_temp_data.cell_contents.data(),
        nep_temp_data.float_temp.data(),
        bec_full.data() + d * N);
      GPU_CHECK_KERNEL
    }
  }

  // long-range part on GPU 0, using the global positions and charges
  set_to_zero_double<<<(N * 3 - 1) / 64 + 1, 64>>>(N * 3, lr_force.data());
  GPU_CHECK_KERNEL
  set_to_zero_double<<<(N * 9 - 1) / 64 + 1, 64>>>(N * 9, lr_virial.data());
  GPU_CHECK_KERNEL
  set_to_zero_double<<<(N - 1) / 64 + 1, 64>>>(N, lr_potential.data());
  GPU_CHECK_KERNEL

  if (use_pppm) {
    pppm.find_force(
      N,
      0,
      N,
      box,
      charge_full,
      position,
      D_real_full,
      lr_force,
      lr_virial,
      lr_potential);
  } else {
    ewald.find_force(
      N,
      0,
      N,
      box.cpu_h,
      charge_full,
      position,
      D_real_full,
      lr_force,
      lr_virial,
      lr_potential);
  }

  if (paramb.charge_mode == 1) {
    // distribute the raw (long-range) D_real to each GPU, since
    // find_force_charge_real_space accumulates the real-space part onto it
    for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
      distribute_float<<<(nep_data[gpu].N3 - 1) / 64 + 1, 64>>>(
        nep_data[gpu].N1,
        nep_data[gpu].N2,
        nep_data[gpu].N3,
        nep_data[gpu].M0,
        nep_data[gpu].M1,
        nep_data[gpu].M2,
        nep_temp_data.cell_contents.data(),
        D_real_full.data(),
        nep_temp_data.float_temp.data());
      GPU_CHECK_KERNEL

      CHECK(gpuMemcpy(
        nep_data[gpu].D_real.data(),
        nep_temp_data.float_temp.data(),
        sizeof(float) * nep_data[gpu].N3,
        gpuMemcpyDeviceToDevice));
    }

    // real-space Coulomb part over [N1, N2) (the neighbors of these atoms are
    // all within [N4, N5) and have valid charges); it also adds to the local D_real
    for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
      CHECK(gpuSetDevice(gpu));
#endif

      find_force_charge_real_space<<<
        (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1,
        64,
        0,
        nep_data[gpu].stream>>>(
        nep_temp_data.num_atoms_per_gpu,
        charge_para,
        nep_data[gpu].N1,
        nep_data[gpu].N2,
        box,
        nep_data[gpu].NN_radial.data(),
        nep_data[gpu].NL_radial.data(),
        nep_data[gpu].charge.data(),
        nep_data[gpu].position.data(),
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].force.data(),
        nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].virial.data(),
        nep_data[gpu].potential.data(),
        nep_data[gpu].D_real.data());
      GPU_CHECK_KERNEL
    }

    for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
      CHECK(gpuSetDevice(gpu));
      CHECK(gpuDeviceSynchronize());
    }

    CHECK(gpuSetDevice(0));

    // gather the updated D_real for [N1, N2) back to GPU 0
    for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
      const int num_local = nep_data[gpu].N2 - nep_data[gpu].N1;
      CHECK(gpuMemcpy(
        nep_temp_data.float_temp.data(),
        nep_data[gpu].D_real.data() + nep_data[gpu].N1,
        sizeof(float) * num_local,
        gpuMemcpyDeviceToDevice));
      collect_float<<<(num_local - 1) / 64 + 1, 64>>>(
        nep_data[gpu].N1,
        nep_data[gpu].N2,
        nep_data[gpu].M1,
        nep_temp_data.cell_contents.data(),
        nep_temp_data.float_temp.data(),
        D_real_full.data());
      GPU_CHECK_KERNEL
    }
  }

  // Chain rule correction: D_real -= mean(D_real), with the mean taken globally
  find_float_sum<<<1, 1024>>>(N, D_real_full.data(), nep_temp_data.charge_sum.data());
  GPU_CHECK_KERNEL
  nep_temp_data.charge_sum.copy_to_host(nep_temp_data.charge_sum_cpu.data(), 1);
  const float mean_D_real = float(nep_temp_data.charge_sum_cpu[0] / N);
  subtract_float<<<(N - 1) / 64 + 1, 64>>>(0, N, mean_D_real, D_real_full.data());
  GPU_CHECK_KERNEL

  // distribute the final (mean-subtracted) D_real to each GPU
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    distribute_float<<<(nep_data[gpu].N3 - 1) / 64 + 1, 64>>>(
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      nep_data[gpu].N3,
      nep_data[gpu].M0,
      nep_data[gpu].M1,
      nep_data[gpu].M2,
      nep_temp_data.cell_contents.data(),
      D_real_full.data(),
      nep_temp_data.float_temp.data());
    GPU_CHECK_KERNEL

    CHECK(gpuMemcpy(
      nep_data[gpu].D_real.data(),
      nep_temp_data.float_temp.data(),
      sizeof(float) * nep_data[gpu].N3,
      gpuMemcpyDeviceToDevice));
  }

  // parallel: the rest of the force calculation
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {

#ifndef ZHEYONG
    CHECK(gpuSetDevice(gpu));
#endif

    find_force_radial<<<
      (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      annmb[gpu],
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      box,
      nep_data[gpu].NN_radial.data(),
      nep_data[gpu].NL_radial.data(),
      nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].Fp.data(),
      nep_data[gpu].charge_derivative.data(),
      nep_data[gpu].D_real.data(),
      nep_data[gpu].force.data(),
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].virial.data());
    GPU_CHECK_KERNEL

    find_partial_force_angular<<<
      (nep_data[gpu].N5 - nep_data[gpu].N4 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      paramb,
      annmb[gpu],
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N4,
      nep_data[gpu].N5,
      box,
      nep_data[gpu].NN_angular.data(),
      nep_data[gpu].NL_angular.data(),
      nep_data[gpu].type.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].Fp.data(),
      nep_data[gpu].charge_derivative.data(),
      nep_data[gpu].D_real.data(),
      nep_data[gpu].sum_fxyz.data(),
      nep_data[gpu].f12x.data(),
      nep_data[gpu].f12y.data(),
      nep_data[gpu].f12z.data());
    GPU_CHECK_KERNEL

    gpu_find_force_many_body<<<
      (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1,
      64,
      0,
      nep_data[gpu].stream>>>(
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      box,
      nep_data[gpu].NN_angular.data(),
      nep_data[gpu].NL_angular.data(),
      nep_data[gpu].f12x.data(),
      nep_data[gpu].f12y.data(),
      nep_data[gpu].f12z.data(),
      nep_data[gpu].position.data(),
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].force.data(),
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
      nep_data[gpu].virial.data());
    GPU_CHECK_KERNEL

    if (zbl.enabled) {
      find_force_ZBL<<<
        (nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1,
        64,
        0,
        nep_data[gpu].stream>>>(
        paramb,
        nep_temp_data.num_atoms_per_gpu,
        zbl,
        nep_data[gpu].N1,
        nep_data[gpu].N2,
        box,
        nep_data[gpu].NN_angular.data(),
        nep_data[gpu].NL_angular.data(),
        nep_data[gpu].type.data(),
        nep_data[gpu].position.data(),
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].position.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].force.data(),
        nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu,
        nep_data[gpu].force.data() + nep_temp_data.num_atoms_per_gpu * 2,
        nep_data[gpu].virial.data(),
        nep_data[gpu].potential.data());
      GPU_CHECK_KERNEL
    }
  }

  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuSetDevice(gpu));
    CHECK(gpuDeviceSynchronize());
  }

  CHECK(gpuSetDevice(0));

  // serial
  for (int gpu = 0; gpu < paramb.num_gpus; ++gpu) {
    CHECK(gpuMemcpy(
      nep_temp_data.potential.data() + nep_data[gpu].N1,
      nep_data[gpu].potential.data() + nep_data[gpu].N1,
      sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1),
      gpuMemcpyDeviceToDevice));

    for (int d = 0; d < 3; ++d) {
      CHECK(gpuMemcpy(
        nep_temp_data.force.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        nep_data[gpu].force.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1),
        gpuMemcpyDeviceToDevice));
    }

    for (int d = 0; d < 9; ++d) {
      CHECK(gpuMemcpy(
        nep_temp_data.virial.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        nep_data[gpu].virial.data() + nep_data[gpu].N1 + nep_temp_data.num_atoms_per_gpu * d,
        sizeof(double) * (nep_data[gpu].N2 - nep_data[gpu].N1),
        gpuMemcpyDeviceToDevice));
    }

    collect_properties<<<(nep_data[gpu].N2 - nep_data[gpu].N1 - 1) / 64 + 1, 64>>>(
      N,
      nep_temp_data.num_atoms_per_gpu,
      nep_data[gpu].N1,
      nep_data[gpu].N2,
      nep_data[gpu].M1,
      nep_temp_data.cell_contents.data(),
      nep_temp_data.force.data(),
      nep_temp_data.potential.data(),
      nep_temp_data.virial.data(),
      force.data(),
      potential.data(),
      virial.data());
    GPU_CHECK_KERNEL
  }

  // add the long-range contributions (already in global atom order)
  add_long_range<<<(N - 1) / 64 + 1, 64>>>(
    N,
    lr_potential.data(),
    lr_force.data(),
    lr_virial.data(),
    potential.data(),
    force.data(),
    virial.data());
  GPU_CHECK_KERNEL
}

GPU_Vector<float>& NEP_Charge_MULTIGPU::get_charge_reference() { return charge_full; }

GPU_Vector<float>& NEP_Charge_MULTIGPU::get_bec_reference() { return bec_full; }

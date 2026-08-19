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

#pragma once
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "ewald.cuh"
#include "pppm.cuh"

struct NEP_Charge_MULTIGPU_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;

  GPU_Vector<int> type;
  GPU_Vector<double> position;
  GPU_Vector<double> force;
  GPU_Vector<double> potential;
  GPU_Vector<double> virial;

  GPU_Vector<float> D_real;
  GPU_Vector<float> charge;
  GPU_Vector<float> charge_derivative;
  GPU_Vector<float> bec;           // BEC
  GPU_Vector<double> charge_sum;   // partial sum of charge for this GPU

  /*
  M0   M1                   M2
  |----|--------------------|----|
  0    N1                   N2   N3
    N4                         N5
  using coordinate for [0, N3)
  compute neighbor list, descriptor, and partial force for [N4, N5)
  compute force for [N1 N2)
  */

  int N1, N2, N3, N4, N5; // for local system
  int M0, M1, M2;         // for global system
  gpuStream_t stream;
};

struct NEP_Charge_TEMP_Data {
  int num_atoms_per_gpu;
  std::vector<int> cell_count_sum_cpu;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> type;
  GPU_Vector<double> position;
  GPU_Vector<double> force;
  GPU_Vector<double> potential;
  GPU_Vector<double> virial;
  GPU_Vector<float> float_temp; // slice buffer for charge/bec/D_real gather/scatter
  GPU_Vector<double> charge_sum; // partial sums of charge from all GPUs
  std::vector<double> charge_sum_cpu;
};

class NEP_Charge_MULTIGPU : public Potential
{
public:
  using Potential::compute;

  struct ParaMB {
    int charge_mode = 0;
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    int num_gpus = 1;
    float rc_radial = 0.0f;     // radial cutoff
    float rc_angular = 0.0f;    // angular cutoff
    float rcinv_radial = 0.0f;  // inverse of the radial cutoff
    float rcinv_angular = 0.0f; // inverse of the angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;  // for nep3
    int basis_size_angular = 8; // for nep3
    int num_types_sq = 0;       // for nep3
    int num_c_radial = 0;       // for nep3
    int num_types = 0;
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* sqrt_epsilon_inf; // sqrt(epsilon_inf) related to BEC
    const float* b1;               // bias for the output layer
    const float* c;
    const float* q_scaler;
  };

  struct ZBL {
    bool enabled = false;
    bool flexibled = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct Charge_Para {
    int num_kpoints_max = 1;
    float alpha = 0.5f; // 1 / (2 Angstrom)
    float two_alpha_over_sqrt_pi = 0.564189583547756f;
    float A;
    float B;
  };

  NEP_Charge_MULTIGPU(
    const int num_gpus,
    const char* file_potential,
    const int num_atoms,
    const int partition_direction);
  virtual ~NEP_Charge_MULTIGPU(void);
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  GPU_Vector<float>& get_charge_reference();

  GPU_Vector<float>& get_bec_reference();

private:
  ParaMB paramb;
  ANN annmb[16];
  ZBL zbl;
  Charge_Para charge_para;
  Ewald ewald;
  PPPM pppm;
  NEP_Charge_MULTIGPU_Data nep_data[16];
  NEP_Charge_TEMP_Data nep_temp_data;

  // full-length data in global atom order, residing on GPU 0
  GPU_Vector<float> charge_full;
  GPU_Vector<float> bec_full;
  GPU_Vector<float> D_real_full;
  GPU_Vector<double> lr_force;
  GPU_Vector<double> lr_virial;
  GPU_Vector<double> lr_potential;

  int number_of_atoms = 0;
  int partition_direction = -1;

  void allocate_memory();
  void update_potential(float* parameters, ANN& ann);

  bool use_pppm = true; // use PPPM by default
  void check_ewald_pppm();
  void initialize_dftd3();
};

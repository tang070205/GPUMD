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
Add deposition functionality for MD simulation:
    - Dynamically add atoms during MD
    - Use type index (0, 1, 2...) directly, like RDF does
    
Usage: deposition <interval> <type1> <num1> [<type2> <num2> ...]
Example: deposition 1000 0 5 1 10
    - every 1000 steps, deposit 5 atoms of type 0 and 10 atoms of type 1
------------------------------------------------------------------------------*/

#pragma once

#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Deposition
{
public:
  void parse(const char** param, int num_param);
  
  void compute(int step, Atom& atom, Box& box);
  
  void finalize(void);

  bool is_active() const { return is_deposition; }

private:
  bool is_deposition = false;
  
  // Configuration parameters
  int deposit_interval = 1000;         // steps between deposition events
  double z_position_fraction = 0.75;   // z position as fraction of box (default 3/4)
  double vz_deposit = 0.001;          // z-velocity (negative for deposition)
  double vx_deposit = 0.0;             // x-velocity
  double vy_deposit = 0.0;             // y-velocity
  
  // Atoms to deposit per event
  int type_deposit = 0;                // type index to deposit
  int num_deposit = 0;                 // number of atoms per deposit
  double deposit_mass = 0.0;           // mass of the deposited atom
  
  // Statistics
  int num_deposited_total = 0;         // total number deposited so far
  int num_deposition_events = 0;       // number of deposition events performed

  std::vector<std::string> atom_symbols;
  
  // Helper functions
  void perform_deposition(Atom& atom, Box& box);
};

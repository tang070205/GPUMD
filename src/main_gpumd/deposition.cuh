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
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class Deposition
{
public:
  void parse(const char** param, int num_param);
  void compute(int step, Atom& atom, Box& box, std::vector<Group>& group);
  void finalize(void);
  bool is_active() const { return is_deposition; }

private:
  bool is_deposition = false;
  
  int deposit_interval = 1000;           // steps between deposition events
  int num_deposit = 0;                   // number of atoms per deposit
  int type_deposit = 0;                  // type index to deposit
  int num_deposited_total = 0;           // total number deposited atoms
  int num_deposition_events = 0;         // number of deposition events
  double deposit_mass = 0.0;             // mass of the deposited atom
  double vz_deposit = 0.0;               // z-velocity (negative for deposition)
  double z_position = 0.0;               // z position 
  std::vector<std::string> atom_symbols; // symbols of the atoms for poential
  std::vector<int> deposit_group;        // group labels for deposited atoms per grouping method

  void perform_deposition(Atom& atom, Box& box, std::vector<Group>& group);
};

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
Use finite difference to calculate the hessian (force constants).
    H_ij^ab = [F_i^a(-) - F_i^a(+)] / [u_j^b(+) - u_j^b(-)]
Then calculate the dynamical matrices with different k points.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "force/force_constant.cuh"
#include "hessian.cuh"
#include "utilities/common.cuh"
#include "utilities/cusolver_wrapper.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <cstring>
#include <map>
#include <cmath>
#include <sstream>
#include <string>

const std::map<std::string, double> table = {
  {"H", 1.0080000000}, {"He", 4.0026020000}, {"Li", 6.9400000000}, {"Be", 9.0121831000},
  {"B", 10.8100000000}, {"C", 12.0110000000}, {"N", 14.0070000000}, {"O", 15.9990000000},
  {"F", 18.9984031630}, {"Ne", 20.1797000000}, {"Na", 22.9897692800}, {"Mg", 24.3050000000},
  {"Al", 26.9815385000}, {"Si", 28.0850000000}, {"P", 30.9737619980}, {"S", 32.0600000000},
  {"Cl", 35.4500000000}, {"Ar", 39.9480000000}, {"K", 39.0983000000}, {"Ca", 40.0780000000},
  {"Sc", 44.9559080000}, {"Ti", 47.8670000000}, {"V", 50.9415000000}, {"Cr", 51.9961000000},
  {"Mn", 54.9380440000}, {"Fe", 55.8450000000}, {"Co", 58.9331940000}, {"Ni", 58.6934000000},
  {"Cu", 63.5460000000}, {"Zn", 65.3800000000}, {"Ga", 69.7230000000}, {"Ge", 72.6300000000},
  {"As", 74.9215950000}, {"Se", 78.9710000000}, {"Br", 79.9040000000}, {"Kr", 83.7980000000},
  {"Rb", 85.4678000000}, {"Sr", 87.6200000000}, {"Y", 88.9058400000}, {"Zr", 91.2240000000},
  {"Nb", 92.9063700000}, {"Mo", 95.9500000000}, {"Tc", 98}, {"Ru", 101.0700000000},
  {"Rh", 102.9055000000}, {"Pd", 106.4200000000}, {"Ag", 107.8682000000}, {"Cd", 112.4140000000},
  {"In", 114.8180000000}, {"Sn", 118.7100000000}, {"Sb", 121.7600000000}, {"Te", 127.6000000000},
  {"I", 126.9044700000}, {"Xe", 131.2930000000}, {"Cs", 132.9054519600}, {"Ba", 137.3270000000},
  {"La", 138.9054700000}, {"Ce", 140.1160000000}, {"Pr", 140.9076600000}, {"Nd", 144.2420000000},
  {"Pm", 145}, {"Sm", 150.3600000000}, {"Eu", 151.9640000000}, {"Gd", 157.2500000000},
  {"Tb", 158.9253500000}, {"Dy", 162.5000000000}, {"Ho", 164.9303300000}, {"Er", 167.2590000000},
  {"Tm", 168.9342200000}, {"Yb", 173.0450000000}, {"Lu", 174.9668000000}, {"Hf", 178.4900000000},
  {"Ta", 180.9478800000}, {"W", 183.8400000000}, {"Re", 186.2070000000}, {"Os", 190.2300000000},
  {"Ir", 192.2170000000}, {"Pt", 195.0840000000}, {"Au", 196.9665690000}, {"Hg", 200.5920000000},
  {"Tl", 204.3800000000}, {"Pb", 207.2000000000}, {"Bi", 208.9804000000}, {"Po", 210}, {"At", 210},
  {"Rn", 222}, {"Fr", 223}, {"Ra", 226}, {"Ac", 227}, {"Th", 232.0377000000}, {"Pa", 231.0358800000},
  {"U", 238.0289100000}, {"Np", 237}, {"Pu", 244}, {"Am", 243}, {"Cm", 247}, {"Bk", 247}, {"Cf", 251},
  {"Es", 252}, {"Fm", 257}, {"Md", 258}, {"No", 259}, {"Lr", 262}
};

void Hessian::compute(
  Force& force,
  Box& box,
  std::vector<std::string> cpu_atom_symbol,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  initialize(cpu_atom_symbol, box, type.size());
  find_H(
    force,
    box,
    cpu_position_per_atom,
    position_per_atom,
    type,
    group,
    potential_per_atom,
    force_per_atom,
    virial_per_atom);

  if (num_kpoints == 1) // currently for Alex's GKMA calculations
  {
    find_D(box, cpu_position_per_atom);
    find_eigenvectors();
  } else {
    find_dispersion(box, cpu_position_per_atom);
  }
}

void Hessian::create_basis(std::vector<std::string> cpu_atom_symbol, size_t N)
{
  std::ifstream fin("run.in");
  std::string key;
  if (fin >> key && key == "replicate")
    fin >> cx >> cy >> cz;
  this->num_basis = N / (cx * cy * cz);

  basis.resize(num_basis);
  mass.resize(num_basis);
  for (size_t i = 0; i < num_basis; ++i) {
    basis[i] = i;
    auto it = table.find(cpu_atom_symbol[i]);
    if (it == table.end()) {
      PRINT_INPUT_ERROR("Error: no such element << sym << \n");
    }
    mass[i] = it->second;
  }

  label.resize(N);
  for (size_t n = 0; n < N; ++n) {
    size_t atom = n % num_basis;
    label[n] = atom;
  }
}

void Hessian::create_kpoints(const Box& box)
{
  auto dot = [](const std::vector<double>& a, const std::vector<double>& b) -> double {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  };
  auto cross = [](const std::vector<double>& a, const std::vector<double>& b) -> std::vector<double> {
    return { a[1]*b[2] - a[2]*b[1],
             a[2]*b[0] - a[0]*b[2],
             a[0]*b[1] - a[1]*b[0] };
  };
  auto transpose = [](const std::vector<std::vector<double>>& m) -> std::vector<std::vector<double>> {
      return { { m[0][0], m[1][0], m[2][0] },
               { m[0][1], m[1][1], m[2][1] },
               { m[0][2], m[1][2], m[2][2] } };
  };
  auto matvec = [](const std::vector<std::vector<double>>& m, const std::vector<double>& v) -> std::vector<double> {
    return { m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
             m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
             m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2] };
    };
  auto lerp = [](const std::vector<double>& a, const std::vector<double>& b, double t) -> std::vector<double> {
    return { a[0] + t*(b[0] - a[0]),
             a[1] + t*(b[1] - a[1]),
             a[2] + t*(b[2] - a[2]) };
  };
  auto reciprocal_lattice = [&dot,&cross,&transpose](const std::vector<std::vector<double>>& lat) -> std::vector<std::vector<double>> {
    const double volume = dot(lat[0], cross(lat[1], lat[2]));
    std::vector<std::vector<double>> rec(3, std::vector<double>(3));
    rec[0] = cross(lat[1], lat[2]);
    rec[1] = cross(lat[2], lat[0]);
    rec[2] = cross(lat[0], lat[1]);
    for (auto& v : rec)
      for (auto& x : v) x *= 2.0 * M_PI / volume;
    return transpose(rec);
  };

  std::ifstream kin("kpoints.in");
  if (!kin) PRINT_INPUT_ERROR("Cannot open kpoints.in\n");
  std::vector<std::vector<std::vector<double>>> hsps;
  std::vector<std::vector<double>> hsp;
  std::string line;
  while (std::getline(kin, line)) {
    const auto beg = line.find_first_not_of(" \t\r\n");
    if (beg == std::string::npos) {
        if (!hsp.empty()) { hsps.push_back(hsp); hsp.clear(); }
        continue;
    }
    if (line[beg] == '#') continue;
    std::istringstream iss(line);
    double x, y, z;
    if (!(iss >> x >> y >> z)) break;
    hsp.emplace_back(std::vector<double>{x, y, z});
  }
  if (!hsp.empty()) hsps.push_back(hsp);
  num_kpoints = 1 - hsps.size();
  for (const auto& seg : hsps) num_kpoints += seg.size();
  kpath_sym.resize(num_kpoints);

  const std::vector<std::vector<double>> lattice = {
    { box.cpu_h[0] / cx, box.cpu_h[3] / cx, box.cpu_h[6] / cx },
    { box.cpu_h[1] / cy, box.cpu_h[4] / cy, box.cpu_h[7] / cy },
    { box.cpu_h[2] / cz, box.cpu_h[5] / cz, box.cpu_h[8] / cz }
  };
  const auto rec_lat = reciprocal_lattice(lattice);

  std::vector<double> num_interps;
  num_interps.reserve(num_kpoints - 1);
  for (const auto& seg : hsps) {
    for (size_t i = 1; i < seg.size(); ++i) {
      auto start = matvec(rec_lat, seg[i-1]);
      auto end = matvec(rec_lat, seg[i]);
      double dx = end[0] - start[0];
      double dy = end[1] - start[1];
      double dz = end[2] - start[2];
      double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
      num_interps.push_back(static_cast<int>(dist * 100.0));
    }
  }
  for (int n : num_interps) num_kpoints += n;

  kpoints.resize(num_kpoints * 3);
  kpath.resize(num_kpoints);
  std::vector<double> sym_idx;
  size_t k_idx = 0;
  size_t interp_idx = 0;
  double kpath_len = 0.0;

  auto k_first = matvec(rec_lat, hsps[0][0]);
  kpoints[k_idx * 3 + 0] = k_first[0];
  kpoints[k_idx * 3 + 1] = k_first[1];
  kpoints[k_idx * 3 + 2] = k_first[2];
  kpath[k_idx] = kpath_len;
  sym_idx.push_back(k_idx);
  ++k_idx;

  for (const auto& hsp : hsps) {
    for (size_t i = 1; i < hsp.size(); ++i) {
      const auto& start = matvec(rec_lat, hsp[i - 1]);
      const auto& end   = matvec(rec_lat, hsp[i]);
      int n = num_interps[interp_idx++] + 2;

      for (int j = 1; j < n - 1; ++j) {
        double t = static_cast<double>(j) / n;
        auto kpt = lerp(start, end, t);

        kpoints[k_idx * 3 + 0] = kpt[0];
        kpoints[k_idx * 3 + 1] = kpt[1];
        kpoints[k_idx * 3 + 2] = kpt[2];

        double dx = kpt[0] - kpoints[k_idx * 3 - 3];
        double dy = kpt[1] - kpoints[k_idx * 3 - 2];
        double dz = kpt[2] - kpoints[k_idx * 3 - 1];
        kpath_len += std::sqrt(dx * dx + dy * dy + dz * dz);
        kpath[k_idx] = kpath_len;
        ++k_idx;
      }

        // Add the end point
      kpoints[k_idx * 3 + 0] = end[0];
      kpoints[k_idx * 3 + 1] = end[1];
      kpoints[k_idx * 3 + 2] = end[2];

      double dx = end[0] - kpoints[k_idx * 3 - 3];
      double dy = end[1] - kpoints[k_idx * 3 - 2];
      double dz = end[2] - kpoints[k_idx * 3 - 1];
      kpath_len += std::sqrt(dx * dx + dy * dy + dz * dz);
      kpath[k_idx] = kpath_len;
      sym_idx.push_back(k_idx);
      ++k_idx;
    }
  }

  for (size_t kp = 0; kp < kpath_sym.size(); ++kp) {
    kpath_sym[kp] = kpath[sym_idx[kp]];
  }
}

void Hessian::initialize(std::vector<std::string> cpu_atom_symbol, const Box& box, size_t N)
{
  create_basis(cpu_atom_symbol, N);
  create_kpoints(box);
  size_t num_H = num_basis * N * 9;
  size_t num_D = num_basis * num_basis * 9 * num_kpoints;
  H.resize(num_H, 0.0);
  DR.resize(num_D, 0.0);
  if (num_kpoints > 1) // for dispersion calculation
  {
    DI.resize(num_D, 0.0);
  }
}

bool Hessian::is_too_far(
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  const size_t n1,
  const size_t n2)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double d12_square = x12 * x12 + y12 * y12 + z12 * z12;
  return (d12_square > (cutoff * cutoff));
}

void Hessian::find_H(
  Force& force,
  Box& box,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int number_of_atoms = type.size();

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }
      size_t offset = (nb * number_of_atoms + n2) * 9;
      find_H12(
        displacement,
        n1,
        n2,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom,
        force,
        H.data() + offset);
    }
  }
}

static void find_exp_ikr(
  const size_t n1,
  const size_t n2,
  const double* k,
  const Box& box,
  const std::vector<double>& cpu_position_per_atom,
  double& cos_kr,
  double& sin_kr)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;
  double x12 = cpu_position_per_atom[n2] - cpu_position_per_atom[n1];
  double y12 =
    cpu_position_per_atom[n2 + number_of_atoms] - cpu_position_per_atom[n1 + number_of_atoms];
  double z12 = cpu_position_per_atom[n2 + number_of_atoms * 2] -
               cpu_position_per_atom[n1 + number_of_atoms * 2];
  apply_mic(box, x12, y12, z12);
  double kr = k[0] * x12 + k[1] * y12 + k[2] * z12;
  cos_kr = cos(kr);
  sin_kr = sin(kr);
}

void Hessian::output_D()
{
  FILE* fid = fopen("D.out", "w");
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t n1 = 0; n1 < num_basis * 3; ++n1) {
      for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
        // cuSOLVER requires column-major
        fprintf(fid, "%g ", DR[offset + n1 + n2 * num_basis * 3]);
      }
      if (num_kpoints > 1) {
        for (size_t n2 = 0; n2 < num_basis * 3; ++n2) {
          // cuSOLVER requires column-major
          fprintf(fid, "%g ", DI[offset + n1 + n2 * num_basis * 3]);
        }
      }
      fprintf(fid, "\n");
    }
  }
  fclose(fid);
}

void Hessian::find_omega(FILE* fid, size_t offset, size_t nk)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  eig_hermitian_QR(dim, DR.data() + offset, DI.data() + offset, W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  fprintf(fid, "%.6f ", kpath[nk]);
  for (size_t n = 0; n < dim; ++n) {
    fprintf(fid, "%g ", W[n] * natural_to_THz);
  }
  fprintf(fid, "\n");
}

void Hessian::find_omega_batch(FILE* fid)
{
  size_t dim = num_basis * 3;
  std::vector<double> W(dim * num_kpoints);
  eig_hermitian_Jacobi_batch(dim, num_kpoints, DR.data(), DI.data(), W.data());
  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * dim;
    fprintf(fid, "%.6f ", kpath[nk]);
    for (size_t n = 0; n < dim; ++n) {
      fprintf(fid, "%g ", W[offset + n] * natural_to_THz);
    }
    fprintf(fid, "\n");
  }
}

void Hessian::find_dispersion(const Box& box, const std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  FILE* fid_omega2 = fopen("omega2.out", "w");
  fprintf(fid_omega2, "# ");
  for (size_t i = 0; i < kpath_sym.size(); ++i) {
    fprintf(fid_omega2, " %.6f", kpath_sym[i]);
    }
  fprintf(fid_omega2, "\n");
  for (size_t nk = 0; nk < num_kpoints; ++nk) {
    size_t offset = nk * num_basis * num_basis * 9;
    for (size_t nb = 0; nb < num_basis; ++nb) {
      size_t n1 = basis[nb];
      size_t label_1 = label[n1];
      double mass_1 = mass[label_1];
      for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
        if (is_too_far(box, cpu_position_per_atom, n1, n2))
          continue;
        double cos_kr, sin_kr;
        find_exp_ikr(n1, n2, kpoints.data() + nk * 3, box, cpu_position_per_atom, cos_kr, sin_kr);

        size_t label_2 = label[n2];
        double mass_2 = mass[label_2];
        double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
        double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
        for (size_t a = 0; a < 3; ++a) {
          for (size_t b = 0; b < 3; ++b) {
            size_t a3b = a * 3 + b;
            size_t row = label_1 * 3 + a;
            size_t col = label_2 * 3 + b;
            // cuSOLVER requires column-major
            size_t index = offset + col * num_basis * 3 + row;
            DR[index] += H12[a3b] * cos_kr * mass_factor;
            DI[index] += H12[a3b] * sin_kr * mass_factor;
          }
        }
      }
    }
    if (num_basis > 10) {
      find_omega(fid_omega2, offset, nk);
    } // > 32x32
  }
  output_D();
  if (num_basis <= 10) {
    find_omega_batch(fid_omega2);
  } // <= 32x32
  fclose(fid_omega2);
}

void Hessian::find_D(const Box& box, std::vector<double>& cpu_position_per_atom)
{
  const int number_of_atoms = cpu_position_per_atom.size() / 3;

  for (size_t nb = 0; nb < num_basis; ++nb) {
    size_t n1 = basis[nb];
    size_t label_1 = label[n1];
    double mass_1 = mass[label_1];
    for (size_t n2 = 0; n2 < number_of_atoms; ++n2) {
      if (is_too_far(box, cpu_position_per_atom, n1, n2)) {
        continue;
      }

      size_t label_2 = label[n2];
      double mass_2 = mass[label_2];
      double mass_factor = 1.0 / sqrt(mass_1 * mass_2);
      double* H12 = H.data() + (nb * number_of_atoms + n2) * 9;
      for (size_t a = 0; a < 3; ++a) {
        for (size_t b = 0; b < 3; ++b) {
          size_t a3b = a * 3 + b;
          size_t row = label_1 * 3 + a;
          size_t col = label_2 * 3 + b;
          // cuSOLVER requires column-major
          size_t index = col * num_basis * 3 + row;
          DR[index] += H12[a3b] * mass_factor;
        }
      }
    }
  }
}

void Hessian::find_eigenvectors()
{
  std::ofstream eigfile;
  eigfile.open("eigenvector.out", std::ios::out | std::ios::binary);

  size_t dim = num_basis * 3;
  std::vector<double> W(dim);
  std::vector<double> eigenvectors(dim * dim);
  eigenvectors_symmetric_Jacobi(dim, DR.data(), W.data(), eigenvectors.data());

  double natural_to_THz = 1.0e6 / (TIME_UNIT_CONVERSION * TIME_UNIT_CONVERSION);

  // output eigenvalues
  float om2;
  for (size_t n = 0; n < dim; n++) {
    om2 = (float)(W[n] * natural_to_THz);
    eigfile.write((char*)&om2, sizeof(float));
  }

  // output eigenvectors
  float eig;
  for (size_t col = 0; col < dim; col++) {
    for (size_t a = 0; a < 3; a++) {
      for (size_t b = 0; b < num_basis; b++) {
        size_t row = a + b * 3;
        // column-major order from cuSolver
        eig = (float)eigenvectors[row + col * dim];
        eigfile.write((char*)&eig, sizeof(float));
      }
    }
  }
  eigfile.close();
}

void Hessian::parse(const char** param, size_t num_param)
{
  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_phonon should have 2 parameters.\n");
  }
  // cutoff
  if (!is_valid_real(param[1], &cutoff)) {
    PRINT_INPUT_ERROR("cutoff for compute_phonon should be a number.\n");
  }
  if (cutoff <= 0) {
    PRINT_INPUT_ERROR("cutoff for compute_phonon should be positive.\n");
  }
  printf("Cutoff distance for compute_phonon = %g A.\n", cutoff);

  // displacement
  if (!is_valid_real(param[2], &displacement)) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be a number.\n");
  }
  if (displacement <= 0) {
    PRINT_INPUT_ERROR("displacement for compute_phonon should be positive.\n");
  }
  printf("displacement for compute_phonon = %g A.\n", displacement);
}

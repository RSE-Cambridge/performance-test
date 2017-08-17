// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include <set>
#include <string>
#include <dolfin.h>
#include <dolfin/io/XMLTable.h>
#include "pugixml.hpp"
#include "poisson_problem.h"
#include "elasticity_problem.h"
#include "mesh.h"

using namespace dolfin;

int main(int argc, char *argv[])
{
  SubSystemsManager::init_mpi();

  // Parse command line options (will intialise PETSc if any PETSc
  // options are present, e.g. --petsc.pc_type=jacobi)
  parameters.parse(argc, argv);

  // Intialise PETSc (if not already initialised when parsing
  // parameters)
  SubSystemsManager::init_petsc();

  // Default parameters
  Parameters application_parameters("application_parameters");
  application_parameters.add("problem_type", "poisson", {"poisson", "elasticity"});
  application_parameters.add("scaling_type", "weak", {"weak", "strong"});
  //application_parameters.add("pc", "BoomerAMG", {"BoomerAMG", "GAMG"});
  application_parameters.add("pc", "GAMG", {"BoomerAMG", "GAMG"});
  //application_parameters.add("ndofs", 640000);
  application_parameters.add("ndofs", 640);
  application_parameters.add("output", false);
  application_parameters.add("output_dir", "./out");
  application_parameters.add("system_name", "unknown");

  // Update from command line
  application_parameters.parse(argc, argv);

  // Extract parameters
  const std::string problem_type = application_parameters["problem_type"];
  const std::string scaling_type = application_parameters["scaling_type"];
  const std::string preconditioner = application_parameters["pc"];
  const std::size_t ndofs = application_parameters["ndofs"];
  const bool output = application_parameters["output"];
  const std::string output_dir = application_parameters["output_dir"];
  const std::string system_name = application_parameters["system_name"];

  // Set mesh partitioner
  parameters["mesh_partitioner"] = "SCOTCH";

  bool strong_scaling;
  if (scaling_type == "strong")
    strong_scaling = true;
  else if (scaling_type == "weak")
    strong_scaling = false;
  else
  {
    throw std::runtime_error("Scaling type '" + scaling_type + "` unknown");
    strong_scaling = true;
  }

  // Get number of processes
  const std::size_t num_processes = dolfin::MPI::size(MPI_COMM_WORLD);

  // Assemble problem
  std::shared_ptr<dolfin::PETScMatrix> A;
  std::shared_ptr<dolfin::PETScVector> b;
  std::shared_ptr<dolfin::Function> u;
  std::shared_ptr<const dolfin::Mesh> mesh;
  if (problem_type == "poisson")
  {
    Timer t0("[PERFORMANCE] Create Mesh");
    mesh = create_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1);
    t0.stop();

    // Create Poisson problem
    auto data = poisson::problem(mesh);
    A = std::get<0>(data);
    b = std::get<1>(data);
    u = std::get<2>(data);
  }
  else if (problem_type == "elasticity")
  {
    Timer t0("[PERFORMANCE] Create Mesh");
    mesh = create_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3);
    t0.stop();

    // Create elasticity problem. Near-nullspace will be attached to
    // the linear operator (matrix)
    auto data = elastic::problem(mesh);
    A = std::get<0>(data);
    b = std::get<1>(data);
    u = std::get<2>(data);
  }
  else
    throw std::runtime_error("Unknown problem type: " + problem_type);

  // Print simulation summary
  if (MPI::rank(mesh->mpi_comm()) == 0)
  {
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  Problem type:   "   << problem_type << std::endl;
    std::cout << "  Preconditioner: " << preconditioner << std::endl;
    std::cout << "  Scaling type:   "   << scaling_type << std::endl;
    std::cout << "  Num processes:  "  << num_processes << std::endl;
    std::cout << "  Total degrees of freedom:               " <<  u->function_space()->dim() << std::endl;
    std::cout << "  Average degrees of freedom per process: "
              << u->function_space()->dim()/MPI::size(mesh->mpi_comm()) << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
  }

  // Create solver
  PETScKrylovSolver solver(mesh->mpi_comm());
  solver.set_from_options();
  solver.set_operator(A);

  // Solve
  Timer t5("[PERFORMANCE] Solve");
  std::size_t num_iter = solver.solve(*u->vector(), *b);
  t5.stop();

  if (output)
  {
    Timer t6("[PERFORMANCE] Output");
    //  Save solution in XDMF format
    std::string filename = output_dir + "/solution-" + std::to_string(num_processes) + ".xdmf";
    XDMFFile file(filename);
    file.write(*u);
    t6.stop();
  }

  // Display timings
  list_timings(TimingClear::keep, {TimingType::wall});

  // Report number of Krylov iterations
  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;

  // FIXME: pugi::xml is included separately from DOLFIN here, because it is not
  // installed by DOLFIN at present. Need to be able to access nodes to add more
  // data
  pugi::xml_document doc;

  auto node = doc.append_child("bench");
  node.append_attribute("system_name") = system_name.c_str();
  node.append_attribute("num_processes") = (int)num_processes;
  node.append_attribute("problem_type") = problem_type.c_str();
  node.append_attribute("totaldofs") = std::to_string(u->function_space()->dim()).c_str();
  auto solver_subnode = node.append_child("solver");
  solver_subnode.append_attribute("pc") = preconditioner.c_str();
  solver_subnode.append_attribute("iteration_count") = (int)num_iter;
  auto dolfin_subnode = node.append_child("dolfin");
  char petsc_version[200];
  PetscGetVersion(petsc_version, 200);
  dolfin_subnode.append_attribute("petsc_version") = petsc_version;
  dolfin_subnode.append_attribute("dolfin_version") = dolfin_version().c_str();
  dolfin_subnode.append_attribute("dolfin_commit") = git_commit_hash().c_str();

  Table t = timings(TimingClear::clear, {TimingType::wall});
  Table t_max = MPI::max(mesh->mpi_comm(), t);
  XMLTable::write(t, node);

  // FIXME: output filename
  // ? optional/default, overwrite ?
  std::string xml_filename = "output.xml";

  if (MPI::rank(mesh->mpi_comm()) == 0)
    doc.save_file(xml_filename.c_str(), "  ");

  return 0;
}

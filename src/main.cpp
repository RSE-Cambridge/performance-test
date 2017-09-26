// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include <set>
#include <string>

#include<boost/property_tree/ptree.hpp>
#include<boost/property_tree/json_parser.hpp>
#include<boost/property_tree/xml_parser.hpp>

namespace pt = boost::property_tree;

#include <dolfin.h>
#include <dolfin/io/XMLTable.h>
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

  Timer whole_program("[PERFORMANCE] Whole Application");

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
  whole_program.stop();

  // Display timings
  list_timings(TimingClear::keep, {TimingType::wall});

  // Report number of Krylov iterations
  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;

  // Get timings and insert into boost::property_tree
  Table t = timings(TimingClear::keep, {TimingType::wall});
  Table t_max = dolfin::MPI::all_reduce(mesh->mpi_comm(), t, MPI_MAX);
  Table t_min = dolfin::MPI::all_reduce(mesh->mpi_comm(), t, MPI_MIN);

  // limit output to rank 0 only
  if (MPI::rank(mesh->mpi_comm()) == 0)
  {
    pt::ptree ptree, timers, libs, dolfin, petsc;

    // system information
    ptree.put("system.name",          system_name);
    ptree.put("system.num_processes", num_processes);
    ptree.put("system.hostname",      std::getenv("HOSTNAME"));
    // problem information
    ptree.put("problem.application",  "dolfin");
    ptree.put("problem.type",         problem_type);
    ptree.put("problem.total_dofs",   u->function_space()->dim());
    ptree.put("problem.preconditioner", preconditioner);
    // results of test
    ptree.put("results.num_iter",     num_iter);
    ptree.put("results.wall_avg",     t.get_value("[PERFORMANCE] Whole Application", "wall avg"));
    ptree.put("results.wall_max",     t_max.get_value("[PERFORMANCE] Whole Application", "wall avg"));
    ptree.put("results.wall_min",     t_min.get_value("[PERFORMANCE] Whole Application", "wall avg"));

    // save all [PERFORMANCE] timers
    // TODO: later on, this could be done using Table.rows fields
    // but it is private for now
    std::vector<std::string> performance_rows {
      "[PERFORMANCE] Whole Application",
      "[PERFORMANCE] Solve",
      "[PERFORMANCE] Assemble",
      "[PERFORMANCE] Create Mesh",
      "[PERFORMANCE] FunctionSpace",
    };
    if (output)
      performance_rows.push_back("[PERFORMANCE] Output");

    for (size_t i = 0; i < performance_rows.size(); i++)
    {
      pt::ptree result;
      result.put("name",      performance_rows[i]);
      result.put("wall_avg",  t.get_value(performance_rows[i], "wall avg"));
      result.put("wall_max",  t_max.get_value(performance_rows[i], "wall avg"));
      result.put("wall_min",  t_min.get_value(performance_rows[i], "wall avg"));
      // add to list of timers
      timers.push_back(std::make_pair("", result));
    }

    // add info about dolfin
    dolfin.put("name", "dolfin");
    dolfin.put("version", dolfin_version());
    dolfin.put("commit",  git_commit_hash());
    libs.push_back(std::make_pair("", dolfin));

    // add info about petsc
    char petsc_version[200];
    PetscGetVersion(petsc_version, 200);
    petsc.put("name", "petsc");
    petsc.put("version", petsc_version);
    libs.push_back(std::make_pair("", petsc));

    // add lists to tree
    ptree.add_child("timers", timers);
    ptree.add_child("libs",   libs);

    std::stringstream json;
    json << "-----------------------------------------------------------------------------" << std::endl;
    pt::write_json(json, ptree);
    json << "-----------------------------------------------------------------------------" << std::endl;
    std::cout << json.str();
  }

  return 0;
}

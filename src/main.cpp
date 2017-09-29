// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include <set>
#include <string>
#include <utility>

#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/common/timing.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/defines.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/parameter/GlobalParameters.h>
#include <dolfin/parameter/Parameters.h>

#include "elasticity_problem.h"
#include "poisson_problem.h"
#include "mesh.h"

#include "HPCReport.h"

using namespace dolfin;

int main(int argc, char *argv[])
{
  dolfin::SubSystemsManager::init_mpi();

  // Parse command line options (will intialise PETSc if any PETSc
  // options are present, e.g. --petsc.pc_type=jacobi)
  dolfin::parameters.parse(argc, argv);

  // Intialise PETSc (if not already initialised when parsing
  // parameters)
  dolfin::SubSystemsManager::init_petsc();

  // Default parameters
  dolfin::Parameters application_parameters("application_parameters");
  application_parameters.add("problem_type", "poisson", {"poisson", "elasticity"});
  application_parameters.add("scaling_type", "weak", {"weak", "strong"});
  application_parameters.add("ndofs", 640);
  application_parameters.add("output", false);
  application_parameters.add("output_dir", "./out");
  application_parameters.add("system_name", "unknown");

  // Update from command line
  application_parameters.parse(argc, argv);

  // Extract parameters
  const std::string problem_type = application_parameters["problem_type"];
  const std::string scaling_type = application_parameters["scaling_type"];
  const std::size_t ndofs = application_parameters["ndofs"];
  const bool output = application_parameters["output"];
  const std::string output_dir = application_parameters["output_dir"];
  const std::string system_name = application_parameters["system_name"];

  // Set mesh partitioner
  dolfin::parameters["mesh_partitioner"] = "SCOTCH";

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

  dolfin::Timer whole_program("[PERFORMANCE] Whole Application");

  // Get number of processes
  const std::size_t num_processes = dolfin::MPI::size(MPI_COMM_WORLD);

  // Assemble problem
  std::shared_ptr<dolfin::PETScMatrix> A;
  std::shared_ptr<dolfin::PETScVector> b;
  std::shared_ptr<dolfin::Function> u;
  std::shared_ptr<const dolfin::Mesh> mesh;
  if (problem_type == "poisson")
  {
    dolfin::Timer t0("[PERFORMANCE] Create Mesh");
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
    dolfin::Timer t0("[PERFORMANCE] Create Mesh");
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
  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
  {
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  Problem type:   "   << problem_type << std::endl;
    std::cout << "  Scaling type:   "   << scaling_type << std::endl;
    std::cout << "  Num processes:  "  << num_processes << std::endl;
    std::cout << "  Total degrees of freedom:               " <<  u->function_space()->dim() << std::endl;
    std::cout << "  Average degrees of freedom per process: "
              << u->function_space()->dim()/dolfin::MPI::size(mesh->mpi_comm()) << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;
  }

  // Create solver
  dolfin::PETScKrylovSolver solver(mesh->mpi_comm());
  solver.set_from_options();
  solver.set_operator(A);

  // Solve
  dolfin::Timer t5("[PERFORMANCE] Solve");
  std::size_t num_iter = solver.solve(*u->vector(), *b);
  t5.stop();

  if (output)
  {
    dolfin::Timer t6("[PERFORMANCE] Output");
    //  Save solution in XDMF format
    std::string filename = output_dir + "/solution-" + std::to_string(num_processes)
      + ".xdmf";
    dolfin::XDMFFile file(filename);
    file.write(*u);
    t6.stop();
  }
  whole_program.stop();

  // Display timings
  list_timings(dolfin::TimingClear::keep, {dolfin::TimingType::wall});

  // Report number of Krylov iterations
  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;

  // Get timings and insert into boost::property_tree
  dolfin::Table t = dolfin::timings(TimingClear::clear, {dolfin::TimingType::wall});
  dolfin::Table t_max = dolfin::MPI::all_reduce(mesh->mpi_comm(), t, MPI_MAX);
  dolfin::Table t_min = dolfin::MPI::all_reduce(mesh->mpi_comm(), t, MPI_MIN);

  // limit output to rank 0 only
  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
  {
    // for now manually list timers we want to include in report
    std::vector<std::string> timers{
        "[PERFORMANCE] Whole Application",    // first timer is used for result section
        "[PERFORMANCE] Solve",
        "[PERFORMANCE] Assemble",
        "[PERFORMANCE] Create Mesh",
        "[PERFORMANCE] FunctionSpace",
    };
    if (output)
      timers.emplace_back("[PERFORMANCE] Output");

    // report json file
    HPCReport report("Dolfin");
    report.put_problem(HPCREPORT_PROBLEM_TYPE, problem_type)
        .put_problem(HPCREPORT_PROBLEM_SIZE, u->function_space()->dim());

    // add system info
    report.put_system(HPCREPORT_SYSTEM_CPUS, num_processes)
        .put_system(HPCREPORT_SYSTEM_MACHINE, system_name);

    // add result info
    report.put_result("num_iter", num_iter)
        .put_result(HPCREPORT_RESULT_WALL_AVG, t.get_value(timers[0], "wall avg"))
        .put_result(HPCREPORT_RESULT_WALL_MAX, t_max.get_value(timers[0], "wall avg"))
        .put_result(HPCREPORT_RESULT_WALL_MIN, t_min.get_value(timers[0], "wall avg"));

    // add dolfin lib and petsc lib
    report.add_lib("dolfin")
        .put(HPCREPORT_LIBRARY_VERSION, dolfin::dolfin_version())
        .put(HPCREPORT_LIBRARY_COMMIT, dolfin::git_commit_hash());

    char petsc_version[200];
    PetscGetVersion(petsc_version, 200);
    report.add_lib("petsc")
        .put(HPCREPORT_LIBRARY_VERSION, petsc_version);

    // add selected timers info
    for (auto & timer : timers)
    {
      report.add_frame(timer)
          .put(HPCREPORT_FRAME_WALL_AVG, t.get_value(timer, "wall avg"))
          .put(HPCREPORT_FRAME_WALL_MAX, t_max.get_value(timer, "wall avg"))
          .put(HPCREPORT_FRAME_WALL_MIN, t_min.get_value(timer, "wall avg"));
    }


    // generate json file and print it to standard stream
    std::stringstream json;
    json << "-----------------------------------------------------------------------------" << std::endl;
    json << report << std::endl;
    json << "-----------------------------------------------------------------------------" << std::endl;
    std::cout << json.str();
  }

  return 0;
}

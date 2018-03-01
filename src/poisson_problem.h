// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <memory>
#include <utility>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/function/Constant.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>

#include "Poisson.h"

namespace poisson
{
  // Source term (right-hand side)
  class Source : public dolfin::function::Expression
  {
  public:
    void eval(Eigen::Ref<EigenRowMatrixXd> values,
              Eigen::Ref<const EigenRowMatrixXd> x) const
    {
      for (unsigned int i = 0; i != x.rows(); ++i)
      {
        double dx = x(i, 0) - 0.5;
        double dy = x(i, 1) - 0.5;
        values(i, 0) = 10*exp(-(dx*dx + dy*dy)/0.02);
      }
    }
  };

  // Normal derivative (Neumann boundary condition)
  class dUdN : public dolfin::function::Expression
  {
  public:
    void eval(Eigen::Ref<EigenRowMatrixXd> values,
              Eigen::Ref<const EigenRowMatrixXd> x) const
    {
      for (unsigned int i = 0; i != x.rows(); ++i)
        values(i, 0) = sin(5*x(i, 0));
    }
  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public dolfin::mesh::SubDomain
  {
  public:
    EigenVectorXb inside(Eigen::Ref<const EigenRowMatrixXd> x, bool on_boundary) const
    {
      EigenVectorXb result(x.rows());
      for (unsigned int i = 0; i != x.rows(); ++i)
        result[i] = (x(i, 0) < DOLFIN_EPS or x(i, 0) > (1.0 - DOLFIN_EPS));
      return result;
    }
  };

  std::tuple<std::shared_ptr<dolfin::la::PETScMatrix>,
    std::shared_ptr<dolfin::la::PETScVector>,
    std::shared_ptr<dolfin::function::Function>>
    problem(std::shared_ptr<const dolfin::mesh::Mesh> mesh)
  {
    dolfin::common::Timer t0("[PERFORMANCE] FunctionSpace");
    auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
    t0.stop();

    dolfin::common::Timer t1("[PERFORMANCE] Assemble");

    // Define boundary condition
    auto u0 = std::make_shared<dolfin::function::Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    auto bc = std::make_shared<dolfin::fem::DirichletBC>(V, u0, boundary);

    // Define variational forms
    auto a = std::make_shared<Poisson::BilinearForm>(V, V);
    auto L = std::make_shared<Poisson::LinearForm>(V);

    // Attach coefficients
    auto f = std::make_shared<Source>();
    auto g = std::make_shared<dUdN>();
    L->f = f;
    L->g = g;

    // Create assembler
    dolfin::fem::SystemAssembler assembler(a, L, {bc});

    // Assemble system
    auto A = std::make_shared<dolfin::la::PETScMatrix>();
    auto b = std::make_shared<dolfin::la::PETScVector>();
    assembler.assemble(*A, *b);

    t1.stop();

    // Create Function to hold solution
    auto u = std::make_shared<dolfin::function::Function>(V);

    return std::tuple<std::shared_ptr<dolfin::la::PETScMatrix>,
      std::shared_ptr<dolfin::la::PETScVector>,
      std::shared_ptr<dolfin::function::Function>>(A, b, u);
  }
}

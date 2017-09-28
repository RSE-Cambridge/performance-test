// Copyright (C) 2017 Jan Hybs
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.


/// This is a demo file demonstrating usage of HPCReport lib

#include <iostream>
#include "HPCReport.h"


int main(int argc, char ** argv, char ** envp)
{
    HPCReport report = HPCReport("Demo");

    report.add_lib("Petsc")
            .put(HPCREPORT_LIBRARY_VERSION, "1.63.0")
            .put("custom-field", true);

    report.add_lib("Dolfin")
            .put(HPCREPORT_LIBRARY_VERSION, "2017.2.0.dev0")
            .put(HPCREPORT_LIBRARY_COMMIT, "f43db07459c1acbb4a13e600082dd7a70bc4c4a5");


    report.add_frame("Frame 1")
            .put(HPCREPORT_FRAME_WALL_SUM, 5.0)
            .put(HPCREPORT_FRAME_WALL_MIN, 1.0)
            .put(HPCREPORT_FRAME_WALL_MAX, 4.0);

    report.add_frame("Frame 2")
            .put(HPCREPORT_FRAME_WALL_AVG, 4.0)
            .add_frame("Sub frame")
            .put(HPCREPORT_FRAME_WALL_AVG, 1.0);

    report.put_problem(HPCREPORT_PROBLEM_TYPE, "Poisson")
            .put_problem(HPCREPORT_PROBLEM_SIZE, 4096);

    std::cout << report << std::endl;
    return 0;
}

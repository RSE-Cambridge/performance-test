// Copyright (C) 2017 Jan Hybs
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include "HPCReport.h"


HPCReport::Library & HPCReport::add_lib(const std::string & name)
{
    std::shared_ptr<HPCReport::Library> lib = std::make_shared<HPCReport::Library>(name);
    this->libs.push_back(lib);
    return * lib;
}

HPCReport::Frame & HPCReport::add_frame(const std::string & name)
{
    std::shared_ptr<HPCReport::Frame> frame = std::make_shared<HPCReport::Frame>(name);
    this->frames.push_back(frame);
    return * frame;
}

std::ostream & operator<<(std::ostream & os, const HPCReport & report)
{
    pt::ptree tree, libs, frames;
    for (auto & lib : report.libs)
    {
        libs.push_back(std::make_pair("", lib->tree));
    }
    for (auto & frame : report.frames)
    {
        frames.push_back(std::make_pair("", frame->get_tree()));
    }
    tree.add_child("libs", libs);
    tree.add_child("frames", frames);
    tree.add_child("system", report.system);
    tree.add_child("problem", report.problem);
    tree.add_child("result", report.result);

    pt::write_json(os, tree);
    return os;
}

HPCReport::HPCReport(const std::string & application) : application(application)
{
    this->put_problem(HPCREPORT_SYSTEM_APPLICATION, application);


    // try to get hostname value
    char hostname[HOST_NAME_MAX];
    if (gethostname(hostname, HOST_NAME_MAX) == 0)
    {
        this->put_system(HPCREPORT_SYSTEM_HOSTNAME, hostname);
    }

    // try to get username value
    char * username = getlogin();
    if (username)
    {
        this->put_system(HPCREPORT_SYSTEM_USERNAME, username);
    }

    // try to get os name, variable should return something like
    //      x86_64-redhat-linux-gnu
    // or
    //      x86_64-pc-linux-gnu
    char * machtype = getenv("MACHTYPE");
    if (machtype)
    {
        this->put_system(HPCREPORT_SYSTEM_MACHINE, machtype);
    }
}



HPCReport::Frame & HPCReport::Frame::add_frame(const std::string & name)
{
    std::shared_ptr<HPCReport::Frame> frame = std::make_shared<HPCReport::Frame>(name);
    this->frames.push_back(frame);
    return * frame;
}

pt::ptree HPCReport::Frame::get_tree()
{
    if (frames.empty())
        return pt::ptree(tree);

    pt::ptree _frames;
    pt::ptree _tree(tree);

    for (auto & frame : frames)
    {
        _frames.push_back(std::make_pair("", frame->get_tree()));
    }
    _tree.add_child("frames", _frames);
    return _tree;
}

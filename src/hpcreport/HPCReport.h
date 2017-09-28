// Copyright (C) 2017 Jan Hybs
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.


#ifndef HPCREPORT_HPCREPORT_H
#define HPCREPORT_HPCREPORT_H

// Library constants
#define HPCREPORT_LIBRARY_NAME          "name"
#define HPCREPORT_LIBRARY_VERSION       "version"
#define HPCREPORT_LIBRARY_COMMIT        "commit"

// Frame constants
#define HPCREPORT_FRAME_NAME            "name"
#define HPCREPORT_FRAME_WALL_AVG        "wall_avg"
#define HPCREPORT_FRAME_WALL_MIN        "wall_min"
#define HPCREPORT_FRAME_WALL_MAX        "wall_max"
#define HPCREPORT_FRAME_WALL_SUM        "wall_sum"

// Result constants
#define HPCREPORT_RESULT_WALL_AVG        "wall_avg"
#define HPCREPORT_RESULT_WALL_MIN        "wall_min"
#define HPCREPORT_RESULT_WALL_MAX        "wall_max"
#define HPCREPORT_RESULT_WALL_SUM        "wall_sum"

// System constants
#define HPCREPORT_SYSTEM_APPLICATION    "application"
#define HPCREPORT_SYSTEM_CPUS           "num_cpus"
#define HPCREPORT_SYSTEM_HOSTNAME       "host"
#define HPCREPORT_SYSTEM_USERNAME       "user"
#define HPCREPORT_SYSTEM_MACHINE        "machine"

// Problem constants
#define HPCREPORT_PROBLEM_TYPE          "type"
#define HPCREPORT_PROBLEM_SIZE          "size"

#include <iostream>
#include <vector>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace pt = boost::property_tree;


/// Class for generating json report file containing
/// HPC and benchmark run details.
class HPCReport
{

public:
    /// Abstract class which stores additional ptrees not present in the main tree
    class Element
    {
    public:

        friend std::ostream & operator<<(std::ostream & os, const HPCReport & report);

    protected:
        /// Default constructor creates instance with given name
        /// \param name
        explicit Element(const std::string & _name) : name(_name)
        {};

        /// boost ptree holding fields
        pt::ptree tree;
        /// required name for a Element
        std::string name;
    };

    /// Class which stores information about Libraries
    class Library : public Element
    {
    public:
        /// Only constructor with name exist
        explicit Library(const std::string & _name) : Element(_name)
        {
            this->put(HPCREPORT_LIBRARY_NAME, name);
        };

        /// Puts a field to the object
        /// \tparam Type any type which can be inserted into ptree
        /// \param key name of the field
        /// \param value any value of the field
        /// \return reference to itself
        template<class Type>
        Library & put(const std::string & key, const Type & value)
        {
            this->tree.put(key, value);
            return * this;
        }
    };

    /// Class which stores information about Frames
    /// class Frame can also contain other Frames to allow nested hierarchy
    class Frame : public Element
    {
    public:
        /// Only constructor with name exist
        explicit Frame(const std::string & _name) : Element(_name)
        {
            this->put(HPCREPORT_FRAME_NAME, name);
        };

        /// Puts a field to the object
        /// \tparam Type any type which can be inserted into ptree
        /// \param key name of the field
        /// \param value any value of the field
        /// \return reference to itself
        template<class Type>
        Frame & put(const std::string & key, const Type & value)
        {
            this->tree.put(key, value);
            return * this;
        }

        /// Creates new frame in current frame
        /// \param name name of the frame
        /// \return reference to newly created Frame
        Frame & add_frame(const std::string & name);

        /// Recursive function will collect all frames
        /// from the current frame level
        /// \return ptree containing current frame and all its sub frames
        pt::ptree get_tree();

    private:
        /// vector of sub level Frames
        std::vector<std::shared_ptr<Frame>> frames;
    };


    /// Puts a field to a system section
    /// \tparam Type any type which can be used as value in boost ptree
    /// \param key name of the field
    /// \param value value of the field
    /// \return pointer to itself
    template<class Type>
    HPCReport & put_system(const std::string & key, const Type & value)
    { return this->put(system, key, value); }

    /// Puts a field to a result section
    /// \tparam Type any type which can be used as value in boost ptree
    /// \param key name of the field
    /// \param value value of the field
    /// \return pointer to itself
    template<class Type>
    HPCReport & put_result(const std::string & key, const Type & value)
    { return this->put(result, key, value); }

    /// Puts a field to a problem section
    /// \tparam Type any type which can be used as value in boost ptree
    /// \param key name of the field
    /// \param value value of the field
    /// \return pointer to itself
    template<class Type>
    HPCReport & put_problem(const std::string & key, const Type & value)
    { return this->put(problem, key, value); }

    /// Adds new Library to this report
    /// \param name name of the library
    /// \return pointer to newly created Library
    Library & add_lib(const std::string & name);

    /// Adds new Frame to this report
    /// \param name name of the frame
    /// \return pointer to newly created Frame
    Frame & add_frame(const std::string & name);

    /// Stream operator
    /// \param os
    /// \param report
    /// \return
    friend std::ostream & operator<<(std::ostream & os, const HPCReport & report);

    // ----------------------------------------------------
public:
    /// Only constructor is with an application name
    explicit HPCReport(const std::string & name);

private:
    // required name of the application
    std::string application;
    /// ptree sections
    pt::ptree system, problem, result;
    /// vector of Libraries
    std::vector<std::shared_ptr<Library>> libs;
    /// vector of global level Frames
    std::vector<std::shared_ptr<Frame>> frames;

    /// Puts an item into collection
    /// \tparam Type any type which can be used as value in boost ptree
    /// \param collection which ptree to insert to
    /// \param key name of the field
    /// \param value value of the field
    /// \return pointer to itself
    template<class Type>
    HPCReport & put(pt::ptree & collection, const std::string & key, const Type & value)
    {
        collection.put(key, value);
        return * this;
    }
};


#endif //HPCREPORT_HPCREPORT_H

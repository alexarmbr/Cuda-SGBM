#ifndef obj_structures
#define obj_structures

#include<iostream>
#include<string>
#include<sstream>
#include<vector>

struct Vertex {
    double x;
    double y;
    double z;
};

struct Face {
    int v1;
    int v2;
    int v3;
};

struct Object {
    std::vector<Vertex> vertices;
    std::vector<Face>   faces;
};

// Returns an Object struct of a particular file.
Object read_obj_file(std::string filename);

float * vertex_array_from_obj(Object obj);
Object obj_from_vertex_array(float* array, int num_points, int point_dim, Object face_obj);

// Outputs a particular object struct to a output stream.
void print_obj_data(Object obj, std::ostream& stream);
void print_obj_vertices(Object obj, std::ostream& stream);

#endif

#include<fstream>
#include<cstdlib>
#include "obj_structures.h"

using namespace std;

Object read_obj_file(string filename) {
    ifstream obj_file (filename);
    vector<Vertex> vertices;
    vector<Face> faces;
    string line;

    while (!obj_file.eof()) {
        getline(obj_file, line);
        if (obj_file.eof()) { break; }

        char type;
        istringstream iss(line);
        iss >> type;

        if (type == 'v') {
            Vertex temp_v;
            iss >> temp_v.x >> temp_v.y >> temp_v.z;
            // Notice that the i'th row, is the i'th element in the vector
            vertices.push_back(temp_v);

        } else if (type == 'f') {
            Face temp_f;
            int v1_loc, v2_loc, v3_loc;
            iss >> v1_loc >> v2_loc >> v3_loc;
            // Undo 1-indexing
            temp_f.v1 = v1_loc - 1;
            temp_f.v2 = v2_loc - 1;
            temp_f.v3 = v3_loc - 1;
            faces.push_back(temp_f);

        } else {
            cout << "A parsing error occurred while reading in "
                 << "the .obj file" << filename << endl;
            throw "Parsing error";
        }
    }
    Object obj;
    obj.vertices = vertices;
    obj.faces = faces;
    return obj;
}

float * vertex_array_from_obj(Object obj)
{
    int vertex_count = obj.vertices.size();
    // Allocate space for 4 floats per vertex: (x, y, z, 1)
    float * arr = (float *) malloc(sizeof(float) * vertex_count * 4);
    for (int i = 0; i < vertex_count; i++)
    {
        Vertex v = obj.vertices[i];
	      arr[i] = v.x;
	      arr[1 * vertex_count + i] = v.y;
        arr[2 * vertex_count + i] = v.z;
        arr[3 * vertex_count + i] = 1.0;
    }
   return arr;
}

Object obj_from_vertex_array(float* array, int num_points, int point_dim, Object face_obj)
{
   vector<Vertex> vertices;
   for (int i = 0; i < num_points; i++) {
       Vertex temp;
       temp.x = array[i * point_dim + 0];
       temp.y = array[i * point_dim + 1];
       temp.z = array[i * point_dim + 2];
       vertices.push_back(temp);
    }

   Object obj;
   obj.vertices = vertices;
   obj.faces = face_obj.faces;
   return obj;
}

/*Should be able to stream the output to a file.*/
void print_obj_data(Object obj, ostream& stream) {
    ostringstream stored_oss;
    for (auto &vertex : obj.vertices) {
        stream << "v " << vertex.x << ' ' << vertex.y << ' '
                       << vertex.z << endl;
    }
    for (auto &face : obj.faces) {
        stored_oss << "f " << (face.v1 + 1) << ' ' << (face.v2 + 1) << ' '
                           << (face.v3 + 1) << endl;
    }
    stream << stored_oss.str() << endl;
    return;
}

void print_obj_vertices(Object obj, ostream& stream) {
    ostringstream stored_oss;
    for (auto &vertex : obj.vertices) {
        stream << "v " << vertex.x << ' ' << vertex.y << ' '
                       << vertex.z << endl;
    }
    stream << stored_oss.str() << endl;
    return;
}

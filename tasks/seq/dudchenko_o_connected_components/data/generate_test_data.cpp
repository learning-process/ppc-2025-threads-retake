#include <fstream>
#include <vector>

void generate_test_image(const std::string& filename, int width, int height, 
                        const std::vector<int>& data) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&height), sizeof(int));
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int));
}
#include "planner/planner_utils.hpp"
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// La resolución alta es crucial para el rendimiento
cv::Mat1b inflated_map;
double resolution = 1000.0;

double round2(double v) {
    return round(v * 40.0) / 40.0;
}

double normalize_angle(double theta) {
    return fmod(theta + CV_PI, 2 * CV_PI) - CV_PI;
}

// Exactamente igual al código de referencia para máxima compatibilidad
inline bool is_collision(double x, double y) {
    int row = inflated_map.rows - int(y * resolution) - 1;
    int col = int(x * resolution);
    if (row >= 0 && row < inflated_map.rows && col >= 0 && col < inflated_map.cols)
        return inflated_map(row, col) != 0;
    return true;
}

// Optimizada para el rendimiento pero manteniendo la precisión
bool is_path_collision(double x1, double y1, double x2, double y2) {
    double dist = hypot(x2 - x1, y2 - y1);
    
    // Usar validation_distance exactamente como en el código de referencia
    const double validation_distance = 0.05;
    int steps = max(int(dist / validation_distance), 10);
    
    // Precalcular incrementos para evitar multiplicaciones en cada paso
    double dx = x2 - x1;
    double dy = y2 - y1;
    double step_x = dx / steps;
    double step_y = dy / steps;
    
    double xi = x1;
    double yi = y1;
    
    for (int i = 0; i <= steps; ++i) {
        if (is_collision(xi, yi)) return true;
        xi += step_x;
        yi += step_y;
    }
    return false;
}

bool trajectory_has_reverse(const vector<State>& path) {
    for (size_t i = 1; i < path.size(); ++i) {
        auto [x1, y1, t1] = path[i - 1];
        auto [x2, y2, t2] = path[i];
        double dx = x2 - x1;
        double dy = y2 - y1;
        double heading = atan2(dy, dx);
        double angle_diff = normalize_angle(heading - t1);
        if (fabs(angle_diff) > CV_PI / 2.0) return true;
    }
    return false;
}
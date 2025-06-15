#include "planner/planner_utils.hpp"
#include <unordered_map>
#include <queue>
#include <cmath>
#include <rclcpp/rclcpp.hpp>


using namespace std;
using namespace cv;

struct Motion {
    int v;
    double delta;
    double tan_delta;
};

struct StateHash {
    size_t operator()(const State& s) const {
        auto [x, y, theta] = s;
        size_t h1 = hash<double>()(x);
        size_t h2 = hash<double>()(y);
        size_t h3 = hash<double>()(theta);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

std::vector<State> hybrid_astar(State start, const Goal& goal) {
    // Diagnóstico de inicio
    auto [start_x, start_y, start_theta] = start;
    auto [goal_x, goal_y] = goal.pos;
    RCLCPP_INFO(rclcpp::get_logger("hybrid_astar"), 
                "Iniciando planificación: Start(%.3f, %.3f, %.3f) -> Goal(%.3f, %.3f, %.3f)",
                start_x, start_y, start_theta, goal_x, goal_y, goal.theta);
    
    // Verificar si el inicio está en colisión
    if (is_collision(start_x, start_y)) {
        RCLCPP_ERROR(rclcpp::get_logger("hybrid_astar"), 
                     "Punto inicial en colisión (%.3f, %.3f). Intentando pequeños ajustes...", start_x, start_y);
        
        // Intentar ajustes pequeños si el punto inicial está en colisión
        const double offsets[] = {0.01, 0.02, 0.03, 0.05};
        bool fixed = false;
        
        for (double offset : offsets) {
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) continue;
                    double new_x = start_x + dx * offset;
                    double new_y = start_y + dy * offset;
                    if (!is_collision(new_x, new_y)) {
                        RCLCPP_INFO(rclcpp::get_logger("hybrid_astar"), 
                                   "Ajustado inicio a (%.3f, %.3f)", new_x, new_y);
                        start_x = new_x;
                        start_y = new_y;
                        start = {start_x, start_y, start_theta};
                        fixed = true;
                        break;
                    }
                }
                if (fixed) break;
            }
            if (fixed) break;
        }
        
        if (!fixed) {
            RCLCPP_ERROR(rclcpp::get_logger("hybrid_astar"), 
                         "No se pudo ajustar el inicio, imposible planificar");
            return {};
        }
    }
    
    // Verificar si el objetivo está en colisión
    if (is_collision(goal_x, goal_y)) {
        RCLCPP_ERROR(rclcpp::get_logger("hybrid_astar"), 
                     "Meta en colisión (%.3f, %.3f). Planificación imposible.", goal_x, goal_y);
        return {};
    }
    
    // Parámetros de planificación optimizados como en el código de referencia
    const double step_size = 0.25;
    const double min_turning_radius = 0.6;
    const bool allow_reverse = true;
    const double REVERSE_COST = 5.0;
    const double FORWARD_COST = 1.0;
    const double SWITCH_DIRECTION_COST = 0.8;
    const int MAX_EXPANSIONS = 1000000; // Límite de expansiones para evitar cálculos excesivos

    // Creación de movimientos posibles
    vector<Motion> motions;
    for (int i = 0; i < 7; ++i) {
        double delta = -CV_PI / 4 + i * (CV_PI / 12);
        double tan_delta = tan(delta);
        motions.push_back({1, delta, tan_delta});
        if (allow_reverse)
            motions.push_back({-1, delta, tan_delta});
    }

    priority_queue<pair<double, State>, vector<pair<double, State>>, greater<>> open_list;
    unordered_map<State, State, StateHash> came_from;
    unordered_map<State, double, StateHash> cost_so_far;
    unordered_map<State, int, StateHash> direction_map;

    open_list.emplace(0.0, start);
    cost_so_far[start] = 0.0;
    direction_map[start] = 1;
    
    int expansions = 0;
    int collision_count = 0;

    while (!open_list.empty()) {
        auto [_, current] = open_list.top(); open_list.pop();
        auto [x, y, theta] = current;
        
        expansions++;
        
        if (expansions % 1000 == 0 || expansions <= 10) {
            RCLCPP_INFO(rclcpp::get_logger("hybrid_astar"), 
                        "Expansión %d, posición (%.3f, %.3f), cola: %zu, colisiones: %d", 
                        expansions, x, y, open_list.size(), collision_count);
        }
        
        if (expansions > MAX_EXPANSIONS) {
            RCLCPP_WARN(rclcpp::get_logger("hybrid_astar"), "Límite de expansiones alcanzado: %d", MAX_EXPANSIONS);
            return {};
        }

        if (hypot(goal.pos.x - x, goal.pos.y - y) < 0.1) {
            vector<State> path{current};
            while (came_from.count(current)) {
                current = came_from[current];
                path.push_back(current);
            }
            reverse(path.begin(), path.end());
            
            RCLCPP_INFO(rclcpp::get_logger("hybrid_astar"), "Ruta encontrada con %zu puntos en %d expansiones", path.size(), expansions);
            return path;
        }

        // Contador de movimientos explorados
        int valid_moves = 0;
        
        for (const auto& m : motions) {
            double omega = m.v / min_turning_radius * m.tan_delta;
            double x_new = x + m.v * cos(theta) * step_size;
            double y_new = y + m.v * sin(theta) * step_size;
            double theta_new = normalize_angle(theta + omega * step_size);
            State new_state = {round2(x_new), round2(y_new), round2(theta_new)};
            
            // Verificación de colisión optimizada
            if (is_path_collision(x, y, x_new, y_new)) {
                collision_count++;
                continue;
            }
            
            valid_moves++;

            // Costos exactamente como en el código de referencia
            int prev_dir = direction_map[current];
            double switch_penalty = (m.v != prev_dir) ? SWITCH_DIRECTION_COST : 0.0;
            double move_cost = (m.v < 0) ? REVERSE_COST : FORWARD_COST;
            double new_cost = cost_so_far[current] + step_size * move_cost + switch_penalty;

            if (!cost_so_far.count(new_state) || new_cost < cost_so_far[new_state]) {
                cost_so_far[new_state] = new_cost;
                
                // Función heurística como en el código de referencia
                double priority = new_cost + hypot(goal.pos.x - x_new, goal.pos.y - y_new);
                
                open_list.emplace(priority, new_state);
                came_from[new_state] = current;
                direction_map[new_state] = m.v;
            }
        }
        
        // Si no hay movimientos válidos en la primera expansión, reportarlo
        if (expansions == 1 && valid_moves == 0) {
            RCLCPP_ERROR(rclcpp::get_logger("hybrid_astar"),
                        "Primera expansión sin movimientos válidos. Todos en colisión.");
        }
    }

    RCLCPP_WARN(rclcpp::get_logger("hybrid_astar"), "No se encontró ruta después de %d expansiones", expansions);
    return {};
}

double dubins_like_cost(const State& start, const Goal& goal) {
    const double R = 0.6;

    double x0 = get<0>(start), y0 = get<1>(start), theta0 = get<2>(start);
    double dx = goal.pos.x - x0, dy = goal.pos.y - y0;
    double dist = hypot(dx, dy);
    double heading = atan2(dy, dx);
    double angle_diff_start = fabs(normalize_angle(heading - theta0));
    double angle_diff_end = fabs(normalize_angle(goal.theta - heading));
    
    int count_collisions = 0;
    int steps = max(int(dist / 0.05), 10);
    for (int i = 0; i <= steps; ++i) {
        double t = double(i) / steps;
        double xi = x0 + t * dx;
        double yi = y0 + t * dy;
        if (is_collision(xi, yi)) count_collisions++;
    }
    
    double penalty_collision = count_collisions * 0.1;
    
    double cost = dist + R * angle_diff_start + R * angle_diff_end + penalty_collision;
    
    if (angle_diff_start > CV_PI / 2.0) cost *= 3.0;
    
    return cost;
}
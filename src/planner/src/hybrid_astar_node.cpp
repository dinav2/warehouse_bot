#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <chrono>
#include <thread>
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace std;
using namespace cv;

using State = std::tuple<double, double, double>;

struct Goal {
    Point2d pos;
    double theta;
    bool visited = false;
};

Mat1b inflated_map;
double resolution = 10.0;
const double robot_radius = 0.3;
const double validation_distance = 0.05;
const double goal_timeout = 30.0;
vector<bool> goal_confirmed(4, false);

// === Utilidades ===
double round2(double v) {
    return round(v * 40.0) / 40.0;
}

double normalize_angle(double theta) {
    return fmod(theta + CV_PI, 2 * CV_PI) - CV_PI;
}

bool is_collision(double x, double y) {
    int row = inflated_map.rows - int(y * resolution) - 1;
    int col = int(x * resolution);
    if (row >= 0 && row < inflated_map.rows && col >= 0 && col < inflated_map.cols)
        return inflated_map(row, col) != 0;
    return true;
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

struct Motion {
    int v;
    double delta;
    double tan_delta;
};

struct StateHash {
    size_t operator()(const State& s) const {
        auto [x, y, theta] = s;
        return ((std::hash<double>()(x) << 1) ^ std::hash<double>()(y)) ^ std::hash<double>()(theta);
    }
};

vector<State> hybrid_astar(State start, const Goal& goal) {
    const double step_size = 0.25;
    const double min_turning_radius = 0.6;
    const bool allow_reverse = true;

    vector<Motion> motions;
    for (int i = 0; i < 7; ++i) {
        double delta = -CV_PI / 4 + i * (CV_PI / 12);
        motions.push_back({1, delta, tan(delta)});
        if (allow_reverse) motions.push_back({-1, delta, tan(delta)});
    }

    priority_queue<pair<double, State>, vector<pair<double, State>>, greater<>> open_list;
    unordered_map<State, State, StateHash> came_from;
    unordered_map<State, double, StateHash> cost_so_far;
    unordered_map<State, int, StateHash> direction_map;

    open_list.emplace(0.0, start);
    cost_so_far[start] = 0.0;
    direction_map[start] = 1;

    while (!open_list.empty()) {
        auto [_, current] = open_list.top(); open_list.pop();
        auto [x, y, theta] = current;

        if (hypot(goal.pos.x - x, goal.pos.y - y) < 0.1 &&
            fabs(normalize_angle(theta - goal.theta)) < 0.2) {
            vector<State> path{current};
            while (came_from.count(current)) {
                current = came_from[current];
                path.push_back(current);
            }
            reverse(path.begin(), path.end());
            RCLCPP_INFO(rclcpp::get_logger("hybrid_astar"), "Ruta generada con %zu puntos", path.size());
            return path;
        }

        for (const auto& m : motions) {
            double omega = m.v / min_turning_radius * m.tan_delta;
            double x_new = x + m.v * cos(theta) * step_size;
            double y_new = y + m.v * sin(theta) * step_size;
            double theta_new = normalize_angle(theta + omega * step_size);
            State new_state = {round2(x_new), round2(y_new), round2(theta_new)};
            if (is_collision(x_new, y_new)) continue;

            int prev_dir = direction_map[current];
            double cost = cost_so_far[current] + step_size * (m.v < 0 ? 5.0 : 1.0);
            if (m.v != prev_dir) cost += 3.0;

            if (!cost_so_far.count(new_state) || cost < cost_so_far[new_state]) {
                cost_so_far[new_state] = cost;
                double priority = cost + hypot(goal.pos.x - x_new, goal.pos.y - y_new);
                open_list.emplace(priority, new_state);
                came_from[new_state] = current;
                direction_map[new_state] = m.v;
            }
        }
    }

    return {};
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub,
                  const vector<State>& path, string frame_id = "map") {
    nav_msgs::msg::Path msg;
    msg.header.stamp = rclcpp::Clock().now();
    msg.header.frame_id = frame_id;
    for (auto [x, y, theta] : path) {
        geometry_msgs::msg::PoseStamped p;
        p.header = msg.header;
        p.pose.position.x = x - 0.374 - 0.05;
        p.pose.position.y = y - 0.315 - 0.05;
        p.pose.position.z = 0.0;
        p.pose.orientation.z = sin(theta / 2.0);
        p.pose.orientation.w = cos(theta / 2.0);
        msg.poses.push_back(p);
    }
    pub->publish(msg);
}

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("hybrid_astar_node");

    auto path_pub = node->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
    auto goal_pub = node->create_publisher<std_msgs::msg::Int32>("current_goal_index", 10);

    auto done_sub = node->create_subscription<std_msgs::msg::Int32>(
        "goal_completed", 10,
        [&](const std_msgs::msg::Int32::SharedPtr msg) {
            int id = msg->data;
            RCLCPP_INFO(node->get_logger(), "Objetivo confirmado: %d", id);
            if (id >= 0 && id < (int)goal_confirmed.size()) {
                goal_confirmed[id] = true;
            }
        });

    // Cargar mapa
    string path = ament_index_cpp::get_package_share_directory("planner") + "/maps/mapaFidedigno1.png";
    Mat map_img = imread(path, IMREAD_GRAYSCALE);
    if (map_img.empty()) {
        RCLCPP_ERROR(node->get_logger(), "No se encontró el mapa: %s", path.c_str());
        return 1;
    }

    Mat resized;
    resize(map_img, resized, Size(), 10.0, 10.0, INTER_NEAREST);
    Mat binary;
    threshold(resized, binary, 127, 1, THRESH_BINARY_INV);
    binary.convertTo(binary, CV_8U);
    int inflation_radius = int(ceil((robot_radius + validation_distance) * resolution));
    Mat kernel = getStructuringElement(MORPH_RECT, Size(inflation_radius, inflation_radius));
    dilate(binary, inflated_map, kernel);

    // Inicializar puntos y objetivos
    double xAdjust = 0.05, yAdjust = 0.05;
    State current = {0.3 + xAdjust, 0.3 + yAdjust, 0.0};
    double current_x = get<0>(current);
    double current_y = get<1>(current);

    vector<Goal> goals = {
        {{0.3 + xAdjust, 0.3 + yAdjust}, CV_PI},
        {{0.3 + xAdjust, 0.9 + yAdjust}, CV_PI},
        {{0.3 + xAdjust, 1.5 + yAdjust}, CV_PI},
        {{0.3 + xAdjust, 2.1 + yAdjust}, CV_PI}
    };

    while (rclcpp::ok()) {
        int best_goal = -1;
        double best_cost = 1e9;

        for (int i = 0; i < goals.size(); ++i) {
            if (goal_confirmed[i]) continue;
            double cost = hypot(current_x - goals[i].pos.x, current_y - goals[i].pos.y);
            if (cost < best_cost) {
                best_cost = cost;
                best_goal = i;
            }
        }

        if (best_goal == -1) {
            RCLCPP_INFO(node->get_logger(), "Todos los objetivos han sido visitados.");
            break;
        }

        auto path = hybrid_astar(current, goals[best_goal]);
        if (!path.empty()) {
            publish_path(path_pub, path);
            std_msgs::msg::Int32 gmsg;
            gmsg.data = best_goal;
            goal_pub->publish(gmsg);
            current = path.back();
            current_x = get<0>(current);
            current_y = get<1>(current);
        } else {
            RCLCPP_WARN(node->get_logger(), "No se pudo encontrar camino al objetivo %d", best_goal);
            goal_confirmed[best_goal] = true;
            continue;
        }

        rclcpp::Time start = node->get_clock()->now();
        while (rclcpp::ok() && !goal_confirmed[best_goal]) {
            rclcpp::spin_some(node);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            rclcpp::Time now = node->get_clock()->now();
            if ((now - start).seconds() > goal_timeout) {
                RCLCPP_WARN(node->get_logger(), "Timeout al esperar confirmación del objetivo %d", best_goal);
                goal_confirmed[best_goal] = true;
                break;
            }
        }
    }

    rclcpp::shutdown();
    return 0;
}


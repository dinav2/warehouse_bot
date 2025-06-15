#pragma once
#include <opencv2/opencv.hpp>
#include <tuple>
#include <vector>

using State = std::tuple<double, double, double>;

struct Goal {
    cv::Point2d pos;
    double theta;
    bool visited = false;
};

extern cv::Mat1b inflated_map;
extern double resolution;

bool is_collision(double x, double y);
bool is_path_collision(double x1, double y1, double x2, double y2);
double round2(double v);
double normalize_angle(double theta);
bool trajectory_has_reverse(const std::vector<State>& path);

std::vector<State> hybrid_astar(State start, const Goal& goal);


double dubins_like_cost(const State& start, const Goal& goal);

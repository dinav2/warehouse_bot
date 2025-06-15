#include "planner/planner_utils.hpp"
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <optional>
#include <map>
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace std;
using namespace cv;

struct GoalWithClass {
    Goal goal;
    int detected_class;
    bool visited_605;
    bool visited_304;
    
    GoalWithClass(const Goal& g) : goal(g), detected_class(-1), visited_605(false), visited_304(false) {}
};

class IntegratedMissionPlanner : public rclcpp::Node {
private:
    // === PLANNING CONSTANTS ===
    // Robot parameters
    const double robot_radius_ = 0.3;
    const double validation_distance_ = 0.01;
    
    // Map parameters
    const double map_scale_factor_ = 10.0;
    const double map_inflation_radius_multiplier_x_ = 1.2;
    const double map_inflation_radius_multiplier_y_ = 1.0;
    
    // Goal coordinates - FASE 1: Posiciones externas (.605)
    const double goal_base_x_605_ = 0.605;
    const double goal_base_y_start_ = 0.25;
    const double goal_spacing_y_ = 0.6;
    const double goal_adjust_x_ = 0.05;
    const double goal_adjust_y_ = 0.05;
    
    // Goal coordinates - FASE 2: Posiciones internas (.304)
    const double goal_base_x_304_ = 0.304;
    
    // Coordinate transformations
    const double amcl_to_planning_offset_x_ = 0.3 + goal_adjust_x_;
    const double amcl_to_planning_offset_y_ = 0.2 + goal_adjust_y_;
    const double path_publish_offset_x_ = 0.3 + goal_adjust_x_;
    const double path_publish_offset_y_ = 0.2 + goal_adjust_y_;
    
    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr goal_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    
    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr goal_completed_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr amcl_sub_;
    
    // Vision service client
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr vision_client_;
    
    // State variables
    std::optional<State> current_state_opt_;
    vector<GoalWithClass> goals_;
    vector<bool> goal_confirmed_;
    
    // Mission state
    enum class MissionPhase { EXPLORATION_605, VISIT_SPECIFIC_304, COMPLETED };
    MissionPhase current_phase_;
    
    // Speed control
    bool half_speed_mode_;
    
public:
    IntegratedMissionPlanner() : Node("integrated_mission_planner"), current_phase_(MissionPhase::EXPLORATION_605), half_speed_mode_(false) {
        // Initialize publishers
        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planned_path", 10);
        goal_pub_ = this->create_publisher<std_msgs::msg::Int32>("current_goal_index", 10);
        status_pub_ = this->create_publisher<std_msgs::msg::String>("mission_status", 10);
        
        // Initialize subscribers
        goal_completed_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "goal_completed", 10,
            [this](const std_msgs::msg::Int32::SharedPtr msg) {
                this->goal_completed_callback(msg);
            });
            
        amcl_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/amcl_pose", 10,
            [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
                this->amcl_pose_callback(msg);
            });
        
        // Initialize vision service client
        vision_client_ = this->create_client<std_srvs::srv::Trigger>("classify_signal");
        
        // Initialize map and goals
        initialize_map_and_goals();
        
        RCLCPP_INFO(this->get_logger(), "==== Inicialización de IntegratedMissionPlanner completada ====");
        RCLCPP_INFO(this->get_logger(), "Publicando en:");
        RCLCPP_INFO(this->get_logger(), "- planned_path (nav_msgs/msg/Path)");
        RCLCPP_INFO(this->get_logger(), "- current_goal_index (std_msgs/msg/Int32)");
        RCLCPP_INFO(this->get_logger(), "- mission_status (std_msgs/msg/String)");
        RCLCPP_INFO(this->get_logger(), "Suscrito a:");
        RCLCPP_INFO(this->get_logger(), "- goal_completed (std_msgs/msg/Int32)");
        RCLCPP_INFO(this->get_logger(), "- /amcl_pose (geometry_msgs/msg/PoseWithCovarianceStamped)");
        RCLCPP_INFO(this->get_logger(), "Cliente de servicio:");
        RCLCPP_INFO(this->get_logger(), "- classify_signal (std_srvs/srv/Trigger)");
        RCLCPP_INFO(this->get_logger(), "=======================================================");
    }
    
private:
    void initialize_map_and_goals() {
        extern double resolution;
        
        RCLCPP_INFO(this->get_logger(), "Resolución global: %.2f", resolution);
        RCLCPP_INFO(this->get_logger(), "Factor de escala del mapa: %.2f", map_scale_factor_);
        
        // Load map
        string map_path = ament_index_cpp::get_package_share_directory("planner") + "/maps/mapaFidedigno1.png";
        Mat map_img = imread(map_path, IMREAD_GRAYSCALE);
        if (map_img.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No se encontró el mapa: %s", map_path.c_str());
            return;
        }
        
        RCLCPP_INFO(this->get_logger(), "Mapa original cargado: %dx%d", map_img.cols, map_img.rows);
        
        // Scale map
        Mat img2;
        resize(map_img, img2, Size(), map_scale_factor_, map_scale_factor_, INTER_NEAREST);
        
        RCLCPP_INFO(this->get_logger(), "Mapa escalado: %dx%d", img2.cols, img2.rows);
        
        Mat binary;
        threshold(img2, binary, 127, 1, THRESH_BINARY_INV);
        binary.convertTo(binary, CV_8U);
        
        // Inflation
        int inflation_radius = int(ceil((robot_radius_ + validation_distance_) * resolution));
        Mat kernel = getStructuringElement(MORPH_RECT, Size(inflation_radius, inflation_radius));
        dilate(binary, inflated_map, kernel);
        
        RCLCPP_INFO(this->get_logger(), "Mapa inflado: %dx%d", inflated_map.cols, inflated_map.rows);
        
        // Initialize goals for PHASE 1 (.605 positions)
        goals_ = {
            GoalWithClass({{goal_base_x_605_, 0.3029}, CV_PI}),
            GoalWithClass({{goal_base_x_605_, 0.9079}, CV_PI}),
            GoalWithClass({{goal_base_x_605_, 1.5129}, CV_PI}),
            GoalWithClass({{goal_base_x_605_, 2.1179}, CV_PI})
        };
        
        // Verificar coordenadas
        RCLCPP_INFO(this->get_logger(), "Coordenadas de metas FASE 1 (.605):");
        for (int i = 0; i < goals_.size(); i++) {
            RCLCPP_INFO(this->get_logger(), "Meta %d: (%.3f, %.3f)", i, goals_[i].goal.pos.x, goals_[i].goal.pos.y);
            
            if (is_collision(goals_[i].goal.pos.x, goals_[i].goal.pos.y)) {
                RCLCPP_WARN(this->get_logger(), "¡ADVERTENCIA! Meta %d está en colisión", i);
            }
        }
        
        goal_confirmed_.resize(goals_.size(), false);
        
        RCLCPP_INFO(this->get_logger(), "Mapa cargado y objetivos inicializados");
        RCLCPP_INFO(this->get_logger(), "Dimensiones del mapa: %dx%d", inflated_map.cols, inflated_map.rows);
        RCLCPP_INFO(this->get_logger(), "Radio de inflación: %d", inflation_radius);
        RCLCPP_INFO(this->get_logger(), "Resolución: %.2f (1m = %.0f px)", resolution, resolution);
    }
    
    void goal_completed_callback(const std_msgs::msg::Int32::SharedPtr msg) {
        int id = msg->data;
        RCLCPP_INFO(this->get_logger(), "Objetivo completado recibido: %d", id);
        
        if (id >= 0 && id < (int)goal_confirmed_.size()) {
            goal_confirmed_[id] = true;
            RCLCPP_INFO(this->get_logger(), "Objetivo %d confirmado y marcado como completado", id);
            
            // Llamar al servicio de visión cuando se llega a un goal
            if (current_phase_ == MissionPhase::EXPLORATION_605) {
                request_vision_classification(id);
            }
            
        } else {
            RCLCPP_ERROR(this->get_logger(), "ID de objetivo inválido: %d. Rango válido: 0-%zu", 
                         id, goal_confirmed_.size() - 1);
        }
    }
    
    void request_vision_classification(int goal_id) {
        RCLCPP_INFO(this->get_logger(), "Solicitando clasificación de visión para goal %d", goal_id);
        
        if (!vision_client_->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_WARN(this->get_logger(), "Servicio de visión no disponible");
            return;
        }
        
        auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
        
        auto future = vision_client_->async_send_request(request);
        
        // Wait for response (blocking for simplicity)
        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), future) ==
            rclcpp::FutureReturnCode::SUCCESS) {
            
            auto response = future.get();
            if (response->success) {
                int detected_class = std::stoi(response->message);
                RCLCPP_INFO(this->get_logger(), "Visión detectó clase %d en goal %d", detected_class, goal_id);
                
                // Guardar solo si es clase 0 (parking) o clase 2 (loading)
                if (detected_class == 0 || detected_class == 2) {
                    goals_[goal_id].detected_class = detected_class;
                    goals_[goal_id].visited_605 = true;
                    RCLCPP_INFO(this->get_logger(), "Guardada clase %d para goal %d", detected_class, goal_id);
                } else if (detected_class == 1) {
                    // También guardar clase 1 pero no la mencionamos en los logs de almacenamiento
                    goals_[goal_id].detected_class = detected_class;
                    goals_[goal_id].visited_605 = true;
                    RCLCPP_INFO(this->get_logger(), "Detectada clase 1 en goal %d", goal_id);
                } else if (detected_class >= 3) {
                    // Señales de velocidad - activar modo media velocidad
                    RCLCPP_WARN(this->get_logger(), "Señal de velocidad detectada (clase %d) - Activando modo media velocidad", detected_class);
                    half_speed_mode_ = true;
                }
                
            } else {
                RCLCPP_ERROR(this->get_logger(), "Error en servicio de visión: %s", response->message.c_str());
            }
        } else {
            RCLCPP_ERROR(this->get_logger(), "Timeout en servicio de visión");
        }
    }
    
    void amcl_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        double x = msg->pose.pose.position.x + amcl_to_planning_offset_x_;
        double y = msg->pose.pose.position.y + amcl_to_planning_offset_y_;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        double theta = atan2(2 * qw * qz, 1 - 2 * qz * qz);
        
        static double last_x = 0, last_y = 0;
        static int count = 0;
        
        if (count++ % 10 == 0 || hypot(x - last_x, y - last_y) > 0.1) {
            RCLCPP_INFO(this->get_logger(), "Posición AMCL: (%.3f, %.3f, %.3f)", x, y, theta);
            last_x = x;
            last_y = y;
        }
        
        current_state_opt_ = State{x, y, theta};
    }
    
    void publish_path(const vector<State>& path) {
        nav_msgs::msg::Path msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";
        
        for (auto [x, y, theta] : path) {
            geometry_msgs::msg::PoseStamped p;
            p.header = msg.header;
            p.pose.position.x = x - path_publish_offset_x_;
            p.pose.position.y = y - path_publish_offset_y_;
            p.pose.position.z = 0.0;
            p.pose.orientation.z = sin(theta / 2.0);
            p.pose.orientation.w = cos(theta / 2.0);
            msg.poses.push_back(p);
        }
        
        path_pub_->publish(msg);
        
        // Aplicar modificación de velocidad si está activo el modo media velocidad
        if (half_speed_mode_) {
            RCLCPP_INFO(this->get_logger(), "Publicando path con modo media velocidad activado");
        }
    }
    
    void publish_status(const string& status) {
        std_msgs::msg::String msg;
        msg.data = status;
        status_pub_->publish(msg);
    }
    
    void wait_for_amcl_pose() {
        publish_status("WAITING_POSE");
        while (rclcpp::ok() && !current_state_opt_.has_value()) {
            RCLCPP_INFO(this->get_logger(), "Esperando pose inicial de AMCL...");
            rclcpp::spin_some(this->shared_from_this());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        RCLCPP_INFO(this->get_logger(), "Pose inicial recibida de AMCL");
    }
    
    bool all_phase1_goals_visited() {
        for (const auto& goal_with_class : goals_) {
            if (!goal_with_class.visited_605) {
                return false;
            }
        }
        return true;
    }
    
    int find_goal_with_class(int target_class) {
        for (int i = 0; i < goals_.size(); i++) {
            if (goals_[i].detected_class == target_class && !goals_[i].visited_304) {
                return i;
            }
        }
        return -1;
    }
    
    Goal create_304_goal(int original_goal_id) {
        // Crear objetivo en la posición .304 correspondiente
        double y_pos = goals_[original_goal_id].goal.pos.y;
        return {{goal_base_x_304_, y_pos}, CV_PI};
    }
    
public:
    void run_mission() {
        wait_for_amcl_pose();
        
        if (!current_state_opt_.has_value()) {
            RCLCPP_ERROR(this->get_logger(), "No se pudo obtener pose inicial");
            return;
        }
        
        State current = current_state_opt_.value();
        publish_status("PLANNING");
        
        RCLCPP_INFO(this->get_logger(), "=== INICIANDO FASE 1: EXPLORACIÓN EN POSICIONES .605 ===");
        
        // FASE 1: Exploración de las 4 posiciones .605
        while (current_phase_ == MissionPhase::EXPLORATION_605 && rclcpp::ok()) {
            // Verificar si se completó la fase 1
            if (all_phase1_goals_visited()) {
                current_phase_ = MissionPhase::VISIT_SPECIFIC_304;
                RCLCPP_INFO(this->get_logger(), "=== COMPLETADA FASE 1 - INICIANDO FASE 2: VISITAS ESPECÍFICAS .304 ===");
                break;
            }
            
            double best_cost = 1e9;
            int best_idx = -1;
            
            // Encontrar el mejor goal no visitado
            for (int i = 0; i < goals_.size(); ++i) {
                if (goals_[i].visited_605) continue;
                
                double cost = dubins_like_cost(current, goals_[i].goal);
                RCLCPP_INFO(this->get_logger(), "Costo estimado a objetivo %d: %.3f", i, cost);
                
                if (cost < best_cost) {
                    best_cost = cost;
                    best_idx = i;
                }
            }
            
            if (best_idx == -1) break;
            
            // Planificar y ejecutar ruta
            RCLCPP_INFO(this->get_logger(), "Planificando hacia objetivo %d (FASE 1)", best_idx);
            auto& goal = goals_[best_idx].goal;
            
            auto path = hybrid_astar(current, goal);
            if (path.empty()) {
                RCLCPP_WARN(this->get_logger(), "No se encontró ruta hacia objetivo %d", best_idx);
                goals_[best_idx].visited_605 = true; // Marcar como visitado para continuar
                continue;
            }
            
            publish_path(path);
            std_msgs::msg::Int32 goal_msg;
            goal_msg.data = best_idx;
            goal_pub_->publish(goal_msg);
            
            publish_status("EXECUTING_PHASE1");
            
            // Esperar confirmación
            while (rclcpp::ok() && !goal_confirmed_[best_idx]) {
                rclcpp::spin_some(this->shared_from_this());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                if (current_state_opt_.has_value()) {
                    current = current_state_opt_.value();
                }
            }
            
            goal_confirmed_[best_idx] = false; // Reset para próxima fase
            RCLCPP_INFO(this->get_logger(), "Objetivo %d completado (FASE 1)", best_idx);
        }
        
        // FASE 2: Visitas específicas a posiciones .304
        if (current_phase_ == MissionPhase::VISIT_SPECIFIC_304) {
            // Secuencia: Clase 1 -> Clase 2 (loading) -> Clase 0 (parking)
            vector<int> visit_sequence = {1, 2, 0};
            
            for (int target_class : visit_sequence) {
                int goal_id = find_goal_with_class(target_class);
                if (goal_id == -1) {
                    RCLCPP_WARN(this->get_logger(), "No se encontró goal con clase %d", target_class);
                    continue;
                }
                
                Goal target_304 = create_304_goal(goal_id);
                
                string class_name = (target_class == 0) ? "parking" : 
                                  (target_class == 1) ? "intermedio" : "loading";
                
                RCLCPP_INFO(this->get_logger(), "=== FASE 2: Dirigiéndose a zona %s (clase %d) en posición .304 ===", 
                           class_name.c_str(), target_class);
                
                auto path = hybrid_astar(current, target_304);
                if (path.empty()) {
                    RCLCPP_WARN(this->get_logger(), "No se encontró ruta hacia zona %s", class_name.c_str());
                    continue;
                }
                
                publish_path(path);
                std_msgs::msg::Int32 goal_msg;
                goal_msg.data = goal_id + 100; // Diferente ID para fase 2
                goal_pub_->publish(goal_msg);
                
                publish_status("EXECUTING_PHASE2_" + class_name);
                
                // Simular llegada (en implementación real esperarías confirmación)
                std::this_thread::sleep_for(std::chrono::seconds(5));
                
                goals_[goal_id].visited_304 = true;
                current = {target_304.pos.x, target_304.pos.y, target_304.theta};
                
                RCLCPP_INFO(this->get_logger(), "Completada visita a zona %s", class_name.c_str());
            }
        }
        
        current_phase_ = MissionPhase::COMPLETED;
        publish_status("MISSION_COMPLETED");
        RCLCPP_INFO(this->get_logger(), "=== MISIÓN COMPLETADA ===");
        
        // Mostrar resumen
        RCLCPP_INFO(this->get_logger(), "Resumen de detecciones:");
        for (int i = 0; i < goals_.size(); i++) {
            RCLCPP_INFO(this->get_logger(), "Goal %d: Clase detectada = %d", i, goals_[i].detected_class);
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    
    auto planner = std::make_shared<IntegratedMissionPlanner>();
    
    // Run mission in separate thread to allow service callbacks
    std::thread mission_thread([&planner]() {
        planner->run_mission();
    });
    
    // Spin to handle service calls and callbacks
    rclcpp::spin(planner);
    
    mission_thread.join();
    rclcpp::shutdown();
    return 0;
} 
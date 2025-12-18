import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import networkx as nx
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class XianMarathonOptimizer:
    def __init__(self):
        self.attractions = None
        self.accommodations = None
        self.restaurants = None
        self.metro_stations = None
        self.road_network = None

    def generate_xian_data(self):
        """生成西安市模拟数据"""
        np.random.seed(42)

        # 西安市中心坐标范围 (经度, 纬度)
        lon_range = (108.8, 109.1)
        lat_range = (34.1, 34.4)

        # 生成景点数据
        n_attractions = 20
        self.attractions = pd.DataFrame({
            'id': range(n_attractions),
            'name': [f'景点{i + 1}' for i in range(n_attractions)],
            'longitude': np.random.uniform(lon_range[0], lon_range[1], n_attractions),
            'latitude': np.random.uniform(lat_range[0], lat_range[1], n_attractions),
            'importance': np.random.uniform(0.5, 1.0, n_attractions)
        })

        # 生成住宿设施数据
        n_accommodations = 15
        self.accommodations = pd.DataFrame({
            'id': range(n_accommodations),
            'name': [f'酒店{i + 1}' for i in range(n_accommodations)],
            'longitude': np.random.uniform(lon_range[0], lon_range[1], n_accommodations),
            'latitude': np.random.uniform(lat_range[0], lat_range[1], n_accommodations),
            'capacity': np.random.randint(100, 800, n_accommodations)
        })

        # 生成餐饮设施数据
        n_restaurants = 30
        self.restaurants = pd.DataFrame({
            'id': range(n_restaurants),
            'name': [f'餐厅{i + 1}' for i in range(n_restaurants)],
            'longitude': np.random.uniform(lon_range[0], lon_range[1], n_restaurants),
            'latitude': np.random.uniform(lat_range[0], lat_range[1], n_restaurants),
            'rating': np.random.uniform(3.0, 5.0, n_restaurants)
        })

        # 生成地铁站数据
        n_metro = 12
        self.metro_stations = pd.DataFrame({
            'id': range(n_metro),
            'name': [f'地铁站{i + 1}' for i in range(n_metro)],
            'longitude': np.random.uniform(lon_range[0], lon_range[1], n_metro),
            'latitude': np.random.uniform(lat_range[0], lat_range[1], n_metro),
            'line': np.random.choice(['1号线', '2号线', '3号线', '4号线'], n_metro)
        })

        print("西安市模拟数据生成完成!")
        print(f"景点数量: {len(self.attractions)}")
        print(f"住宿设施数量: {len(self.accommodations)}")
        print(f"餐饮设施数量: {len(self.restaurants)}")
        print(f"地铁站数量: {len(self.metro_stations)}")

    def calculate_distance(self, coord1, coord2):
        """计算两点间距离（公里）"""
        # 简化的距离计算（实际应使用地理距离公式）
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # 将经纬度转换为近似的公里数
        lat_km = (lat2 - lat1) * 111.0  # 1度纬度约111公里
        lon_km = (lon2 - lon1) * 111.0 * np.cos(np.radians(lat1))  # 经度随纬度变化

        return np.sqrt(lat_km ** 2 + lon_km ** 2)

    def evaluate_service_area(self, point, accommodations_nearby, road_density):
        """评价函数：考虑容量权重与邻近路网密度"""
        total_capacity = accommodations_nearby['capacity'].sum()
        capacity_weight = min(total_capacity / 1000.0, 1.0)  # 归一化到0-1
        density_weight = min(road_density / 100.0, 1.0)  # 归一化到0-1

        # 综合评价分数
        score = 0.6 * capacity_weight + 0.4 * density_weight
        return score

    def find_nearby_points(self, center, points_df, radius_km):
        """查找指定半径内的点"""
        distances = []
        for _, point in points_df.iterrows():
            dist = self.calculate_distance(
                (center[1], center[0]),
                (point['latitude'], point['longitude'])
            )
            distances.append(dist)

        points_df = points_df.copy()
        points_df['distance'] = distances
        return points_df[points_df['distance'] <= radius_km]

    def solve_part1(self):
        """解决问题(1)：寻找最优起点-终点组合"""
        print("\n=== 解决问题(1)：寻找最优起点-终点组合 ===")

        best_routes = []

        # 遍历所有可能的起点-终点组合
        for i, start_point in self.attractions.iterrows():
            for j, end_point in self.attractions.iterrows():
                if i >= j:  # 避免重复计算
                    continue

                start_coord = (start_point['longitude'], start_point['latitude'])
                end_coord = (end_point['longitude'], end_point['latitude'])

                # 计算距离
                distance = self.calculate_distance(
                    (start_coord[1], start_coord[0]),
                    (end_coord[1], end_coord[0])
                )

                # 检查距离要求
                if distance < 42:
                    continue

                # 检查起点3000米内住宿容量
                nearby_accommodations = self.find_nearby_points(
                    start_coord, self.accommodations, 3.0
                )
                total_capacity = nearby_accommodations['capacity'].sum()

                if total_capacity < 3000:
                    continue

                # 检查起终点是否邻近地铁站
                start_metro = self.find_nearby_points(
                    start_coord, self.metro_stations, 0.5
                )
                end_metro = self.find_nearby_points(
                    end_coord, self.metro_stations, 0.5
                )

                if len(start_metro) == 0 or len(end_metro) == 0:
                    continue

                # 计算路网密度（模拟）
                road_density = np.random.uniform(50, 150)

                # 计算评价分数
                score = self.evaluate_service_area(
                    start_coord, nearby_accommodations, road_density
                )

                route_info = {
                    'start_point': start_point['name'],
                    'end_point': end_point['name'],
                    'start_coord': start_coord,
                    'end_coord': end_coord,
                    'distance': distance,
                    'capacity': total_capacity,
                    'score': score,
                    'start_metro': start_metro.iloc[0]['name'] if len(start_metro) > 0 else None,
                    'end_metro': end_metro.iloc[0]['name'] if len(end_metro) > 0 else None
                }

                best_routes.append(route_info)

        # 按评价分数排序
        best_routes = sorted(best_routes, key=lambda x: x['score'], reverse=True)

        print(f"找到 {len(best_routes)} 条符合条件的路线")

        if best_routes:
            print("\n前3条最优路线：")
            for i, route in enumerate(best_routes[:3]):
                print(f"{i + 1}. {route['start_point']} -> {route['end_point']}")
                print(f"   距离: {route['distance']:.2f}km")
                print(f"   住宿容量: {route['capacity']}人")
                print(f"   评价分数: {route['score']:.3f}")
                print(f"   起点地铁: {route['start_metro']}")
                print(f"   终点地铁: {route['end_metro']}")
                print()

        return best_routes

    def solve_part2(self):
        """解决问题(2)：设计闭合回路马拉松路线"""
        print("\n=== 解决问题(2)：设计闭合回路马拉松路线 ===")

        # 马拉松距离要求
        distances = {
            '全马': 42.195,
            '半马': 21.0975,
            '健康跑': 10.0
        }

        results = {}

        for race_type, target_distance in distances.items():
            print(f"\n设计{race_type}路线 (目标距离: {target_distance}km)")

            # 选择起点（选择住宿容量最大的景点附近）
            start_idx = 0
            start_point = self.attractions.iloc[start_idx]
            current_pos = (start_point['longitude'], start_point['latitude'])

            # 构建路线
            route_points = [start_point]
            visited_attractions = set([start_idx])
            visited_restaurants = set()
            total_distance = 0
            total_gain = 0
            supply_stations = []

            while total_distance < target_distance * 0.9:  # 90%完成时开始考虑返回
                # 寻找下一个景点
                min_dist = float('inf')
                next_idx = None

                for i, attraction in self.attractions.iterrows():
                    if i in visited_attractions:
                        continue

                    dist = self.calculate_distance(
                        (current_pos[1], current_pos[0]),
                        (attraction['latitude'], attraction['longitude'])
                    )

                    if dist < min_dist:
                        min_dist = dist
                        next_idx = i

                if next_idx is None or total_distance + min_dist > target_distance:
                    break

                # 移动到下一个景点
                next_point = self.attractions.iloc[next_idx]
                route_points.append(next_point)
                visited_attractions.add(next_idx)
                total_distance += min_dist
                current_pos = (next_point['longitude'], next_point['latitude'])

                # 检查沿途餐饮设施（增益节点）
                nearby_restaurants = self.find_nearby_points(
                    current_pos, self.restaurants, 0.5
                )

                for _, restaurant in nearby_restaurants.iterrows():
                    if restaurant['id'] not in visited_restaurants:
                        visited_restaurants.add(restaurant['id'])
                        total_gain += 0.2

                # 每5km设置补给站
                if int(total_distance / 5) > len(supply_stations):
                    nearby_supply = self.find_nearby_points(
                        current_pos, self.restaurants, 1.0
                    )
                    if len(nearby_supply) > 0:
                        best_supply = nearby_supply.loc[nearby_supply['rating'].idxmax()]
                        supply_stations.append({
                            'km': len(supply_stations) * 5 + 5,
                            'location': best_supply['name'],
                            'coord': (best_supply['longitude'], best_supply['latitude'])
                        })

            # 返回起点完成闭合回路
            return_distance = self.calculate_distance(
                (current_pos[1], current_pos[0]),
                (start_point['latitude'], start_point['longitude'])
            )
            total_distance += return_distance

            results[race_type] = {
                'route_points': route_points,
                'total_distance': total_distance,
                'total_gain': total_gain,
                'supply_stations': supply_stations,
                'visited_restaurants': len(visited_restaurants)
            }

            print(f"路线总距离: {total_distance:.3f}km")
            print(f"经过景点数: {len(route_points)}")
            print(f"总增益值: {total_gain:.1f}")
            print(f"补给站数量: {len(supply_stations)}")

        return results

    def plot_results(self, best_routes, marathon_routes):
        """绘制结果图形"""
        fig = plt.figure(figsize=(20, 12))

        # 子图1: 服务范围划分和最优路线(问题1)
        ax1 = plt.subplot(2, 3, 1)

        # 绘制所有设施点
        ax1.scatter(self.attractions['longitude'], self.attractions['latitude'],
                    c='red', s=100, marker='*', label='景点', alpha=0.8)
        ax1.scatter(self.accommodations['longitude'], self.accommodations['latitude'],
                    c='blue', s=60, marker='s', label='住宿', alpha=0.6)
        ax1.scatter(self.metro_stations['longitude'], self.metro_stations['latitude'],
                    c='green', s=80, marker='^', label='地铁站', alpha=0.7)

        # 绘制最优路线
        if best_routes:
            best_route = best_routes[0]
            ax1.plot([best_route['start_coord'][0], best_route['end_coord'][0]],
                     [best_route['start_coord'][1], best_route['end_coord'][1]],
                     'purple', linewidth=3, label='最优路线')
            ax1.scatter([best_route['start_coord'][0], best_route['end_coord'][0]],
                        [best_route['start_coord'][1], best_route['end_coord'][1]],
                        c='purple', s=150, marker='o', edgecolor='black', linewidth=2)

        ax1.set_xlabel('经度')
        ax1.set_ylabel('纬度')
        ax1.set_title('问题1: 最优起点-终点路线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 住宿容量分布
        ax2 = plt.subplot(2, 3, 2)
        scatter = ax2.scatter(self.accommodations['longitude'], self.accommodations['latitude'],
                              c=self.accommodations['capacity'], s=self.accommodations['capacity'] / 3,
                              cmap='YlOrRd', alpha=0.7)
        plt.colorbar(scatter, ax=ax2, label='住宿容量')
        ax2.set_xlabel('经度')
        ax2.set_ylabel('纬度')
        ax2.set_title('住宿设施容量分布')
        ax2.grid(True, alpha=0.3)

        # 子图3-5: 马拉松路线(问题2)
        race_types = ['健康跑', '半马', '全马']
        colors = ['green', 'orange', 'red']

        for i, (race_type, color) in enumerate(zip(race_types, colors)):
            ax = plt.subplot(2, 3, 3 + i)

            # 绘制基础设施
            ax.scatter(self.attractions['longitude'], self.attractions['latitude'],
                       c='lightcoral', s=50, marker='*', alpha=0.5, label='景点')
            ax.scatter(self.restaurants['longitude'], self.restaurants['latitude'],
                       c='lightblue', s=30, marker='o', alpha=0.4, label='餐饮')

            if race_type in marathon_routes:
                route = marathon_routes[race_type]
                route_points = route['route_points']

                # 绘制路线
                lons = [p['longitude'] for p in route_points] + [route_points[0]['longitude']]
                lats = [p['latitude'] for p in route_points] + [route_points[0]['latitude']]
                ax.plot(lons, lats, color=color, linewidth=2, label=f'{race_type}路线')

                # 标记起点
                ax.scatter(route_points[0]['longitude'], route_points[0]['latitude'],
                           c=color, s=200, marker='o', edgecolor='black', linewidth=2,
                           label='起/终点')

                # 标记补给站
                supply_coords = [(s['coord'][0], s['coord'][1]) for s in route['supply_stations']]
                if supply_coords:
                    supply_lons, supply_lats = zip(*supply_coords)
                    ax.scatter(supply_lons, supply_lats, c='purple', s=100,
                               marker='D', label='补给站')

            ax.set_xlabel('经度')
            ax.set_ylabel('纬度')
            ax.set_title(f'{race_type}路线设计')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 子图6: 统计信息
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')

        # 准备统计文本
        stats_text = "路线统计信息\n" + "=" * 20 + "\n\n"

        if best_routes:
            best_route = best_routes[0]
            stats_text += f"问题1最优路线:\n"
            stats_text += f"起点: {best_route['start_point']}\n"
            stats_text += f"终点: {best_route['end_point']}\n"
            stats_text += f"距离: {best_route['distance']:.2f}km\n"
            stats_text += f"住宿容量: {best_route['capacity']}人\n"
            stats_text += f"评价分数: {best_route['score']:.3f}\n\n"

        stats_text += "问题2马拉松路线:\n"
        for race_type in ['健康跑', '半马', '全马']:
            if race_type in marathon_routes:
                route = marathon_routes[race_type]
                stats_text += f"\n{race_type}:\n"
                stats_text += f"  距离: {route['total_distance']:.3f}km\n"
                stats_text += f"  经过景点: {len(route['route_points'])}个\n"
                stats_text += f"  增益值: {route['total_gain']:.1f}\n"
                stats_text += f"  补给站: {len(route['supply_stations'])}个\n"

        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.savefig('xian_marathon_optimization.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制详细的路线高程图（模拟）
        self.plot_elevation_profile(marathon_routes)

    def plot_elevation_profile(self, marathon_routes):
        """绘制路线高程剖面图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        race_types = ['健康跑', '半马', '全马']
        colors = ['green', 'orange', 'red']

        for i, (race_type, color) in enumerate(zip(race_types, colors)):
            ax = axes[i]

            if race_type in marathon_routes:
                route = marathon_routes[race_type]
                distance = route['total_distance']

                # 生成模拟高程数据（确保坡度≤5%）
                n_points = int(distance * 10)  # 每100米一个点
                distances = np.linspace(0, distance, n_points)

                # 生成高程变化（限制坡度）
                elevation_changes = np.random.normal(0, 2, n_points - 1)  # 随机高程变化
                # 限制坡度不超过5%
                max_change = 0.05 * (distances[1] - distances[0]) * 1000  # 5%坡度对应的高程变化(米)
                elevation_changes = np.clip(elevation_changes, -max_change, max_change)

                elevations = np.zeros(n_points)
                elevations[0] = 400  # 西安平均海拔约400米
                for j in range(1, n_points):
                    elevations[j] = elevations[j - 1] + elevation_changes[j - 1]

                # 绘制高程剖面
                ax.fill_between(distances, elevations, alpha=0.3, color=color)
                ax.plot(distances, elevations, color=color, linewidth=2)

                # 标记补给站位置
                for station in route['supply_stations']:
                    if station['km'] <= distance:
                        station_idx = int(station['km'] / distance * n_points)
                        if station_idx < len(elevations):
                            ax.axvline(x=station['km'], color='purple', linestyle='--', alpha=0.7)
                            ax.scatter(station['km'], elevations[station_idx],
                                       c='purple', s=100, marker='D', zorder=10)

                # 计算并显示最大坡度
                gradients = np.diff(elevations) / (np.diff(distances) * 1000) * 100  # 转换为百分比
                max_gradient = np.max(np.abs(gradients))

                ax.set_xlabel('距离 (km)')
                ax.set_ylabel('海拔 (m)')
                ax.set_title(f'{race_type}高程剖面\n最大坡度: {max_gradient:.1f}%')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('elevation_profiles.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_optimization(self):
        """运行完整的优化流程"""
        print("西安市马拉松路线优化系统")
        print("=" * 50)

        # 生成数据
        self.generate_xian_data()

        # 解决问题1
        best_routes = self.solve_part1()

        # 解决问题2
        marathon_routes = self.solve_part2()

        # 绘制结果
        self.plot_results(best_routes, marathon_routes)

        return best_routes, marathon_routes


# 运行优化系统
if __name__ == "__main__":
    optimizer = XianMarathonOptimizer()
    best_routes, marathon_routes = optimizer.run_optimization()

    print("\n" + "=" * 50)
    print("优化完成！请查看生成的图表。")
    print("生成的文件:")
    print("- xian_marathon_optimization.png: 主要结果图")
    print("- elevation_profiles.png: 路线高程剖面图")
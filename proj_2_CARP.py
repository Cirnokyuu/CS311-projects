import argparse
import random
import time
import math
import copy
from collections import defaultdict

DEBUG = False

def read_carp_file(filename):
    metadata = {}
    graph = defaultdict(list)
    edges = []
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    current_line = 0
    while current_line < len(lines) and not lines[current_line].startswith('NODES'):
        line = lines[current_line]
        if ':' in line:
            key, value = map(str.strip, line.split(':', 1))
            metadata[key] = value
        current_line += 1
    
    int_keys = ['VERTICES', 'DEPOT', 'REQUIRED EDGES', 'NON-REQUIRED EDGES',
                'VEHICLES', 'CAPACITY', 'TOTAL COST OF REQUIRED EDGES']
    for key in int_keys:
        if key in metadata:
            metadata[key] = int(metadata[key])
    
    if current_line < len(lines) and lines[current_line].startswith('NODES'):
        current_line += 1
        while current_line < len(lines) and lines[current_line] != 'END':
            parts = lines[current_line].split()
            if len(parts) >= 4:
                node1 = int(parts[0])
                node2 = int(parts[1])
                cost = int(parts[2])
                demand = int(parts[3])
                graph[node1].append((node2, cost, demand))
                graph[node2].append((node1, cost, demand))
                edges.append((node1, node2, cost, demand))
            current_line += 1
    
    metadata['edges'] = edges
    return metadata, dict(graph)

def floyd(graph):
    nodes = list(graph.keys())
    dist = {node: {node2: float('inf') for node2 in nodes} for node in nodes}
    for node in nodes:
        dist[node][node] = 0
    for node1, edges in graph.items():
        for node2, cost, _ in edges:
            dist[node1][node2] = min(dist[node1][node2], cost)
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

def path_scanning(tasks, depot, capacity, dist, ruleid):
    rules = [
        (lambda t, load: dist[t[1]][depot]),
        (lambda t, load: -dist[t[1]][depot]),
        (lambda t, load: t[3]/t[2]),
        (lambda t, load: -t[3]/t[2]),
        (lambda t, load: dist[t[1]][depot] if load < capacity * 0.4 else -dist[t[1]][depot])
    ]
    
    # best_routes = []
    # best_cost = float('inf')
    
    rule_func = rules[ruleid]
    remaining = {t[:2]: t for t in tasks}
    routes = []
    while remaining:
        route = []
        load = 0
        pos = depot
        while True:
            candidates = []
            for (u, v) in list(remaining.keys()):
                t = remaining[(u, v)]
                if load + t[3] <= capacity:
                    candidates.append((u, v, t[2], t[3]))
            
            if not candidates:
                break
            
            min_dist = min(dist[pos][t[0]] for t in candidates)
            nearest = [t for t in candidates if dist[pos][t[0]] == min_dist]
            scores = [rule_func(t, load) for t in nearest]
            best_idx = random.choice([i for i, score in enumerate(scores) if score == max(scores)])
            selected = nearest[best_idx]
            
            route.append(selected)
            load += selected[3]
            pos = selected[1]
            del remaining[(selected[0], selected[1])]
            if (selected[1], selected[0]) in remaining:
                del remaining[(selected[1], selected[0])]
        
        if route:
            routes.append(route)
    
    # 计算总成本
    total_cost = 0
    for route in routes:
        current = depot
        for t in route:
            total_cost += dist[current][t[0]] + t[2]
            current = t[1]
        total_cost += dist[current][depot]
    
    return routes, total_cost

def greedy(metadata, tasks, depot, capacity, dist, greedy_option):
    required = set()
    for edge in metadata['edges']:
        u, v, cost, demand = edge
        if demand > 0:
            required.add(frozenset({u, v}))
    
    processed = set()
    routes = []
    current_route = []
    current_load = 0
    current_position = depot
    
    while len(processed) < len(required):
        # 寻找所有未被处理的候选任务
        candidates = []
        for task in tasks:
            u, v, cost, demand = task
            pair = frozenset({u, v})
            if pair in processed:
                continue
            # 计算当前节点到任务起点的距离
            distance = dist[current_position][u]
            candidates.append((distance, task))
        
        if not candidates:
            break
        
        # 找出距离最小的候选任务，随机选择其中一个
        min_dist = min(c[0] for c in candidates)
        min_candidates = [c for c in candidates if c[0] == min_dist]
        if greedy_option == 1:
            min_candidates += [c for c in candidates if c[0] <= min_dist + 1]

        selected = random.choice(min_candidates)
        selected_dist, selected_task = selected
        u, v, task_cost, demand = selected_task
        pair = frozenset({u, v})
        
        if current_load + demand <= capacity:
            current_route.append((u, v, task_cost, demand))
            current_load += demand
            current_position = v
            processed.add(pair)
        else:
            if current_route:
                routes.append(current_route)
                current_route = []
                current_load = 0
                current_position = depot
            else:
                break
    
    if current_route:
        routes.append(current_route)
    total_cost = calculate_cost(routes, depot, dist, capacity, metadata['edges'])
    return routes, total_cost

def merge_split(routes, depot, capacity, dist, tasks):
    if len(routes) < 2:
        return routes
    
    selected = random.sample(routes, 2)
    merged = [t for route in selected for t in route]
    
    # 应用路径扫描生成单一路由（忽略容量）
    remaining = {t[:2]: t for t in merged}
    new_route = []
    while remaining:
        pos = depot
        route = []
        load = 0
        while True:
            candidates = []
            for (u, v) in list(remaining.keys()):
                t = remaining[(u, v)]
                candidates.append((u, v, t[2], t[3]))
            
            if not candidates:
                break
            
            min_dist = min(dist[pos][t[0]] for t in candidates)
            nearest = [t for t in candidates if dist[pos][t[0]] == min_dist]
            selected = random.choice(nearest)
            
            route.append(selected)
            pos = selected[1]
            del remaining[(selected[0], selected[1])]
            if (selected[1], selected[0]) in remaining:
                del remaining[(selected[1], selected[0])]
        
        new_route.extend(route)
    
    # 应用Ulusoy拆分算法
    split_routes = []
    current_route = []
    current_load = 0
    for t in new_route:
        if current_load + t[3] > capacity:
            split_routes.append(current_route)
            current_route = []
            current_load = 0
        current_route.append(t)
        current_load += t[3]
    if current_route:
        split_routes.append(current_route)
    
    return [r for r in routes if r not in selected] + split_routes

def mutate(routes, depot, capacity, dist, tasks):
    mutation_type = random.choices(
        ['reverse', 'insert', 'swap', 'merge_split', 'two_opt'],
        weights=[0.3, 0.1, 0.2, 0.15, 0.25],
        k=1
    )[0]
    
    new_routes = copy.deepcopy(routes)
    
    try:
        if mutation_type == 'reverse':
            route_idx = random.randint(0, len(new_routes)-1)
            route = new_routes[route_idx]
            if len(route) < 2: return new_routes
            i, j = sorted(random.sample(range(len(route)), 2))
            new_segment = [(t[1], t[0], t[2], t[3]) for t in reversed(route[i:j+1])]
            new_routes[route_idx] = route[:i] + new_segment + route[j+1:]
        
        elif mutation_type == 'insert':
            src_route = random.randint(0, len(new_routes)-1)
            if not new_routes[src_route]: return new_routes
            task = new_routes[src_route].pop(random.randrange(len(new_routes[src_route])))
            dst_route = random.randint(0, len(new_routes))
            if dst_route >= len(new_routes):
                new_routes.append([task])
            else:
                new_routes[dst_route].insert(random.randrange(len(new_routes[dst_route])+1), task)
        
        elif mutation_type == 'swap':
            if len(new_routes) < 2: return new_routes
            r1, r2 = random.sample(range(len(new_routes)), 2)
            if not new_routes[r1] or not new_routes[r2]: return new_routes
            i = random.randrange(len(new_routes[r1]))
            j = random.randrange(len(new_routes[r2]))
            new_routes[r1][i], new_routes[r2][j] = new_routes[r2][j], new_routes[r1][i]
        
        elif mutation_type == 'merge_split':
            new_routes = merge_split(new_routes, depot, capacity, dist, tasks)
    
        elif mutation_type == 'two_opt':
            route_idx = random.randint(0, len(new_routes)-1)
            route = new_routes[route_idx]
            if len(route) < 2: return new_routes
            
            i, j = sorted(random.sample(range(len(route)), 2))
            
            new_segment = route[i:j+1][::-1]
            new_routes[route_idx] = route[:i] + new_segment + route[j+1:]

    except:
        return routes
    
    return new_routes

def genetic_annealing(metadata, tasks, depot, capacity, dist, start_time, time_limit):
    required_tasks = []
    seen = set()
    for t in tasks:
        u, v, cost, demand = t
        if demand > 0 and frozenset({u, v}) not in seen:
            required_tasks.append(t)
            seen.add(frozenset({u, v}))
    
    population = []
    for i in range(5):
        for j in range(2):
            routes, _ = path_scanning(tasks, depot, capacity, dist, i)
            population.append((routes, calculate_cost(routes, depot, dist, capacity, required_tasks)))
    
    greedy_best_routes = []
    greedy_best_cost = float('inf')
    st_time_1 = time.time()
    while (time.time() - st_time_1) < time_limit * 0.1:
        routes, cost = greedy(metadata, tasks, depot, capacity, dist, 0)
        if cost < greedy_best_cost:
            greedy_best_routes = routes
            greedy_best_cost = cost
    population.append((greedy_best_routes, greedy_best_cost))
    if DEBUG:
        print(f"min best cost: {greedy_best_cost}")
    
    greedy_best_routes = []
    greedy_best_cost = float('inf')
    st_time_2 = time.time()
    while (time.time() - st_time_2) < time_limit * 0.1:
        routes, cost = greedy(metadata, tasks, depot, capacity, dist, 1)
        if cost < greedy_best_cost:
            greedy_best_routes = routes
            greedy_best_cost = cost
    population.append((greedy_best_routes, greedy_best_cost))
    if DEBUG:
        print(f"min+1 best cost: {greedy_best_cost}")

    # 排序选择最佳个体
    population.sort(key=lambda x: x[1])
    best_routes, best_cost = population[0]
    
    initial_temp = 1
    cooling_rate = 0.993
    
    while time.time() - start_time < time_limit * 0.9:
        # 模拟退火过程
        new_pop = []
        for routes, cost in population:
            temp = initial_temp
            current_routes, current_cost = copy.deepcopy(routes), cost
            
            while temp > 0.1 and time.time() - start_time < time_limit * 0.9:
                new_routes = mutate(current_routes, depot, capacity, dist, tasks)
                new_cost = calculate_cost(new_routes, depot, dist, capacity, required_tasks)
                
                if new_cost < current_cost or math.exp((current_cost - new_cost)/temp) > random.random():
                    current_routes, current_cost = new_routes, new_cost
                    if new_cost < best_cost:
                        best_routes, best_cost = current_routes, new_cost
                
                temp *= cooling_rate
            
            new_pop.append((current_routes, current_cost))
        
        population = sorted(new_pop + [(best_routes, best_cost)], key=lambda x: x[1])[:8]
        if DEBUG:
            print(f"Current best cost: {best_cost}")
    
    return best_routes, best_cost

def calculate_cost(routes, depot, dist, capacity, required_tasks):
    total_cost = 0
    served = set()
    required = set()
    for t in required_tasks:
        u, v, _, d = t
        if d > 0:
            required.add(frozenset({u, v}))
    for route in routes:
        route_load = 0
        current = depot
        for t in route:
            u, v, cost, demand = t
            # 检查任务是否已服务
            if (u, v) in served or (v, u) in served:
                return float('inf')
            served.add((u, v))
            route_load += demand
            total_cost += dist[current][u] + cost
            current = v
        total_cost += dist[current][depot]
        if route_load > capacity:
            return float('inf')
    # 检查所有必需任务是否被服务
    served_pairs = set()
    for (u, v) in served:
        served_pairs.add(frozenset({u, v}))
    if not required.issubset(served_pairs):
        return float('inf')
    return total_cost

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str)
    parser.add_argument('-t', type=int, required=True)
    parser.add_argument('-s', type=int, required=True)
    args = parser.parse_args()
    start_time = time.time()
    
    metadata, graph = read_carp_file(args.file)
    depot = metadata['DEPOT']
    capacity = metadata['CAPACITY']
    dist = floyd(graph)
    
    tasks = []
    for u in graph:
        for v, cost, demand in graph[u]:
            if demand > 0:
                tasks.append((u, v, cost, demand))
                tasks.append((v, u, cost, demand))
    
    random.seed(args.s)
    best_routes, best_cost = genetic_annealing(metadata, tasks, depot, capacity, dist, start_time, (args.t - 5) * 0.8)
    fi = 0
    for path in best_routes:
        if fi == 0:
            print("s 0", end='')
        else:
            print(",0", end='')
        fi = 1
        for edge in path:
            print(f",({edge[0]},{edge[1]})", end='')
        print(",0", end='')
    print(f"\nq {best_cost}")

if __name__ == "__main__":
    main()
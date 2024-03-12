# Non-Parametric Networks for 3D Point Cloud Part Segmentation
# changed from Point-NN
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from sklearn.cluster import KMeans
from model_utils import *
import math

# 球均匀点分布
def fibonacci_sphere(samples=1):
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append([x, y, z])
    points = torch.tensor(points)
    return points


def normalize(features, dim=-1):
    if features.shape[dim] > 100 :
        return F.normalize(features, dim=dim)
    else:
        return features / (torch.sum(features, dim=dim).unsqueeze(dim)+1e-9)


# FPS + k-NN
class FPS_kNN(nn.Module):
    def __init__(self, group_num, k_neighbors):
        super().__init__()
        self.group_num = group_num  # 
        self.k_neighbors = k_neighbors

    def forward(self, xyz, x, last_xyz_bag=None):
        B, N, _ = xyz.shape

        # FPS
        # import pdb; pdb.set_trace()
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        # fps_idx = farthest_point_sample(xyz, self.group_num).long() 
        lc_xyz = index_points(xyz, fps_idx) # 1, 512, 3
        lc_x = index_points(x, fps_idx)

        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        xyz_bag = None
        if last_xyz_bag is not None:
            direction_percentage, avg_direction, avg_features = last_xyz_bag["percentage"], last_xyz_bag["direction"], last_xyz_bag["features"]
            lc_percentage = index_points(direction_percentage, fps_idx)
            lc_direction = index_points(avg_direction, fps_idx)
            lc_features = index_points(avg_features, fps_idx)
            knn_percentage = index_points(direction_percentage, knn_idx)
            knn_direction = index_points(avg_direction, knn_idx)
            knn_features = index_points(avg_features, knn_idx)
            xyz_bag = {}
            xyz_bag["lc_percentage"] = lc_percentage
            xyz_bag["lc_direction"] = lc_direction
            xyz_bag["lc_features"] = lc_features
            xyz_bag["knn_percentage"] = knn_percentage
            xyz_bag["knn_direction"] = knn_direction
            xyz_bag["knn_features"] = knn_features

        return lc_xyz, lc_x, knn_xyz, knn_x, xyz_bag


# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha # min cluster size
        self.beta = beta 
        self.gamma = gamma
        self.ball_sphere_points = fibonacci_sphere(beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x, last_xyz_bag = None):

        # Normalize x (features) and xyz (coordinates)
        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        knn_xyz = (knn_xyz - mean_xyz) 
        std_xyz = torch.std(knn_xyz)
        
        # B, N, K, 3S  *  betaA, 3S
        xyz_distance = torch.sqrt(knn_xyz[:,:,:,0]**2+knn_xyz[:,:,:,1]**2+knn_xyz[:,:,:,2]**2)
        norm_knn_xyz = knn_xyz / (xyz_distance.unsqueeze(-1) + 1e-8)
        ball_sphere_points = self.ball_sphere_points.to(norm_knn_xyz.device)
        simm = torch.einsum('bnks,as->bnka', norm_knn_xyz, ball_sphere_points) # B, N, K, beta
        simm_max_value, simm_max_idx = torch.max(simm, dim=-1)

        # counts 
        # unique_direction = (avg_direction[:,:,:,0] != 0) & (avg_direction[:,:,:,1] != 0) & (avg_direction[:,:,:,2] != 0)
        counts = []
        for i in range(self.beta):
            if i == 0:
                counts.append(torch.sum(simm_max_idx==i, dim=-1)-1)
            else:
                counts.append(torch.sum(simm_max_idx==i, dim=-1))
        direction_counts = torch.stack(counts, dim=-1)

        # direction
        sum_direction = torch.zeros(simm.shape[0], simm.shape[1], self.beta, knn_xyz.shape[-1]).to(simm.device)
        sum_direction = sum_direction.scatter_add_(-2, simm_max_idx.unsqueeze(-1).long().repeat(1,1,1,knn_xyz.shape[-1]), knn_xyz)
        # 单位长度
        # avg_direction = sum_direction / (torch.sqrt(sum_direction[:,:,:,0]**2+sum_direction[:,:,:,1]**2+sum_direction[:,:,:,2]**2).unsqueeze(-1) + 1e-7)  
        # or保留对应方向长度信息 
        avg_direction = sum_direction / (direction_counts+1e-8).unsqueeze(-1)


        if last_xyz_bag is None:
            # features
            sum_features = torch.zeros(simm.shape[0], simm.shape[1], self.beta, knn_x.shape[-1]).to(simm.device)
            sum_features = sum_features.scatter_add_(-2, simm_max_idx.unsqueeze(-1).long().repeat(1,1,1,knn_x.shape[-1]), knn_x)
            avg_features = normalize(sum_features, dim=-1) 

            # percentage
            k_influence = torch.ones(simm.shape[0], simm.shape[1]).to(simm.device)
            direction_counts = (direction_counts > self.alpha) * direction_counts
            direction_percentage = direction_counts / (torch.sum(direction_counts, dim=-1).unsqueeze(-1)+1e-8)

        if last_xyz_bag is not None:
            last_lc_percentage = last_xyz_bag["lc_percentage"]  # 1, 256, 20
            last_lc_direction = last_xyz_bag["lc_direction"]   # 1, 256, 20, 3
            last_lc_features = last_xyz_bag["lc_features"]   # 1, 256, 20, 768
            last_knn_percentage = last_xyz_bag["knn_percentage"]  # 1, 256, 90, 20
            last_knn_direction = last_xyz_bag["knn_direction"]   # 1, 256, 90, 20, 3
            last_knn_features = last_xyz_bag["knn_features"]   # 1, 256, 90, 20, 768

            # norm_knn_xyz = knn_xyz / (torch.sqrt(knn_xyz[:,:,:,0]**2+knn_xyz[:,:,:,1]**2+knn_xyz[:,:,:,2]**2).unsqueeze(-1) + 1e-7)
            lc_distance = torch.sqrt(last_lc_direction[:,:,:,0]**2+last_lc_direction[:,:,:,1]**2+last_lc_direction[:,:,:,2]**2)
            last_lc_direction = last_lc_direction / (lc_distance.unsqueeze(-1) + 1e-8)
            knn_distance = torch.sqrt(last_knn_direction[:,:,:,:,0]**2+last_knn_direction[:,:,:,:,1]**2+last_knn_direction[:,:,:,:,2]**2)
            last_knn_direction = last_knn_direction / (knn_distance.unsqueeze(-1) + 1e-8)

            # 流形分析
            simm_lc = torch.einsum('bnks,bnas->bnka', norm_knn_xyz, last_lc_direction) # 1, 256, 90, 3/ 1, 256, 20, 3 和中心点哪个原方向更接近, 之前根本就没有值的话就是0
            simm_knn = torch.einsum('bnks,bnkas->bnka', norm_knn_xyz, last_knn_direction) # 和哪个相邻点的原方向更接近
            # lc_mask = torch.abs(simm_lc) > self.gamma   # 反方向在同一条线上也算进去了
            # knn_mask = torch.abs(simm_knn) > self.gamma
            lc_mask = simm_lc > self.gamma   # or 反方向在同一条线上不算进去， 这样可以减少中间隔了一块区域的问题 或者是只算0.5这样？
            knn_mask = simm_knn > self.gamma
            lc_mask_reverse = -simm_lc > self.gamma
            knn_mask_reverse = -simm_knn > self.gamma
            lc_mask_same_line = lc_mask + lc_mask_reverse
            knn_mask_same_line = knn_mask + knn_mask_reverse
            # print(((torch.abs(simm_lc) > self.gamma).long() == lc_mask_same_line.long()).min()) # 验证两者相等
            last_lc_percentage = last_lc_percentage.unsqueeze(-2).repeat(1,1,last_knn_percentage.shape[2],1)
            lc_percentage = last_lc_percentage * lc_mask_same_line
            lc_percentage = torch.sum(lc_percentage, dim=-1)
            knn_percentage = last_knn_percentage * knn_mask_same_line
            knn_percentage = torch.sum(knn_percentage, dim=-1)
            k_percentage = lc_percentage * knn_percentage
            
            # 距离分析
            lc_distance = lc_distance.unsqueeze(-2).repeat(1,1,knn_distance.shape[2],1)
            lc_d = lc_distance * lc_mask
            lc_d,_ = torch.max(lc_d, dim=-1)
            lc_d_reverse = lc_distance * lc_mask_reverse
            lc_d_reverse,_ = torch.max(lc_d_reverse, dim=-1)
            knn_d = knn_distance * knn_mask
            knn_d,_ = torch.max(knn_d, dim=-1)
            knn_d_reverse = knn_distance * knn_mask_reverse
            knn_d_reverse,_ = torch.max(knn_d_reverse, dim=-1)
            lc_knn_mask = ((lc_d + lc_d_reverse) != 0) & ((knn_d + knn_d_reverse) != 0)
            lc_knn_d = ((lc_d + lc_d_reverse) + (knn_d + knn_d_reverse)) * lc_knn_mask  # * or + ?之后试试
            xyz_d = xyz_distance + lc_d_reverse + knn_d
            xyz_d = xyz_d ** 1 + 1e-8    # change here
            lc_knn_d /= xyz_d
            # or *
            # lc_knn_d = (lc_d + lc_d_reverse) * (knn_d + knn_d_reverse)
            # xyz_d = xyz_distance + lc_d_reverse + knn_d_reverse
            # xyz_d = xyz_d ** 2.2 + 1e-7
            # lc_knn_d /= xyz_d
            
            k_percentage = k_percentage * lc_knn_d
            k_influence = torch.sum(k_percentage, dim=-1)
            k_percentage = k_percentage / (k_influence.unsqueeze(-1)+1e-7) # 1, 256, 90
            direction_percentage = torch.zeros(simm.shape[0], simm.shape[1], self.beta).to(simm.device)
            direction_percentage = direction_percentage.scatter_add_(-1, simm_max_idx.long(), k_percentage)
            k_avg_influence = k_influence.sum() / k_influence.shape[-1]
            k_influence = k_influence / (k_avg_influence.item()+1e-8)
            
            # features 也要加权吗?
            # import pdb; pdb.set_trace()
            sum_features = torch.zeros(simm.shape[0], simm.shape[1], self.beta, knn_x.shape[-1]).to(simm.device)
            # knn_x *= k_percentage.unsqueeze(-1).repeat(1,1,1,knn_x.shape[-1])
            sum_features = sum_features.scatter_add_(-2, simm_max_idx.unsqueeze(-1).long().repeat(1,1,1,knn_x.shape[-1]), knn_x)
            avg_features = normalize(sum_features, dim=-1) 

        # 防止无影响
        return knn_x, direction_percentage, avg_direction, avg_features, k_influence


# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, last_xyz_bag, knn_x, lc_x):
        # Feature Aggregation (Pooling)
        # or only avg pooling?
        direction_percentage, avg_direction, avg_features, k_influence = last_xyz_bag["percentage"], last_xyz_bag["direction"], last_xyz_bag["features"], last_xyz_bag["influence"]
        lc_x_max,_ = torch.max(knn_x, dim=-2)
        direction_length = torch.sqrt(avg_direction[:,:,:,0]**2+avg_direction[:,:,:,1]**2+avg_direction[:,:,:,2]**2)
        weight = 1 / (direction_length+1e-8)
        direction_percentage = normalize(direction_percentage * weight, dim=-1)
        lc_x_avg = torch.einsum('bna,bnad->bnd', direction_percentage, avg_features)
        # lc_max, _ = torch.max(lc_x_avg, dim=-1)
        # import pdb; pdb.set_trace()
        lc_x = k_influence.unsqueeze(-1) * lc_x_avg + 0.1*lc_x + lc_x_max * 1e-8   #####                    # change here
        lc_x = normalize(lc_x, dim=-1)
        return lc_x


# Non-Parametric Encoder
class EncNP(nn.Module):  
    def __init__(self, input_points, num_stages, embed_dim, k_neighbors, alpha, beta, gamma, delta):
        super().__init__()
        self.input_points = input_points  # 1024
        self.num_stages = num_stages   # 5
        self.embed_dim = embed_dim    # 144
        self.alpha, self.beta, self.gamma = alpha, beta, gamma   # 1000, 100
        self.delta = delta

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim
        group_num = self.input_points

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # out_dim = out_dim * 2
            group_num = group_num // self.delta  # 这个要加多一点
            k = k_neighbors # / (2*(i==0) + 1)
            # import pdb; pdb.set_trace()
            self.FPS_kNN_list.append(FPS_kNN(group_num, int(k)))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta, self.gamma))
            self.Pooling_list.append(Pooling(out_dim))


    def forward(self, xyz, x):

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, C, N]
        bag_list = []

        # Multi-stage Hierarchy
        last_xyz_bag = None
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x, last_xyz_bag = self.FPS_kNN_list[i](xyz, x, last_xyz_bag)
            # Local Geometry Aggregation
            knn_x, direction_percentage, avg_direction, avg_features, k_influence = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x, last_xyz_bag)
            last_xyz_bag = {
                "percentage" : direction_percentage, 
                "direction" : avg_direction, 
                "features" : avg_features,
                "influence" : k_influence
            }
            # Pooling
            x = self.Pooling_list[i](last_xyz_bag, knn_x, lc_x)

            xyz_list.append(xyz)
            x_list.append(x)
            bag_list.append(last_xyz_bag)
            
        return xyz_list, x_list, bag_list


# Non-Parametric Network
class Point_NN_Seg(nn.Module):
    def __init__(self, input_points=2048, num_stages=5, embed_dim=768, 
                    k_neighbors=90, de_neighbors=6, beta=20, alpha=3, gamma=0.97, delta = 3):
        super().__init__()
        # Non-Parametric Encoder and Decoder
        self.EncNP = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta, gamma, delta)
        self.DecNP = DecNP(num_stages, de_neighbors, gamma, delta)

    def forward(self, xyz, x):
        # xyz: point coordinates
        # x: point features
        xyz = xyz.permute(0, 2, 1)

        # Non-Parametric Encoder
        xyz_list, x_list, bag_list = self.EncNP(xyz, x)

        # Non-Parametric Decoder
        x = self.DecNP(xyz_list, x_list, bag_list)
        return x


# Non-Parametric Decoder
class DecNP(nn.Module):  
    def __init__(self, num_stages, de_neighbors, gamma, delta):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors
        self.gamma = gamma
        self.delta = delta


    def propagate(self, xyz1, xyz2, points1, points2, bag):
        """
        Input:
            xyz1: input points position data, [B, N, 3]              1, 128, 3
            xyz2: sampled input points position data, [B, S, 3]      1, 64, 3
            points1: input points data, [B, D', N]                   1, 128, 768
            points2: input points data, [B, D'', S]                  1, 64, 768
            percentages = posibility in every directions             1, 64, 20 
            directions = directions                                  1, 64, 20, 3
            features = features in every directions                  1, 64, 20, 768
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        percentages = bag["percentage"]
        directions = bag["direction"]
        directions = directions / (torch.sqrt(directions[:,:,:,0]**2+directions[:,:,:,1]**2+directions[:,:,:,2]**2).unsqueeze(-1)+1e-8)
        features = bag["features"]

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        dists = square_distance(xyz1, xyz2)            # 1, 128, 64
        dists, idx = dists.sort(dim=-1)
        dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]
        
        de_k_directions = xyz2.squeeze(0)[idx]
        de_k_directions = de_k_directions - xyz1.unsqueeze(-2).repeat(1,1,de_k_directions.shape[2],1)
        de_k_distace = torch.sqrt(de_k_directions[:,:,:,0]**2+de_k_directions[:,:,:,1]**2+de_k_directions[:,:,:,2]**2)
        de_k_directions = de_k_directions / (de_k_distace.unsqueeze(-1)+1e-8)      # 1, 128, 10, 3

        k_weight = percentages.squeeze(0)[idx]         # 1, 128, 10, 20
        k_directions = directions.squeeze(0)[idx]         # 1, 128, 10, 20, 3
        de_k_directions = de_k_directions.unsqueeze(-2).repeat(1,1,1,k_directions.shape[-2],1)
        de_k_simm = torch.sum(k_directions * de_k_directions, dim=-1)  # 和自己咋办？后面能直接交互到自己吗？
        # de_k_mask_reverse = -de_k_simm > self.gamma  # 要不要加?
        # de_k_mask = de_k_simm > self.gamma
        # import pdb; pdb.set_trace()
        de_k_mask = torch.abs(de_k_simm) > self.gamma
        de_k_weight = torch.sum(de_k_mask * k_weight, dim=-1)
        de_k_weight_sum = torch.sum(de_k_weight, dim=-1) + 1e-8        # 周围点的影响力度, 防止无影响
        de_k_weight_norm = de_k_weight / (de_k_weight_sum.unsqueeze(-1))
        de_k_weight_norm = 1 * de_k_weight_norm + 0 * de_k_weight_sum.unsqueeze(-1) + 1e-6  # here，要不要中和一下？


        dist_recip = 1.0 / (de_k_distace ** 2 + 1e-10) 
        # dist_recip = 1.0 / (de_k_distace + 1e-10) 
        dist_recip = de_k_weight_norm + 1e-10
        # dist_recip = dist_recip * de_k_weight_norm + 1e-10 # try to # this
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / (norm+1e-8)
        weight = weight.view(B, N, self.de_neighbors, 1)

        # index_points(xyz1, idx)  
        interpolated_points = torch.sum(index_points(points2, idx) * weight, dim=2)

        if N == self.delta * S:
            new_points = normalize(interpolated_points * de_k_weight_sum.unsqueeze(-1) + (1e-8 + 0.3 * (de_k_weight_sum.sum()/de_k_weight_sum.shape[-1]).item()) * points1, dim=-1)  # change here
        else:
            new_points = normalize(interpolated_points * de_k_weight_sum.unsqueeze(-1) + (1e-8 + 0.01 * (de_k_weight_sum.sum()/de_k_weight_sum.shape[-1]).item()) * points1, dim=-1)
        # import pdb; pdb.set_trace()
        # torch.isnan(new_points).sum()
        # mm, _ = torch.max(new_points, dim=-1)
        return new_points


    def forward(self, xyz_list, x_list, bag_list):
        xyz_list.reverse()
        x_list.reverse()
        bag_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x, bag_list[i])
        return x
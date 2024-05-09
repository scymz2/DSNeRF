import numpy as np
import os, imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json
import logging



########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    # 存放数据集的位姿和边界信息(深度范围) [images_num,17] eg:[20,17] 最后两位存储边界信息
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))

    # [images_num,17]--->[3,5,images_num] eg:[3,5,20]
    # 3×3是旋转矩阵R,3×1(第4列)是平移矩阵T,3×1(第5列)是h,w,f
    # reshape(-1,3,5) -1表示将自动计算第一个维度的大小，所以这步之后数组变成[images_num, 3, 5]
    # transpose([1,2,0])则表示将数组的维度按照提供的顺序调换， 所以数组维度的结果如下
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N

    # bounds 边界(深度)范围[2,images_num] eg:[2,20]
    bds = poses_arr[:, -2:].transpose([1,0])
    
    # 获取第一张图像的地址,图像必须是jpg或png格式
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]

    # 获取图像shape[h,w,c]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    # poses: [3 5 N], 不用管数组形状如何变化，我们按照原来的方式访问，只不过改一下顺序，
    # 在原来的[N,3,5]中，我们访问hw是 [:, :2, 4]
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1]) # 更新相机的内参矩阵中与图像尺寸有关的部分, 修改h,w
    poses[2, 4, :] = poses[2, 4, :] * 1./factor # 位姿矩阵的第四列的第三行，这通常对应于相机内参矩阵中的焦距, 修改f
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            # return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f, apply_gamma=False)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)


# 用于构造相机矩阵， 在图形学中，z轴通常定义为摄像机的视线方向，而Y轴定义为摄像机的上方向。
# 通过Y轴和Z轴的叉乘，可以得到一个垂直于这两个轴的X轴。
# 第二步，再次用Z轴和新的X轴叉乘得到新的Y轴：这一步是为了纠正和精确Y轴的方向。
# 原始的Y轴可能与Z轴不完全垂直，通过Z轴与新计算的X轴叉乘，可以得到一个完全垂直于Z轴和X轴的新Y轴。
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0) # 平移向量中心
    vec2 = normalize(poses[:, :3, 2].sum(0)) # Z
    up = poses[:, :3, 1].sum(0) # up
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = [] # 初始化一个列表来存储所有计算出的渲染位姿
    rads = np.array(list(rads) + [1.]) # 将半径参数rads转换为NumPy数组，并添加一个额外的维度，通常这个维度被设置为1（没有缩放）
    hwf = c2w[:,4:5] # 从c2w矩阵中提取HWF信息（高度、宽度和焦距）
    
    # 迭代生成每个位姿
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        # 计算相机中心c：通过旋转和缩放rads向量来确定相机在螺旋路径上的位置
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        # 计算观察方向z：相机的前向矢量，指向从相机位置指向焦点
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        # 构造完整的位姿矩阵并添加到列表中：使用viewmatrix函数生成视图矩阵，然后与HWF信息拼接
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses # 返回所有渲染位姿的列表
    


def recenter_poses(poses):
    # 中心化相机位姿
    # 1.计算平均位姿：首先，从一系列相机位姿中计算出一个平均位姿。
    # 2.位姿的逆：取这个平均位姿的逆。位姿矩阵通常包含旋转和平移成分，其逆矩阵相应地包含逆旋转和逆平移，这可以用来将任何以该位姿为参照的坐标转换回世界坐标系的原点。
    # 3.左乘所有位姿：使用这个逆变换矩阵左乘（前乘）所有的相机位姿。变换过程：当你将每个相机位姿与平均位姿的逆相乘时，实际上是在执行以下操作：首先通过平均位姿的逆将所有的位姿从世界坐标系转换到以平均位姿为原点的坐标系。这相当于重新定义了坐标系统的原点和方向，使得平均位姿成为了新的“原点”，并且与世界坐标系的方向一致。
    # 归一化的影响：通过这样的归一化，你可以把所有相机的光心（即位姿矩阵中的平移部分所指示的位置）的平均位置移动到新坐标系统的原点。这也意味着，所有的相机位姿都被转换到了一个以平均位姿为中心的坐标系中。在这个新的坐标系中，平均位姿变成了单位矩阵（没有旋转和平移），即完全与标准的世界坐标系重合。

    # 我的理解：转换之后平均位姿成为了世界坐标的原点，所有相机位姿都以这个世界坐标系为参考

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    # 让位姿[3x4]变成[4x4]
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    # 位姿的旋转矩阵R的第三列（z轴相关）方向向量
    rays_d = poses[:,:3,2:3]

    # 位姿的平移矩阵t 相机光心
    rays_o = poses[:,:3,3:4]

    # 找到离所有相机中心射线距离之和最短的点
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    # 简单理解为场景的中心位置
    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist

    # 所有相机光心到场景中心的方向向量的平均距离向量(xyz轴上)
    up = (poses[:,:3,3] - center).mean(0)

    # 归一化:平均单位向量
    vec0 = normalize(up)

    # 找到两两垂直的单位方向向量
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center

    # 构建坐标系
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    # 求c2w的逆矩阵,并与poses进行矩阵运算,目的是完成所有相机位姿的归一化
    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    # 将所有相机的位置缩放到单位圆内。
    # 理解为归一化后所有光心距离的平均
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    # 缩放因子
    sc = 1./rad

    # 缩放光心
    poses_reset[:,:3,3] *= sc
    
    # 缩放边界
    bds *= sc

    # 归一化
    rad *= sc
    
    # 平均光心位置
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]  # 平均光心z轴距离
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        # 构建坐标系
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    # 新视角：拼接在一起
    new_poses = np.stack(new_poses, 0)
    
    # [num,3,5] 新视角位姿都拼接了原始位姿的起始位姿
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    
    # [num,3,5] 旋转平移后的新位姿都拼接了原始位姿的起始位姿
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    
    # poses[3,5,N] N是数据集个数, 3×3是旋转矩阵R，3×1(第4列)是平移矩阵T，3×1(第5列)是h,w,f
    # bds[2,N] 采样far,near信息,即深度值范围
    # imgs[h,w,c,N] c代表颜色通道

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # print('poses_bound.npy:\n', poses[:,:,0])

    # 重新排列相机姿态矩阵的列，调整方向 [x,y,z,t,whf]--->[y,-x,z,t,whf]
    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    
    # 变换维度:调换第-1轴到第0轴位置
    poses = np.moveaxis(poses, -1, 0).astype(np.float32) # [N,3,5]
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)   # [N,h,w,c]
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)     # [N,2]
    print("bds:", bds[0])
    
    # Rescale if bd_factor is provided
    # 获得缩放因子,以bds.min为基准,有点类似归一化
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)

    # 对位姿的平移矩阵t进行缩放
    poses[:,:3,3] *= sc

    # 对边界进行缩放
    bds *= sc
    
    # print('before recenter:\n', poses[0])


    # 计算pose的均值,将所有pose做个均值逆转换
    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 找到平均位姿，以平均位姿中心作为新的坐标系原点
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## 获取螺旋路径
        # 获取平均位姿
        up = normalize(poses[:, :3, 1].sum(0)) # 计算所有位姿上向量的平均，用于确定相机的“上”方向

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5. # 获取近景和远景深度
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth)) # 计算平均视距
        focal = mean_dz # 设置聚焦距离

        # 获取螺旋路径的半径
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T # 获取所有位姿的平移部分
        rads = np.percentile(np.abs(tt), 90, 0) # 计算90百分位数作为路径半径
        c2w_path = c2w
        N_views = 120  # 视图数量
        N_rots = 2  # 旋转次数
        if path_zflat:
            # 如果路径应该在z轴上平坦
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2] # 调整路径的z坐标
            rads[2] = 0. # 设置z方向半径为0
            N_rots = 1
            N_views/=2  # 减少视图数量

        # 生成螺旋路径的位姿
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    # 读取images.bin 和point3D.bin然后保存为colmap_depth.npy
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    # 计算所有3D点的平均投影误差
    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    # 获取相机姿态
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    # 根据bd_factor调整近点和远点的范围, scale_factor调整深度范围
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    # 初始化数据列表，用于存储处理后的深度信息
    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            # 根据图像id取对应的2D坐标点和他们在三维空间中对应的id
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            # 忽略不在近点远点范围内的深度值
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            # 计算权重：权重的计算方式使用了投影误差来调整每个3D点的贡献
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

def load_sensor_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depths = [imageio.imread(f) for f in depthfiles]
    depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

def load_colmap_llff(basedir):
    basedir = Path(basedir)
    current_directory = os.getcwd()
    #logging.error(f'Current Directory: {current_directory}')
    train_imgs = np.load(basedir / 'train_images.npy')
    test_imgs = np.load(basedir / 'test_images.npy')
    train_poses = np.load(basedir / 'train_poses.npy')
    test_poses = np.load(basedir / 'test_poses.npy')
    video_poses = np.load(basedir / 'video_poses.npy')
    depth_data = np.load(basedir / 'train_depths.npy', allow_pickle=True)
    bds = np.load(basedir / 'bds.npy')

    return train_imgs, test_imgs, train_poses, test_poses, video_poses, depth_data, bds

    


import matplotlib.pyplot as plt
import numpy as np
import os
import ffmpeg
import imageio
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2

# colors for visualization
colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,8],[8,7],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14]]
)
body_edges_mocap = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)

def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def draw_mocap(base_dir, group, pred, gt):
    #base_dir 表示存储当前序列的小文件夹名
    #group 表示当前存储视频文档的前缀
    #pred 表示预测结果
    #gt 表示ground truth
    fig = plt.figure(figsize=(20, 9))
            # fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')        
    
    length_=120
    i=0
    p_x=np.linspace(-5,5,25)
    p_y=np.linspace(-5,5,25)
    X,Y=np.meshgrid(p_x,p_y)

    while i < length_:
        ax.lines = []
        for x_i in range(p_x.shape[0]):
                    temp_x=[p_x[x_i],p_x[x_i]]
                    temp_y=[p_y[0],p_y[-1]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
        for y_i in range(p_x.shape[0]):
                    temp_x=[p_x[0],p_x[-1]]
                    temp_y=[p_y[y_i],p_y[y_i]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
        for j in range(pred.shape[0]):
                    xs=pred[j,i,:,0]
                    # print(xs)
                    ys=pred[j,i,:,1]
                    zs=pred[j,i,:,2]
                    alpha=1
                    ax.plot( zs,xs, ys, 'y.',alpha=alpha)
                    
                    if True:
                        x=gt[j,i,:,0]
                        y=gt[j,i,:,1]
                        z=gt[j,i,:,2]
                        ax.plot(z, x, y, 'y.')

                    plot_edge=True
                    if plot_edge:
                        for edge in body_edges_mocap:
                            x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                            y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                            z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                            if i>=30:
                                ax.plot(z, x, y,zdir='z',c='blue',alpha=alpha)
                            else:
                                ax.plot(z, x, y,zdir='z',c='purple',alpha=alpha)
                            if True:
                                x=[gt[j,i,edge[0],0],gt[j,i,edge[1],0]]
                                y=[gt[j,i,edge[0],1],gt[j,i,edge[1],1]]
                                z=[gt[j,i,edge[0],2],gt[j,i,edge[1],2]]
                                if i>=30:
                                    ax.plot(z, x, y,zdir='z',c='red',alpha=alpha)
                                else:
                                    ax.plot(z, x, y,zdir='z',c='purple',alpha=alpha)
                            
                    ax.set_xlim3d([-2 , 5])
                    ax.set_ylim3d([-4 , 3])
                    ax.set_zlim3d([0,4])
                    ax.set_axis_off()
                    plt.title(str(i),y=-0.1)
        plt.pause(0.1)
        prefix = '{:02}'.format(i)
        filename = base_dir + prefix
        plt.savefig(filename)
        i += 1

    plt.savefig(filename)
    plt.close()
            
    pics_3d = base_dir + '/%2d.png'
    out_dir_3d = base_dir + f'{group}_3d.mp4'
    (
                ffmpeg
                .input(pics_3d, framerate=30)
                .output(out_dir_3d)
                .run()
    )

def draw(base_dir, group, pred, gt):
    #base_dir 表示存储当前序列的小文件夹名
    #group 表示当前存储视频文档的前缀
    #pred 表示预测结果
    #gt 表示ground truth
    fig = plt.figure(figsize=(20, 9))
            # fig.tight_layout()
    ax = fig.add_subplot(111, projection='3d')        
    
    length_=100
    i=0
    p_x=np.linspace(-5,5,25)
    p_y=np.linspace(-5,5,25)
    X,Y=np.meshgrid(p_x,p_y)

    while i < length_:
        ax.lines = []
        for x_i in range(p_x.shape[0]):
                    temp_x=[p_x[x_i],p_x[x_i]]
                    temp_y=[p_y[0],p_y[-1]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
        for y_i in range(p_x.shape[0]):
                    temp_x=[p_x[0],p_x[-1]]
                    temp_y=[p_y[y_i],p_y[y_i]]
                    z=[0,0]
                    ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
        for j in range(pred.shape[0]):
                    xs=pred[j,i,:,0]
                    # print(xs)
                    ys=pred[j,i,:,1]
                    zs=pred[j,i,:,2]
                    alpha=1
                    ax.plot( xs, ys,zs, 'y.',alpha=alpha)
                    
                    if True:
                        x=gt[j,i,:,0]
                        y=gt[j,i,:,1]
                        z=gt[j,i,:,2]
                        ax.plot(x, y, z, 'y.')

                    plot_edge=True
                    if plot_edge:
                        for edge in body_edges:
                            x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                            y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                            z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                            if i>=25:
                                ax.plot(x, y, z,zdir='z',c='blue',alpha=alpha)
                            else:
                                ax.plot(x, y, z,zdir='z',c='purple',alpha=alpha)
                            if True:
                                x=[gt[j,i,edge[0],0],gt[j,i,edge[1],0]]
                                y=[gt[j,i,edge[0],1],gt[j,i,edge[1],1]]
                                z=[gt[j,i,edge[0],2],gt[j,i,edge[1],2]]
                                if i>=30:
                                    ax.plot(x, y, z,zdir='z',c='red',alpha=alpha)
                                else:
                                    ax.plot(x, y, z,zdir='z',c='purple',alpha=alpha)
                            
                    ax.set_xlim3d([-2 , 5])
                    ax.set_ylim3d([-4 , 3])
                    ax.set_zlim3d([0,4])
                    ax.set_axis_off()
                    plt.title(str(i),y=-0.1)
        plt.pause(0.1)
        prefix = '{:02}'.format(i)
        filename = base_dir + prefix
        plt.savefig(filename)
        i += 1

    plt.savefig(filename)
    plt.close()
            
    pics_3d = base_dir + '/%2d.png'
    out_dir_3d = base_dir + f'{group}_3d.mp4'
    (
                ffmpeg
                .input(pics_3d, framerate=30)
                .output(out_dir_3d)
                .run()
    )

def draw_all(pred, gt, save_path, fps=25, plot_gt=True):  
    # pred/gt: [M, T, J*3] or [M, T, J, 3]
    assert pred.shape==gt.shape
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,8],[7,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14]])
    M, vlen = gt.shape[:2]
    gt = gt.reshape([M, vlen, -1, 3])
    pred = pred.reshape([M, vlen, -1, 3])
    with imageio.get_writer(save_path, fps=fps) as writer:
        for i in tqdm(range(vlen)):
            fig = plt.figure(figsize=(10, 4.5))
            ax = fig.add_subplot(111, projection='3d')        

            p_x=np.linspace(-10,10,25)
            p_y=np.linspace(-10,10,25)
            X,Y=np.meshgrid(p_x,p_y)
    
            for x_i in range(p_x.shape[0]):
                        temp_x=[p_x[x_i],p_x[x_i]]
                        temp_y=[p_y[0],p_y[-1]]
                        z=[0,0]
                        ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
            for y_i in range(p_x.shape[0]):
                        temp_x=[p_x[0],p_x[-1]]
                        temp_y=[p_y[y_i],p_y[y_i]]
                        z=[0,0]
                        ax.plot(temp_x,temp_y,z,color='black',alpha=0.1)
            for j in range(M):
                        xs=pred[j,i,:,0]
                        # print(xs)
                        ys=pred[j,i,:,1]
                        zs=pred[j,i,:,2]
                        alpha=1
                        ax.plot( xs, ys,zs, 'y.',alpha=alpha)

                        if plot_gt:
                            x=gt[j,i,:,0]
                            y=gt[j,i,:,1]
                            z=gt[j,i,:,2]
                            ax.plot( x, y, z, 'y.')

                        plot_edge=True
                        if plot_edge:
                            for edge in body_edges:
                                x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                                y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                                z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                                if i>=25:
                                    ax.plot(x, y, z,zdir='z',c='blue',alpha=alpha)
                                else:
                                    ax.plot(x, y, z,zdir='z',c='purple',alpha=alpha)
                            
                                if plot_gt:
                                    x=[gt[j,i,edge[0],0],gt[j,i,edge[1],0]]
                                    y=[gt[j,i,edge[0],1],gt[j,i,edge[1],1]]
                                    z=[gt[j,i,edge[0],2],gt[j,i,edge[1],2]]
                                    if i>=25:
                                        ax.plot(x, y, z,zdir='z',c='red',alpha=alpha)
                                    else:
                                        ax.plot(x, y, z,zdir='z',c='purple',alpha=alpha)

                        ax.set_xlim3d([-2 , 5])
                        ax.set_ylim3d([-4 , 3])
                        ax.set_zlim3d([0,4])
                        ax.set_axis_off()
                        plt.title(str(i),y=-0.1)

            frame_vis = get_img_from_fig(fig)
            writer.append_data(frame_vis)


def draw_fig(pred, gt, save_path, idx=[0,10,20], fps=25):  
    # pred/gt: [M, T, J*3] or [M, T, J, 3]
    assert pred.shape==gt.shape
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,8],[7,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14]])
    M, vlen = gt.shape[:2]
    gt = gt.reshape([M, vlen, -1, 3])
    pred = pred.reshape([M, vlen, -1, 3])
    i = idx
    # with imageio.get_writer(save_path, fps=fps) as writer:
    #     for i in tqdm(range(vlen)):
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111, projection='3d')        

    p_x=np.linspace(-10,10,25)
    p_y=np.linspace(-10,10,25)
    X,Y=np.meshgrid(p_x,p_y)

    for x_i in range(p_x.shape[0]):
        temp_x=[p_x[x_i],p_x[x_i]]
        temp_y=[p_y[0],p_y[-1]]
        z=[0,0]
        ax.plot(temp_x,temp_y,z,color='black',alpha=0.05)
    for y_i in range(p_x.shape[0]):
        temp_x=[p_x[0],p_x[-1]]
        temp_y=[p_y[y_i],p_y[y_i]]
        z=[0,0]
        ax.plot(temp_x,temp_y,z,color='black',alpha=0.05)

    alphas = [0.9, 0.6, 0.3]
    for i, a in zip(idx, alphas):
        for j in range(M):
            for edge in body_edges:
                x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                # if i>=25:
                ax.plot(x, y, z,zdir='z', c='#235f25', alpha=a)
                        
            xs=pred[j,i,:,0]
            ys=pred[j,i,:,1]
            zs=pred[j,i,:,2]
            ax.plot(xs, ys,zs, '.', color='#53b070', alpha=a, markersize=2)

            ax.set_xlim3d([-1 , 4])
            ax.set_ylim3d([-3 , 2])
            ax.set_zlim3d([0, 3.2])
            ax.set_axis_off()
            # plt.title(str(i),y=-0.1)
    plt.savefig(save_path)
    # frame_vis = get_img_from_fig(fig)
    # writer.append_data(frame_vis)



def draw_figs(pred, save_path, idx=[0,10,20],palette=['#235f25','#53b070','#480ca8','#7209b7']):  
    # pred/gt: [M, T, J*3] or [M, T, J, 3]
    # there it should be four colours in palette, 0&2 for joint color, 1&3 for bone color
    body_edges = np.array([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,8],[7,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14]])
    M, vlen = pred.shape[:2]
    # gt = gt.reshape([M, vlen, -1, 3])
    pred = pred.reshape([M, vlen, -1, 3])
    i = idx
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot(111, projection='3d')        
    ax.view_init(elev=20, azim=-60)
    p_x=np.linspace(-5,15,25)
    p_y=np.linspace(-5,15,25)
    X,Y=np.meshgrid(p_x,p_y)

    # alphas = [0.9, 0.6, 0.3]
    alphas  = [1] * len(idx)
    for i, a in zip(idx, alphas):
        
        ax.lines = []

        for x_i in range(p_x.shape[0]):
                temp_x=[p_x[x_i],p_x[x_i]]
                temp_y=[p_y[0],p_y[-1]]
                z=[0,0]
                ax.plot(temp_x,temp_y,z,color='black',alpha=0.03)
        for y_i in range(p_x.shape[0]):
                temp_x=[p_x[0],p_x[-1]]
                temp_y=[p_y[y_i],p_y[y_i]]
                z=[0,0]
                ax.plot(temp_x,temp_y,z,color='black',alpha=0.03)

        for j in range(M):
            for edge in body_edges:
                x=[pred[j,i,edge[0],0],pred[j,i,edge[1],0]]
                y=[pred[j,i,edge[0],1],pred[j,i,edge[1],1]]
                z=[pred[j,i,edge[0],2],pred[j,i,edge[1],2]]
                if i>=25:
                    ax.plot(x, y, z,zdir='z', c=palette[3], alpha=a)
                else:
                    ax.plot(x, y, z,zdir='z', c=palette[1], alpha=a)
            xs=pred[j,i,:,0]
            ys=pred[j,i,:,1]
            zs=pred[j,i,:,2]           
            if i >= 25:
                ax.plot(xs, ys,zs, '.', color=palette[2], alpha=a, markersize=2)
            else:
                ax.plot(xs, ys,zs, '.', color=palette[0], alpha=a, markersize=2)

            ax.set_xlim3d([-0.5 , 3])
            ax.set_ylim3d([-2.5 , 1])
            ax.set_zlim3d([0, 3])
            ax.set_axis_off()
        plt.savefig(save_path+f'{i}.pdf')
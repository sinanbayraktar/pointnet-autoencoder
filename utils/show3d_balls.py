""" Original Author: Haoqiang Fan """
import numpy as np
import ctypes as ct
import cv2
import sys
import os
from colour import Color


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz=800
mousex,mousey=0.5,0.5
zoom=1.0
changed=True
def onmouse(*args):
    global mousex,mousey,changed
    y=args[1]
    x=args[2]
    mousex=x/float(showsz)
    mousey=y/float(showsz)
    changed=True
cv2.namedWindow('show3d')
cv2.moveWindow('show3d',0,0)
cv2.setMouseCallback('show3d',onmouse)

dll=np.ctypeslib.load_library(os.path.join(BASE_DIR, 'render_balls_so'),'.')

def showpoints(xyz, c_gt=None, c_pred=None ,waittime=0, showrot=False, magnifyBlue=0,
              freezerot=False, background=(0,0,0), normalizecolor=True, ballradius=10, 
              gradient=False):
    
    global showsz,mousex,mousey,zoom,changed
    num_points = len(xyz)
    xyz=xyz-xyz.mean(axis=0)
    radius=((xyz**2).sum(axis=-1)**0.5).max()
    xyz/=(radius*2.2)/showsz
    if not gradient:
        if c_gt is None and c_pred is None:
            c0=np.zeros((num_points,),dtype='float32')+255
            c1=np.zeros((num_points,),dtype='float32')+255
            c2=np.zeros((num_points,),dtype='float32')+255
        elif c_gt is not None and c_pred is None:
            c0=c_gt[:,0]
            c1=c_gt[:,1]
            c2=c_gt[:,2]
        elif c_gt is None and c_pred is not None:
            c0=c_pred[:,0]
            c1=c_pred[:,1]
            c2=c_pred[:,2]
        else:
            print("There are 2 colors specified, using gt colors")
            c0=c_gt[:,0]
            c1=c_gt[:,1]
            c2=c_gt[:,2]
    else: # gradient from red to blue for all the points 
        red = Color("red")
        colors = list(red.range_to(Color("blue"),num_points))
        colors_rgb = [colors[i].rgb for i in range(len(colors))]
        colors_rgb_arr = np.array(colors_rgb, dtype='float32')
        colors_rgb_arr *= 255
        c0 = colors_rgb_arr[:,0]
        c1 = colors_rgb_arr[:,1]
        c2 = colors_rgb_arr[:,2]

    if normalizecolor:
        c0/=(c0.max()+1e-14)/255.0
        c1/=(c1.max()+1e-14)/255.0
        c2/=(c2.max()+1e-14)/255.0

    c0=np.require(c0,'float32','C')
    c1=np.require(c1,'float32','C')
    c2=np.require(c2,'float32','C')

    show=np.zeros((showsz,showsz,3),dtype='uint8')
    def render():
        rotmat=np.eye(3)
        if not freezerot:
            xangle=(mousey-0.5)*np.pi*1.2
        else:
            xangle=0
        rotmat=rotmat.dot(np.array([
            [1.0,0.0,0.0],
            [0.0,np.cos(xangle),-np.sin(xangle)],
            [0.0,np.sin(xangle),np.cos(xangle)],
            ]))
        if not freezerot:
            yangle=(mousex-0.5)*np.pi*1.2
        else:
            yangle=0
        rotmat=rotmat.dot(np.array([
            [np.cos(yangle),0.0,-np.sin(yangle)],
            [0.0,1.0,0.0],
            [np.sin(yangle),0.0,np.cos(yangle)],
            ]))
        rotmat*=zoom
        nxyz=xyz.dot(rotmat)+[showsz/2,showsz/2,0]

        ixyz=nxyz.astype('int32')
        show[:]=background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        if magnifyBlue>0:
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=0))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=0))
            show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],1,axis=1))
            if magnifyBlue>=2:
                show[:,:,0]=np.maximum(show[:,:,0],np.roll(show[:,:,0],-1,axis=1))
        if showrot:
            cv2.putText(show,'xangle %d'%(int(xangle/np.pi*180)),(30,showsz-30),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'yangle %d'%(int(yangle/np.pi*180)),(30,showsz-50),0,0.5,cv2.cv.CV_RGB(255,0,0))
            cv2.putText(show,'zoom %d%%'%(int(zoom*100)),(30,showsz-70),0,0.5,cv2.cv.CV_RGB(255,0,0))
    
    changed=True
    while True:
        if changed:
            render()
            changed=False
        cv2.imshow('show3d',show)
        if waittime==0:
            cmd=cv2.waitKey(10)%256
        else:
            cmd=cv2.waitKey(waittime)%256
        if cmd==32: # SPACE
            break
        elif cmd==27: # ESC
            sys.exit(0)

        if cmd==ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0=np.zeros((num_points,),dtype='float32')+255
                    c1=np.zeros((num_points,),dtype='float32')+255
                    c2=np.zeros((num_points,),dtype='float32')+255
                else:
                    c0=c_gt[:,0]
                    c1=c_gt[:,1]
                    c2=c_gt[:,2]
            else:
                if c_pred is None:
                    c0=np.zeros((num_points,),dtype='float32')+255
                    c1=np.zeros((num_points,),dtype='float32')+255
                    c2=np.zeros((num_points,),dtype='float32')+255
                else:
                    c0=c_pred[:,0]
                    c1=c_pred[:,1]
                    c2=c_pred[:,2]
            if normalizecolor:
                c0/=(c0.max()+1e-14)/255.0
                c1/=(c1.max()+1e-14)/255.0
                c2/=(c2.max()+1e-14)/255.0
            c0=np.require(c0,'float32','C')
            c1=np.require(c1,'float32','C')
            c2=np.require(c2,'float32','C')
            changed = True


        if cmd==ord('n'):
            zoom*=1.1
            changed=True
        elif cmd==ord('m'):
            zoom/=1.1
            changed=True
        elif cmd==ord('r'):
            zoom=1.0
            changed=True
        elif cmd==ord('s'):
            cv2.imwrite('show3d.png',show)
        elif cmd==ord('f'):
            freezerot = ~freezerot
        if waittime!=0:
            break
    return cmd


if __name__=='__main__':
    np.random.seed(100)
    showpoints(np.random.randn(2500,3))


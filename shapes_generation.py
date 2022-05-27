import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import io

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt


bernstein = lambda n, k, t: binom(n,k)* t**k * (1.-t)**(n-k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1; self.p2 = p2
        self.angle1 = angle1; self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2-self.p1)**2))
        self.r = r*d
        self.p = np.zeros((4,2))
        self.p[0,:] = self.p1[:]
        self.p[3,:] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self,r):
        self.p[1,:] = self.p1 + np.array([self.r*np.cos(self.angle1),
                                    self.r*np.sin(self.angle1)])
        self.p[2,:] = self.p2 + np.array([self.r*np.cos(self.angle2+np.pi),
                                    self.r*np.sin(self.angle2+np.pi)])
        self.curve = bezier(self.p,self.numpoints)


def get_curve(points, **kw):
    segments = []
    for i in range(len(points)-1):
        seg = Segment(points[i,:2], points[i+1,:2], points[i,2],points[i+1,2],**kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p-np.mean(p,axis=0)
    s = np.arctan2(d[:,0], d[:,1])
    return p[np.argsort(s),:]

def get_bezier_curve(a, rad=0.2, edgy=0):
    """ given an array of points *a*, create a curve through
    those points. 
    *rad* is a number between 0 and 1 to steer the distance of
          control points.
    *edgy* is a parameter which controls how "edgy" the curve is,
           edgy=0 is smoothest."""
    p = np.arctan(edgy)/np.pi+.5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0,:]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:,1],d[:,0])
    f = lambda ang : (ang>=0)*ang + (ang<0)*(ang+2*np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang,1)
    ang = p*ang1 + (1-p)*ang2 + (np.abs(ang2-ang1) > np.pi )*np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x,y = c.T
    return x,y, a


def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7/n
    a = np.random.rand(n,2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1)**2)
    if np.all(d >= mindst) or rec>=200:
        return a*scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec+1)

dict_colors = {'green': (1,128,1), 
               'black': (1, 1, 1), 
               'blue': (1,1,255), 
               'brown': (139,69,19), 
               'yellow': (255,255,1), 
               'deeppink': (255,20,147), 
               'orange': (255,165,1), 
               'purple': (138,43,226), 
               'white': (254,254,254),
               'pink': (255,192,203)
              }

def triangle_with_color(width, height, color):
    img_triangle = np.ones((width, height, 3), np.uint8) * 0
    mask_triangle = np.ones((width, height, 3), np.uint8) * 0

    pt1 = np.random.randint(50, height-100, size=2)
    pt1 = (pt1[0], pt1[1])

    low_line = np.random.randint(100, height - pt1[0], size=1)
    pt2 = ((pt1[0] + low_line)[0], pt1[1])

    if pt1[1] < height - pt1[1]:

        pt3_y = pt1[1] + np.random.randint(pt1[1], height - pt1[1], size=1)
    else:
        pt3_y = pt1[1] - np.random.randint(50, pt1[1], size=1)
    pt3_x = np.random.randint(min(pt1[0], pt1[0] - 50), max(pt2[0], pt2[0]+50), size=1)

    pt3 = (pt3_x[0], pt3_y[0])

    triangle = np.array( [pt1, pt2, pt3] )

    img_color = dict_colors[color]

    cv2.drawContours(img_triangle, [triangle], 0, img_color, -1)
#     cv2.drawContours(img_triangle, [triangle], 0, (255,255,255), -1)
    cv2.drawContours(mask_triangle, [triangle], 0, (255,255,255), -1)

    return img_triangle, mask_triangle


def rectangle_with_color(width, height, color):
    img_rectangle = np.ones((width, height, 3), np.uint8) * 0
    mask_rectangle = np.ones((width, height, 3), np.uint8) * 0

    low_left_point = np.random.randint(0, 200, size =2)
    max_size = max(min(low_left_point[0], low_left_point[1]), 
                   min(width - low_left_point[0], height - low_left_point[1]))

    width_rectangle = np.random.randint(50, max(150,max_size), size=1)
    height_rectangle = np.random.randint(50, max(150,max_size), size=1)

    upper_right_point = np.array([low_left_point[0] + width_rectangle, low_left_point[1] + height_rectangle])
    upper_right_point = upper_right_point.flatten()

    img_color = dict_colors[color]

    cv2.rectangle(img_rectangle,(low_left_point[0],low_left_point[1]),
                  (upper_right_point[0],upper_right_point[1]),img_color,-1)
    cv2.rectangle(mask_rectangle,(low_left_point[0],low_left_point[1]),
                  (upper_right_point[0],upper_right_point[1]),(255,255,255),-1)
    
    return img_rectangle, mask_rectangle


def circle_with_color(width, height, color):
    img_circle = np.ones((width, height, 3), np.uint8) * 0
    mask_circle = np.ones((width, height, 3), np.uint8) * 0

    center_x = np.random.randint(100, 200, size=1)[0]
    center_y = np.random.randint(100, height-100, size=1)[0]
    max_radius = min(50, min(min(center_x, center_y), min(width - center_x, height - center_y)))
    radius = np.random.randint(40, max_radius)

    img_color = dict_colors[color]
    cv2.circle(img_circle,(center_y,center_x), radius, img_color, -1)  
    cv2.circle(mask_circle,(center_y,center_x), radius, (255,255,255), -1)
    
    return img_circle, mask_circle


def hexagon_with_color(width, height, color):
    image = np.random.randint(0, 1, size=(width, height, 3)) 
    image = np.asarray(image, dtype="uint8")
    
    image_mask = np.random.randint(0, 1, size=(width, height, 3))
    image_mask = np.asarray(image_mask, dtype="uint8")
    
    x0 = np.random.randint(10, int(image.shape[1]*(1/3)), size=1)[0]
    diff_x = np.random.randint(int(image.shape[1]*(1/6)), int(image.shape[1]*(1/4)), size=1)[0]
    x1 = x0 + diff_x
    x2 = x1 + diff_x

    y0 = np.random.randint(5, int(image.shape[0]*(1/4)), size=1)[0]
    diff_y = np.random.randint(int(image.shape[0]*(1/6)), int(image.shape[0]*(1/5)), size=1)[0]
    y1 = y0 + diff_y
    y2 = y1 + diff_y
    y3 = y2 + diff_y

    xy_leftdown = [x0, y1]
    xy_leftup = [x0, y2]
    xy_up = [x1, y3]
    xy_rightup = [x2, y2]
    xy_rightdown = [x2, y1]
    xy_down = [x1, y0]

    pts = np.array([xy_leftdown, xy_leftup, xy_up, xy_rightup, xy_rightdown, xy_down], np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True

    img_color = dict_colors[color]
    img = cv2.fillPoly(image, [pts], img_color)  
    
    mask = cv2.fillPoly(image_mask, [pts], (255,255,255))
    
    return img, mask


def trapezoid_with_color(width, height, color):
    image = np.random.randint(0, 1, size=(width, height, 3)) 
    image = np.asarray(image, dtype="uint8")
    
    image_mask = np.random.randint(0, 1, size=(width, height, 3))
    image_mask = np.asarray(image_mask, dtype="uint8")
    
    x0 = np.random.randint(10, int(image.shape[1]*(1/3)), size=1)[0]
    diff_x = np.random.randint(int(image.shape[1]*(1/6)), int(image.shape[1]*(1/4)), size=1)[0]
    x1 = x0 + diff_x
    x3 = np.random.randint(int(image.shape[1]*(3/4)), image.shape[1]-20, size=1)[0]
    x2 = x3 - diff_x

    y0 = np.random.randint(5, int(image.shape[0]*(1/3)), size=1)[0]
    diff_y = np.random.randint(int(image.shape[0]*(1/5)), int(image.shape[0]*(3/5)), size=1)[0]
    y1 = y0 + diff_y

    pts = np.array([[x0, width-y0], [x1, width-y1], 
                    [x2, width-y1], [x3, width-y0]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    isClosed = True

    img_color = dict_colors[color]
    img = cv2.fillPoly(image, [pts], img_color)  
    
    mask = cv2.fillPoly(image_mask, [pts], (255,255,255))
    
    return img, mask


def star_with_color(width, height, color):
    image = np.random.randint(0, 1, size=(width, height, 3)) 
    image = np.asarray(image, dtype="uint8")
    
    image_mask = np.random.randint(0, 1, size=(width, height, 3)) 
    image_mask = np.asarray(image_mask, dtype="uint8")
    
    x0 = np.random.randint(10, int(image.shape[1]*(1/5)), size=1)[0]
    diff_x = np.random.randint(int(image.shape[1]*(1/7)), int(image.shape[1]*(1/5)), size=1)[0]
    x1 = x0 + int(1.4*diff_x)
    x4 = np.random.randint(int(image.shape[1]*(4/5)), image.shape[1]-20, size=1)[0]
    x3 = x4 - int(1.4*diff_x)
    x2 = (x1 + x3) // 2

    y0 = np.random.randint(5, int(image.shape[0]*(1/5)), size=1)[0]
    diff_y = np.random.randint(int(image.shape[0]*(1/9)), int(image.shape[0]*(1/6)), size=1)[0]
    y1 = y0 + diff_y
    y2 = y1 + diff_y
    y3 = y2 + int(1.5*diff_y)
    y4 = y3 + int(1.5*diff_y)

    pts = np.array([[x0, width-y0], [x1, width-y2], [x0, width-y3],
                   [x1, width-y3], [x2, width-y4], [x3, width-y3],
                   [x4, width-y3], [x3, width-y2], [x4, width-y0], 
                   [x2, width-y1]], np.int32)
    
    pts = pts.reshape((-1, 1, 2))
    isClosed = True

    img_color = dict_colors[color]
    img = cv2.fillPoly(image, [pts], img_color)  
    
    mask = cv2.fillPoly(image_mask, [pts], (255,255,255))
    
    return img, mask


def cross_with_color(width, height, color):
    image = np.random.randint(0, 1, size=(width, height, 3)) 
    image = np.asarray(image, dtype="uint8")
    
    image_mask = np.random.randint(0, 1, size=(width, height, 3)) 
    image_mask = np.asarray(image_mask, dtype="uint8")
    
    x0 = np.random.randint(10, int(image.shape[1]*(1/5)), size=1)[0]
    diff_x = np.random.randint(int(image.shape[1]*(1/7)), int(image.shape[1]*(1/4)), size=1)[0]
    x1 = x0 + int(1.5*diff_x)
    x3 = np.random.randint(int(image.shape[1]*(3/4)), image.shape[1]-int(image.shape[1]*(1/5)), size=1)[0]
    x2 = x3 - int(1.5*diff_x)

    y0 = np.random.randint(5, int(image.shape[0]*(1/5)), size=1)[0]
    diff_y = np.random.randint(int(image.shape[0]*(1/7)), int(image.shape[0]*(1/4)), size=1)[0]
    y1 = y0 + int(1.5*diff_y)
    y3 = np.random.randint(int(image.shape[0]*(3/4)), image.shape[0]-int(image.shape[0]*(1/5)), size=1)[0]
    y2 = y3 - int(1.5*diff_y)

    pts = np.array([[x0, width-y1], [x0, width-y2], [x1, width-y2],
                   [x1, width-y3], [x2, width-y3], [x2, width-y2],
                   [x3, width-y2], [x3, width-y1], [x2, width-y1], 
                   [x2, width-y0], [x1, width-y0], [x1, width-y1]], np.int32)
    
    pts = pts.reshape((-1, 1, 2))
    isClosed = True

    img_color = dict_colors[color]
    img = cv2.fillPoly(image, [pts], img_color)  
    
    mask = cv2.fillPoly(image_mask, [pts], (255,255,255))
    
    return img, mask


def get_img_from_fig(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# a = [5, 13]
# b = [1, 3]

def heart_with_color(width, height, color, dpi=1):
    a = np.random.randint(5, 13, size=1)[0]
    b = np.random.randint(1, 3, size=1)[0]
    
    theta = np.linspace(0, 2 * np.pi, 100)
    x = a * (np.sin(theta) ** 3 )
    y = a * np.cos(theta) - 1 * np.cos(b*theta) - 2 * np.cos((b+1)*theta) - np.cos((b+2)*theta)
    
    fig1 = plt.figure(figsize=(height, width))
    plt.style.use('dark_background')
    plt.fill(x, y, color)
    plt.axis('off')
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(height, width))
    plt.style.use('dark_background')
    plt.fill(x, y, 'white')
    plt.axis('off')
    plt.close(fig2)
    
    heart_img = get_img_from_fig(fig1, dpi)
    heart_mask = get_img_from_fig(fig2, dpi)
    return heart_img, heart_mask


def moon_with_color(width, height, color):
    img_color = dict_colors[color] 
   
    moon_img = np.ones((width, height, 3), np.uint8) * 0
    moon_mask = np.ones((width, height, 3), np.uint8) * 0
    x = np.random.randint(100, 200, size=1)[0]
    y = np.random.randint(100, 200, size=1)[0]
    r_1 = np.random.randint(50, 100, size=1)[0]
    r_2 = np.random.randint(40, 80, size=1)[0]
    cv2.circle(moon_img, (x, y), r_1, img_color, -1) 
    cv2.circle(moon_img, (x + r_2-10, y), r_2, (0, 0, 0), -1) 

    cv2.circle(moon_mask, (x, y), r_1, (255,255,255), -1) 
    cv2.circle(moon_mask, (x + r_2-10, y), r_2, (0, 0, 0), -1)  

    return moon_img, moon_mask 



def cloud_with_color(width, height, color):
    choise = np.random.randint(2, 4, size=1)[0]
    img_color = dict_colors[color] 
#     if choise == 0:
#         cloud_mask = cv2.imread('./support_imgs/cloud_0.jpg')
#         cloud_img = cv2.imread('./support_imgs/cloud_0.jpg')
#         for i in range(cloud_img.shape[0]):
#             for j in range(cloud_img.shape[1]):
#                 if 255 in cloud_img[i][j]:
#                     cloud_img[i][j] = list(img_color) 
#     elif choise == 1:
#         cloud_mask = cv2.imread('./support_imgs/cloud_1.jpg')
#         cloud_img = cv2.imread('./support_imgs/cloud_1.jpg')
#         for i in range(cloud_img.shape[0]):
#             for j in range(cloud_img.shape[1]):
#                 if 255 in cloud_img[i][j]:
#                     cloud_img[i][j] = list(img_color) 
#     else:   
    cloud_img = np.ones((width, height, 3), np.uint8) * 0
    cloud_mask = np.ones((width, height, 3), np.uint8) * 0
    x = np.random.randint(100, 200, size=1)[0]
    y = np.random.randint(100, 200, size=1)[0]
    if choise == 2:
        cv2.circle(cloud_img, (x, y), 60, img_color, -1) 
        cv2.circle(cloud_img, (x + 50, y), 40, img_color, -1) 
        cv2.circle(cloud_img, (x - 50, y), 40, img_color, -1)

        cv2.circle(cloud_mask, (x, y), 60, (255,255,255), -1) 
        cv2.circle(cloud_mask, (x + 50, y), 40, (255,255,255), -1) 
        cv2.circle(cloud_mask, (x - 50, y), 40, (255,255,255), -1)
    elif choise == 3:    
        cv2.circle(cloud_img, (x, y), 60, img_color, -1) 
        cv2.circle(cloud_img, (x + 40, y), 50, img_color, -1) 
        cv2.circle(cloud_img, (x - 40, y), 50, img_color, -1)
        cv2.circle(cloud_img, (x + 80, y), 25, img_color, -1) 
        cv2.circle(cloud_img, (x - 80, y), 25, img_color, -1)

        cv2.circle(cloud_mask, (x, y), 60, (255,255,255), -1) 
        cv2.circle(cloud_mask, (x + 40, y), 50, (255,255,255), -1) 
        cv2.circle(cloud_mask, (x - 40, y), 50, (255,255,255), -1)
        cv2.circle(cloud_mask, (x + 80, y), 25, (255,255,255), -1) 
        cv2.circle(cloud_mask, (x - 80, y), 25, (255,255,255), -1)

    return cloud_img, cloud_mask    

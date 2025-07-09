#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 00:05:22 2018

@author: geolog
"""
import sys
import time  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage as nd
import skimage.morphology as skm #scikit-image
import scipy.ndimage.measurements as scim #scikit-image
import scipy.ndimage.morphology as scimo
import skimage.measure as skms #label, regionprops
import skimage.feature as skmf #canny filter will be taken from here
from skimage.draw import line as drawline
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import math

import copy
#for hough filters
from skimage.transform import hough_line, hough_line_peaks,probabilistic_hough_line
import os
#for distances
from scipy.spatial.distance import pdist,squareform

from scipy.interpolate import griddata
import imageio as io

try:
    import gdal,ogr
except ModuleNotFoundError:
    from osgeo import gdal,ogr

#timedate to check 
from datetime import date

from PyQt5.QtWidgets import QWidget,QListWidgetItem,QAbstractItemView,QProgressBar,qApp,QDesktopWidget,QLabel
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt


exp_date=date(2019,12,5) #после этой даты приложение не запустится

def detectFlowNetwork(srtm,accuracy):
    min_area_streams=50 #analysis parameters
    
    if accuracy=='min':
        flood_steps=9    
    if accuracy=='mean':
        flood_steps=10
    if accuracy=='max':
        flood_steps=11
    #imshape
      
    r,c=np.shape(srtm)
    
    #gradual flood 
    
    flows=np.zeros([r,c],dtype=float)
    mins=np.min(srtm)
    maxs=np.max(srtm)
    grad_flood_step=int((maxs-mins)/flood_steps)
    
    #ВНИМАНИЕ!!!! ДЛЯ ТОЧНОГО ДЕТЕКТИРОВАНИЯ НИКАКОГО РАЗМЫТИЯ
    #for i in range(mins,maxs,100): #
    for i in range(mins,maxs,grad_flood_step): #flood relief and skeletize it
        
        thinned=skm.thin(np.int16(srtm<=i)) #надо использовать истончение, а не скелетизацию
        
        if(i>mins):
            thinned[srtm<(i-100)]=0
        
        flows=flows+thinned
    flows=np.int16(flows>0)  
    
    #remove orphan streams (area less than 10)
    flows_label=skms.label(np.uint(flows), background=None, return_num=False,
                           connectivity=2)
    for i in range(1,np.max(flows_label),1):
        if np.sum(flows_label==i)<=min_area_streams:
            flows[flows_label==i]=0
    
    
    #close and thin to remove small holes)
    
    strel=skm.disk(1)
    flows=skm.closing(flows,strel)  
    flows=np.int16(skm.skeletonize_3d(flows))    #need to convert into int8, cause closing returns BOOL     
    
    #remove sea level
    flows[srtm<=0]=0
    
    return flows

def detectFlowOrders(srtm,flows):    
    r,c=np.shape(flows)
    #end of orphan flows removal
    
    #from this points flows are extracted, so we need to subordinate them consequently
    #самый главный алгоритм 
    order=0
    is_completed=False
    flow_orders=np.zeros([r,c],dtype=int) #array to store flow generations flat
    prev_remaining_pnt=0
        
    while is_completed==False:
    #for x in range(1,5):
        order+=1 #увеличиваем порядок
        #finding flows beginning points
        flow_start_points=np.zeros([r,c],dtype=int) #array of the sources
        flow_inter_points=np.zeros([r,c],dtype=int) #array of the junctions
           
        for i in range(1,r,1):
            for ii in range(1,c,1):
               #cp=flows[(i-1):(i+2),(ii-1):(ii+2)]; #current pattern 
               if flows[i,ii]==1 and np.sum(flows[(i-1):(i+2),(ii-1):(ii+2)])==2:  #if point EXISTS and has only one neighbour
                         
                   flow_start_points[i,ii]=1
                   
               if flows[i,ii]==1 and np.sum(flows[(i-1):(i+2),(ii-1):(ii+2)])>=4:  #if point has only TWO neighbours    
                   flow_inter_points[i,ii]=1
        if order==1:
            flow_start_points_ini=flow_start_points;
            flow_inter_points_ini=flow_inter_points; #для закрытия дыр в конце работы приложения
        #label для промежуточных точек
        #flow_inter_points_label=skms.label(np.uint(flow_inter_points),neighbors=None, background=None, return_num=False,
        #                   connectivity=2)
        
        #нарощенные промежуточные точки 
        strel=skm.disk(1)
        flow_inter_points_dil=skm.dilation(flow_inter_points,strel)
        flow_inter_points_dil_label=skms.label(np.uint(flow_inter_points_dil),neighbors=None, background=None, return_num=False,
                               connectivity=2)
     
        #трассировка точек от начала водотоков 
        flows_broken=flows-flow_inter_points
        flows_broken_label=skms.label(np.uint(flows_broken),neighbors=None, background=None, return_num=False,
                           connectivity=2)
    
        #flow_cur_order=np.zeros([r,c],dtype=int)
    
        #добавление в массив водотоков первого порядка, labels которых содержит точки начала  
        for i in range(1,np.max(flows_broken_label),1):
            if np.sum(flow_start_points[flows_broken_label==i])>0:
                flow_orders[flows_broken_label==i]=order
        
        if prev_remaining_pnt==np.sum(flow_orders>0):
            print('Iteration stopped due to no start points were left')
            is_completed=True #если больше нет точек начала 
            break;
        
        #убираем добавленные водотоки        
        #reassign label values to strems
        flows_broken[flow_orders==order]=0
        flows_broken_label=skms.label(np.uint(flows_broken),neighbors=None, \
                                  background=None, return_num=False,connectivity=2)
           
        #remaining flows
        flows[flow_orders==order]=0
        flows=skm.skeletonize_3d(flows) #скелет чтобы не было раздвоенных окончаний
        prev_remaining_pnt=np.sum(flow_orders>0)
    #
    #все что нераспределено - наивысший порядок
    if np.sum(np.uint8(flow_orders==order))==0:
        flow_orders[flows==1]=order #undistributed flows to last incremented order
    else:
        flow_orders[flows==1]=order-1
    #label для промежуточных точек
    #нарощенные промежуточные точки 
    strel=skm.disk(2)               #!!!!ниже flow_inter_points_ini
    flow_inter_points_dil=skm.dilation(flow_inter_points_ini,strel)
    flow_inter_points_dil_label=skms.label(np.uint(flow_inter_points_dil),neighbors=None, background=None, return_num=False,
                           connectivity=2)
    
    #закрываем "дырки" соединителями, которые касаются водотока, присваивая наибольший порядок
    for i in range(1,np.max(flow_inter_points_dil_label),1):
        if np.sum(flow_orders[flow_inter_points_dil_label==i])>0:
            tmp_mask=np.zeros([r,c],dtype=int)
            tmp_mask=tmp_mask+skm.erosion(flow_inter_points_dil_label==i,skm.disk(2))
            #tmp_mask[flow_inter_points_dil_label==i]=1
            max_order_connected=np.max(flow_orders[flow_inter_points_dil_label==i])
            flow_orders[tmp_mask==1]=max_order_connected
    
    
    #prepare start and inter points output
    points_out=np.zeros([r,c],dtype=int)
    points_out[flow_inter_points_ini==1]=10;
    points_out[flow_start_points_ini==1]=20;
    
    #remove the sea
    flow_orders[srtm==0]=0
    points_out[srtm==0]=0
    return flow_orders,points_out

def gray2binary(img,method,sigma=10):
    sigmaval = sigma / 10
    #method can be Canny, flow etc.
    if method=='flow':
        print('binary flow detection was called')
        imgBW=detectFlowNetwork(img,'max') 
    elif method=='canny':
        print('binary Canny was called')
        img=np.uint8(np.float64((img-np.min(img))/(np.max(img)-np.min(img)))*255)
        imgBW = np.uint16(skmf.canny(img,sigma=sigmaval))
    else:
        print('no binarization method was recognized. Use Canny')
        img = np.uint8(np.float64((img - np.min(img)) / (np.max(img) - np.min(img))) * 255)
        imgBW = np.uint16(skmf.canny(img))
    return imgBW

def adjusted_canny(img,sigma=1,low_threshold=1, high_threshold=25):
    edges = skmf.canny(img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return edges

def detectPLineHough(imgBW,amount):   #P for probabilistic
    #C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
    #       Hough transform for line detection", in IEEE Computer Society
    #       Conference on Computer Vision and Pattern Recognition, 1999.
    #explaination of Hough params https://scikit-image.org/docs/stable/auto_examples/edges/plot_line_hough_transform.html

    print('hough detection was called')
    print(amount)
    if amount=='small':
        tres=100
        leng=10
        lg=2    
    if amount=='medium':
        tres=10
        leng=5
        lg=3 
    if amount=='many':
        tres=5
        leng=3
        lg=1 
    lines = probabilistic_hough_line(imgBW, threshold=tres, line_length=leng,\
                                 line_gap=lg) #параметры подбираются исходя из порога    
    return lines

def detectPLineHough2(imgBW,tres=None, leng = None, lg = None):   #P for probabilistic
    #C. Galamhos, J. Matas and J. Kittler, "Progressive probabilistic
    #       Hough transform for line detection", in IEEE Computer Society
    #       Conference on Computer Vision and Pattern Recognition, 1999.
    print('hough detection was called')
    if tres == None or leng == None or lg == None:
        print('not enough parameters')
        return None
    lines = probabilistic_hough_line(imgBW, threshold=tres, line_length=leng,\
                                 line_gap=lg) #параметры подбираются исходя из порога
    return lines

def lineCentroids(lines):
    centroids=[]
    for line in lines:
        x=line[0][0]+(line[1][0]-line[0][0])/2
        y=line[0][1]+(line[1][1]-line[0][1])/2
        centroids.append([x,y])
    return centroids

def lineLength(lines):
    lengths=[]
    for line in lines:
        length=((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2)**0.5
        lengths.append(length)
    return lengths

def lineAngle(lines):
    angle=[]
    for line in lines:
        dx=line[1][0]-line[0][0]
        dy=line[1][1]-line[0][1]
        if dx!=0:
            angle.append(np.arctan(dy/dx))
        else:
            angle.append(1.57)
        if angle[-1]<0:
            angle[-1]=angle[-1]+np.pi
    return angle

def lineKB(lines,deg):
    k=[]; b=[]
    for line in lines:
        P=np.polyfit([line[0][0],line[1][0]],[line[0][1],line[1][1]],deg)
        k.append(P[0]); b.append(P[1])
    return k,b

def distMat(pnts):
    #dist=pdist(pnts,'euclidean', p=2)
    dist = pdist(pnts, 'euclidean')
    mat = squareform(dist)
    return mat
    
def uniteLines(lines):
    k,b=lineKB(lines,deg=2)
    lines4unification=[]
    lines4unificationList=[]
    addedLines=[] #remember if line was added
    centroids=lineCentroids(lines)
    angle=lineAngle(lines)
    mat=distMat(centroids) #distance
    #distance k and k and other proximity matrixes
    mat_b=distMat(np.transpose([b,b])) 
    mat_k=distMat(np.transpose([k,k])) 
    mat_angle=distMat(np.transpose([angle,angle])) 
    for i in range(0,len(lines)):
        if len(lines4unification)!=0: #если массиb объединенных линий не пустой - удаляем его содержимое
            lines4unificationList.append(lines4unification)
            lines4unification=[]
        for ii in range(0,len(lines)):
            if i!=ii:
                    tmp_arr=np.array(lines4unification)
                    if len(tmp_arr)==0:
                        x=np.array([lines[i][0][0],lines[i][1][0],lines[ii][0][0],lines[ii][1][0]])
                        y=np.array([lines[i][0][1],lines[i][1][1],lines[ii][0][1],lines[ii][1][1]])
                    else:
                        x=np.append(tmp_arr[:,1,0],[lines[ii][0][0],lines[ii][1][0]])
                        y=np.append(tmp_arr[:,1,1],[lines[ii][0][1],lines[ii][1][1]])
                    
                    r=np.corrcoef(x,y)
                    #if r[0,1]>0.8 and mat[i,ii]<50 and (lines[ii] not in addedLines):
                    #if mat_angle[i,ii]<0.2 and r[0,1]>0.98 and mat[i,ii]<100 and mat_bb[i,ii]<30 and (lines[ii] not in addedLines): #
                    #if r[0,1]>0.9 and mat_angle[i,ii]<0.3 and mat[i,ii]<=70 and (lines[ii] not in addedLines):
                    if r[0,1]>0.95 and mat_b[i,ii]<0.8 and mat_k[i,ii]<=0.9 and mat[i,ii]<100 and mat_angle[i,ii]<0.3 and (lines[ii] not in addedLines): 
                    #if r[0,1]>0.999 and (lines[ii] not in addedLines):
                        lines4unification.append(lines[ii])
                        addedLines.append(lines[ii])            
    #output result for the line
    faults=[]
    for listL in lines4unificationList:
        x=[];y=[]
        if len(listL)>=3:
            for line in listL:
                p0, p1 = line
                x.append(p0[0]);x.append(p1[0])
                y.append(p0[1]);y.append(p1[1])
                
            P=np.polyfit(x,y,1)
            xnew=np.float64(range(np.min(x),np.max(x)))
            ynew=P[0]*np.float64(xnew)+P[1]
            faults.append([[xnew[0],ynew[0]],[xnew[-1],ynew[-1]]])            
            
    return faults                
def uniteLines2(lines):
    k,b=lineKB(lines,deg=2)
    kmat=distMat(np.transpose([k,k]))
    bmat=distMat(np.transpose([b,b]))
    collinear=np.zeros([len(k),len(k)],dtype=int)
    
    sigma_kmat,sigma_bmat=np.std(kmat),np.std(bmat)
    mean_kmat,mean_bmat=np.mean(kmat),np.mean(bmat)
    
    collinear[(kmat<(mean_kmat-0.2*sigma_kmat))==(bmat<(mean_bmat-0.2*sigma_bmat))]=1
    #collinear[(kmat==0)==(bmat==0)]=1
    
    lines2uniteList=[]
    lines2unite=[]
    addedLines=[]
    for i in range(0,len(k)): #по рядам
        if len(lines2unite)!=0:
            lines2uniteList.append(lines2unite)
            lines2unite=[]
        for ii in range(0,len(k)):
            if i!=ii and collinear[i,ii]==1 and (lines[ii] not in addedLines):
                lines2unite.append(lines[ii])
                addedLines.append(lines[ii])
    
    #print(lines2uniteList)
    
    faults=[]
    for listL in lines2uniteList:
        x=[];y=[]
        if len(listL)>0:
            for line in listL:
                p0, p1 = line
                x.append(p0[0]);x.append(p1[0])
                y.append(p0[1]);y.append(p1[1])
                P=np.polyfit(x,y,1)
                xnew=range(np.min(x),np.max(x))
                ynew=P[0]*np.float64(xnew)+P[1]
                if len(xnew)>0:
                    faults.append([[xnew[0],ynew[0]],[xnew[-1],ynew[-1]]])
    return faults


def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

    @type geom:     C{tuple/list}
    @param geom:    List of [[x,y],...[x,y]] coordinates
    @type src_srs:  C{osr.SpatialReference}
    @param src_srs: OSR SpatialReference object
    @type tgt_srs:  C{osr.SpatialReference}
    @param tgt_srs: OSR SpatialReference object
    @rtype:         C{tuple/list}
    @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = ogr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords


def GetExtent(gt, cols, rows):
    """
    srtm_gdal_object.GetGeoTransform()

    (329274.50572846865, - left X
     67.87931651487438,  - dX
     0.0,
     4987329.504699751,  - верх Y
     0.0,
     -92.95187590930819) - dY
    """
    # [[влx,влy],[нлx,нлy],[нпx, нпy],[впx, впy]]
    ext = [[gt[0], gt[3]], [gt[0], (gt[3] + gt[5] * rows)], [(gt[0] + gt[1] * cols), (gt[3] + gt[5] * rows)],
           [(gt[0] + gt[1] * cols), gt[3]]];
    return ext

    
def GetExtent2(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
        yarr.reverse()
    return ext

#TODO saveLinesShpFile
def saveLinesShpFile(lines,filename,gdal_object):
    qApp.processEvents()
    #https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    print('dummy function for exporting SHP file data')
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    
    ###
    gt=gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext=GetExtent(gt,cols,rows) #[[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    #resolution in meters
    dpx=(ext[3][0]-ext[0][0])/cols
    dpy=(ext[0][1]-ext[2][1])/rows

    pbar_window = ProgressBar()
    id_count = 0
    for line in lines:
        pbar_window.doProgress(id_count, len(lines))
        lineout = ogr.Geometry(ogr.wkbLineString)
        lineout.AddPoint(ext[0][0]+dpx*line[0][0], ext[0][1]-dpy*line[0][1])
        lineout.AddPoint(ext[0][0]+dpx*line[1][0], ext[0][1]-dpy*line[1][1])
        multiline.AddGeometry(lineout)
    
        #multiline=multiline.ExportToWkt()
        id_count += 1

    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('', None, ogr.wkbLineString)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    #geom = ogr.CreateGeometryFromWkt(multiline)

    feat.SetGeometry(multiline)

    layer.CreateFeature(feat)
    feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None

#TODO saveLinesShpFile2
def saveLinesShpFile2(lines, filename, gdal_object=None,ext=None,dpxy=None):
    qApp.processEvents()
    # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html
    print('dummy function for exporting SHP file data')
    multiline = ogr.Geometry(ogr.wkbMultiLineString)

    if gdal_object:
        gt = gdal_object.GetGeoTransform()
        cols = gdal_object.RasterXSize
        rows = gdal_object.RasterYSize
        ext = GetExtent(gt, cols, rows)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
        # resolution in meters
        dpx = (ext[3][0] - ext[0][0]) / cols
        dpy = (ext[0][1] - ext[2][1]) / rows
        #dpx = np.abs(gt[1])
        #dpy = np.abs(gt[5])
    if ext and dpxy:
        dpx,dpy = dpxy[0],dpxy[1]


    driver = ogr.GetDriverByName('Esri Shapefile')

    if os.path.exists(filename):
        driver.DeleteDataSource(filename)
    
    ds = driver.CreateDataSource(filename)
    layer = ds.CreateLayer('', None, ogr.wkbLineString)
    # create a field
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    lenField = ogr.FieldDefn('length', ogr.OFTInteger)
    azimField = ogr.FieldDefn('azimuth', ogr.OFTInteger)
    layer.CreateField(idField)
    layer.CreateField(lenField)
    layer.CreateField(azimField)
    defn = layer.GetLayerDefn()
    id_count = 0
    pbar_window = ProgressBar()
    for line in lines:
        pbar_window.doProgress(id_count, len(lines))
        multiline = ogr.Geometry(ogr.wkbMultiLineString)
        lineout = ogr.Geometry(ogr.wkbLineString)
        lineout.AddPoint(ext[0][0] + dpx * line[0][0], ext[0][1] - dpy * line[0][1])
        lineout.AddPoint(ext[0][0] + dpx * line[1][0], ext[0][1] - dpy * line[1][1])
        multiline.AddGeometry(lineout)

        length,azimuth=get_line_azimuth_length2(line)

        #multiline = multiline.ExportToWkt()
        
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', id_count)
        feat.SetField('length', length)
        feat.SetField('azimuth', azimuth)

        # Make a geometry, from Shapely object
        #geom = ogr.CreateGeometryFromWkt(multiline)

        feat.SetGeometry(multiline)

        layer.CreateFeature(feat)
        id_count += 1
    pbar_window.close()
    feat = geom = None  # destroy these

def savePointsShpFile2(points, filename, gdal_object):  # nofield
    gt = gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext = GetExtent(gt, cols, rows)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    # resolution in meters
    dpx = (ext[3][0] - ext[0][0]) / cols
    dpy = (ext[0][1] - ext[2][1]) / rows

    # Create the output shapefile
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(filename):
        shpDriver.DeleteDataSource(filename)

    outDataSource = shpDriver.CreateDataSource(filename)
    outLayer = outDataSource.CreateLayer(filename, geom_type=ogr.wkbPoint)
    # create a field
    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    AreaField = ogr.FieldDefn('Area', ogr.OFTInteger)
    outLayer.CreateField(idField);
    outLayer.CreateField(AreaField);

    # create point geometry
    for i in range(0, len(points[0])):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(ext[0][0] + dpx * points[0][i], ext[0][1] - dpy * points[1][i])

        # Create the feature and set values
        featureDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(point)
        outFeature.SetField('id', i)
        outFeature.SetField('Area', int(points[2][i]))
        outLayer.CreateFeature(outFeature)
        outFeature = None
    outDataSource = None

def generateDensityMap(self,lines,rows,cols,win_size=5):
    print('generate density map was pressed')
    #app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))  # set cursor

    x = [];
    y = [];
    z = []
    centroids = lineCentroids(lines)
    lengths = lineLength(lines)
    for i in range(0, int(rows / win_size) + 1):
        for ii in range(0, int(cols / win_size) + 1):
            c_x, c_y = ii * win_size + win_size / 2, i * win_size + win_size / 2
            x.append(c_x)
            y.append(c_y)
            l = []  # length of lines inside the window
            # search for the centroids inside the winwod
            for iii in range(0, len(centroids)):
                if centroids[iii][0] < (c_x + win_size / 2) and \
                        centroids[iii][0] > (c_x - win_size / 2) and \
                        centroids[iii][1] > (c_y - win_size / 2) and \
                        centroids[iii][1] < (c_y + win_size / 2):
                    l.append(lengths[iii])
            if len(l) != 0:
                z.append(np.mean(l))
            else:
                z.append(0)
    Y, X = np.mgrid[0:self.rows + 1, 0:self.cols + 1]
    densityMap = griddata((x, y), z, (X, Y), method='cubic')
    return densityMap
    #app.restoreOverrideCursor()

def generateDensityMap2(self,cols,rows,centroids,win_size = 5):
    print('generate density map was pressed')
    #app.setOverrideCursor(QCursor(QtCore.Qt.WaitCursor))  # set cursor

    x = [];
    y = [];
    z = []
    centroids = lineCentroids(lines)
    lengths = lineLength(lines)
    for i in range(0, int(rows / win_size) + 1):
        for ii in range(0, int(cols / win_size) + 1):
            c_x, c_y = ii * win_size, i * win_size
            x.append(c_x)
            y.append(c_y)
            l = []  # length of lines inside the window
            # search for the centroids inside the winwod
            for iii in range(0, len(centroids)):
                if (c_x + win_size) > centroids[iii][0] > (c_x - win_size) and \
                        (c_y - win_size) < centroids[iii][1] < (c_y + win_size):
                    l.append(self.lengths[iii])
            if len(l) != 0:
                z.append(np.mean(l))
            else:
                z.append(0)
    Y, X = np.mgrid[0:self.rows, 0:self.cols]
    densityMap = griddata((x, y), z, (X, Y), method='cubic')

    #app.restoreOverrideCursor()
    return densityMap

#TODO rasterize shape file for detecting faults
def rasterize_shp(lines,gdal_object=None,rasterData=None,wh = None):
    # create binary of lines
    if rasterData!=None:
        h, w = np.shape(rasterData)
    elif gdal_object!=None:
        w = gdal_object.RasterXSize
        h = gdal_object.RasterYSize
    
    elif wh != None:
        h, w = wh[1],wh[0]
    else:
        print('no resolution data provided')
        return None
    gt = gdal_object.GetGeoTransform() # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    ext = GetExtent(gt, w, h)
    dpx = (ext[3][0] - ext[0][0]) / w
    dpy = (ext[0][1] - ext[2][1]) / h
    canvas_image = np.zeros([h, w])
    new_lines = copy.deepcopy(lines)
    pbar_window = ProgressBar()
    id_count = 0
    for l in new_lines:
        pbar_window.doProgress(id_count, len(new_lines))
        for pnt_n in range(len(l)-1):
            try:
                p0, p1 = l[pnt_n],l[pnt_n+1]
                # rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
                p0[0] = int((p0[0] - ext[0][0]) / dpx)
                p0[1] = int((p0[1] - ext[0][1]) / dpy)
                p1[0] = int((p1[0] - ext[0][0]) / dpx)
                p1[1] = int((p1[1] - ext[0][1]) / dpy)

                rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
                canvas_image[rr, cc] = 1
            except:
                print('wrong index or segment!')
        # p0, p1 = copy.deepcopy(l)
        # #rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
        # p0[0] = int((p0[0] - ext[0][0]) / dpx)
        # p0[1] = int((p0[1] - ext[0][1]) / dpy)
        # p1[0] = int((p1[0] - ext[0][0]) / dpx)
        # p1[1] = int((p1[1] - ext[0][1]) / dpy)
        #
        # #rr, cc = drawline(int((p0[1]-ext[2][1])/w), int((p0[0]-ext[0][0])/h), int((p1[1]-ext[2][1])/w), int((p1[0]-ext[0][0])/h))  # coordinates should be placed in that order
        # rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
        # try:
        #     canvas_image[rr, cc] = 1
        # except:
        #     print('wrong index!')

        id_count +=1
    return np.flipud(canvas_image)


def rasterize_shp2(lines, extent=None, dpxy=None):
    # create binary of lines
    print('create binary of lines')
    if extent is not None or dpxy is not None:
        dpx, dpy = dpxy[0], dpxy[1]
    else:
        print('no resolution data provided')
        return None
    w = int((extent[1] - extent[0]) / dpx)
    h = int((extent[3] - extent[2]) / dpy)
    canvas_image = np.zeros([h, w])
    new_lines = copy.deepcopy(lines)
    pbar_window = ProgressBar()
    id_count = 0
    for l in new_lines:
        pbar_window.doProgress(id_count, len(new_lines))
        p0, p1 = copy.deepcopy(l)
        # rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
        p0[0] = int((p0[0] - extent[0]) / dpx)
        p0[1] = int((p0[1] - extent[2]) / dpy)
        p1[0] = int((p1[0] - extent[0]) / dpx)
        p1[1] = int((p1[1] - extent[2]) / dpy)

        rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
        try:
            canvas_image[rr, cc] = 1
        except:
            print('wrong index!')
        id_count += 1
    return np.flipud(canvas_image)

def rasterize_shp3(lines, extent=None, dpxy=None):
    # create binary of lines
    print('create binary of lines')
    if extent is not None or dpxy is not None:
        dpx, dpy = dpxy[0], dpxy[1]
    else:
        print('no resolution data provided')
        return None
    w = int((extent[1] - extent[0]) / dpx)
    h = int((extent[3] - extent[2]) / dpy)
    canvas_image = np.zeros([h, w])
    new_lines = copy.deepcopy(lines)
    pbar_window = ProgressBar()
    id_count = 0
    for l in new_lines:
        pbar_window.doProgress(id_count, len(new_lines))
        for pnt_n in range(len(l)-1):
            try:
                p0, p1 = l[pnt_n],l[pnt_n+1]
                # rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
                p0[0] = int((p0[0] - extent[0]) / dpx)
                p0[1] = int((p0[1] - extent[2]) / dpy)
                p1[0] = int((p1[0] - extent[0]) / dpx)
                p1[1] = int((p1[1] - extent[2]) / dpy)

                rr, cc = drawline(p0[1], p0[0], p1[1], p1[0])  # coordinates should be placed in that order
                canvas_image[rr, cc] = 1
            except:
                print('wrong index or segment!')
        id_count += 1
    return np.flipud(canvas_image)

def saveGeoTiff(raster,filename,gdal_object,ColMinInd=0,RowMinInd=0): #ColMinInd,RowMinInd - start row/col for cropped images
    meas=np.shape(raster)
    rows=meas[0]; cols=meas[1];
    if(len(meas)==3):
        zs=meas[2];
    else:
        zs=1;
    print("Saving "+filename)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename, cols, rows, zs, gdal.GDT_Float64)
    (start_x,resx,zerox,start_y,zeroy,resy)=gdal_object.GetGeoTransform()
    outdata.SetGeoTransform((start_x+(resx*ColMinInd),resx,zerox,start_y+(resy*RowMinInd),zeroy,resy));
    #outdata.SetGeoTransform(gdal_object.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(gdal_object.GetProjection())##sets same projection as input
    #write bands
    if zs>1:
        for b in range(0,zs):
            outdata.GetRasterBand(b+1).WriteArray(raster[:,:,b])
            outdata.GetRasterBand(b+1).SetNoDataValue(10000) ##if you want these values transparent
    else:
        outdata.GetRasterBand(1).WriteArray(raster) #write single value raster
    outdata.FlushCache() ##saves

def saveGeoTiffNodata(raster,filename,gdal_object,ColMinInd,RowMinInd,BitMode): #ColMinInd,RowMinInd - start row/col for cropped images
    if BitMode=="float64":
        bitres=gdal.GDT_Float64;
    elif BitMode=="int16":
       bitres=gdal.GDT_Int16;
    else:
       bitres=gdal.GDT_Int8;
    meas=np.shape(raster)
    rows=meas[0]; cols=meas[1];
    if(len(meas)==3):
        zs=meas[2];
    else:
        zs=1;
    print("Saving "+filename)
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename, cols, rows, zs, bitres)
    (start_x,resx,zerox,start_y,zeroy,resy)=gdal_object.GetGeoTransform()
    outdata.SetGeoTransform((start_x+(resx*ColMinInd),resx,zerox,start_y+(resy*RowMinInd),zeroy,resy));
    #outdata.SetGeoTransform(gdal_object.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(gdal_object.GetProjection())##sets same projection as input
    #write bands
    if zs>1:
        for b in range(0,zs):
            outdata.GetRasterBand(b+1).WriteArray(raster[:,:,b])
            outdata.GetRasterBand(b+1).SetNoDataValue(-32768) ##if you want these values transparent
    else:
        outdata.GetRasterBand(1).WriteArray(raster) #write single value raster
        outdata.GetRasterBand(1).SetNoDataValue(-32768)
    outdata.FlushCache() ##saves 


class ProgressBar(QWidget):

    def __init__(self):
        super().__init__()

        # creating progress bar
        self.pbar = QProgressBar(self)

        # create label
        self.label1 = QLabel('Processing...', self)
        self.label1.resize(140,10)
        self.label1.move(30, 25)

        # setting its geometry
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar_val=0 #initial value
        # creating push button
        # self.btn = QPushButton('Start', self)

        # changing its position
        # self.btn.move(40, 80)

        # adding action to push button
        # self.btn.clicked.connect(self.doAction)

        # setting window geometry
        self.setGeometry(300, 300, 280, 80)

        # setting window action
        self.setWindowTitle("Line vectorization")
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # set in the center of screen
        sizeObject = QDesktopWidget().screenGeometry(-1)
        # print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))
        self.move(int(sizeObject.width() / 2) - 140, int(sizeObject.height() / 2) - 40)

        # self.pbar.hide()
        # self.pbar.show()
        print('this is progress bar window!')
        # showing all the widgets
        self.show()
        #self.doAction()

    # when button is pressed this method is being called
    def doAction(self):
        # setting for loop to set value of progress bar
        for i in range(101):
            qApp.processEvents()  # обработка событий
            # slowing down the loop
            time.sleep(0.05)
            # setting value to progress bar
            self.pbar.setValue(i)
            # print(self.pbar.value())
        self.close()

    def doProgress(self, cur_val, max_val):
        #qApp.processEvents()  # обработка событий
        time.sleep(0.01)
        pbar_val = int((cur_val / max_val) * 100)
        self.pbar.setValue(pbar_val)
        # set value for label
        self.label1.setText(f'Processing {pbar_val} %')
        qApp.processEvents()

#functions for computetion of Minkowski dimension
def input_to_8gray(img):
    return img_as_ubyte(rgb2gray(img))

# функция считает количество ячеек, которые оказались содержащими фрагмент исследуемого объекта.
# возвращает два списка - пара значений "размер (длина стороны) ячейки" - "количество ячеек содержащих объект"

def box_count_fn(img):
    box_size_insrease = -10
    box_size_list = []
    box_count_list = []

    # if image is rgb or has more channels, convert it to gray
    if len(np.shape(img)) == 3:
        img = input_to_8gray(img)

    h, w = np.shape(img)
    box_size = min(h, w)  # начальный размер клетки - половина стороны на 2

    # binarize image
    grad_array = np.zeros(np.shape(img))
    ind = (img > 0)
    grad_array[ind] = 1  # ненулевые клетки делаем равны 1
    img = copy.copy(grad_array)

    while box_size > 0:
        cell_cover_sum = 0  # количество клеток, покрывших узор
        # проходим по картинке с заданным шагом методом скользящего окна
        for r in range(0, h, box_size):
            for c in range(0, w, box_size):
                sub_img = img[r:r + box_size, c:c + box_size]
                if np.sum(sub_img) > 0:
                    cell_cover_sum += 1

        box_size_list.append(box_size)
        box_count_list.append(cell_cover_sum)
        # изменяем размер клетки
        box_size += box_size_insrease

    return np.array(box_size_list[::-1]), np.array(box_count_list[::-1])


# функция возвращает log10(n) и log10(N) для ряда значений размера ячеек и его количества
def box_size_num_log(box_size_list, box_count_list):
    # return np.log(1/box_size_list),np.log(box_count_list)
    box_size_list_log, box_count_list_log = [],[]
    for bsl,bcl in zip(box_size_list,box_count_list):
        if bcl != 0:
            box_size_list_log.append(np.log10(1 / bsl))
            box_count_list_log.append(np.log10(bcl))
        else:
            pass;

    return box_size_list_log, box_count_list_log



#TODO get resolution in meters and margins
def get_resolution_m(gdal_object):
    gt = gdal_object.GetGeoTransform()
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    ext = GetExtent(gt, cols, rows)  # [[влx,влy],[нлx,нлy],[нпy, нпy],[впx, впy]]
    # resolution in meters
    dpx = (ext[3][0] - ext[0][0]) / cols
    dpy = (ext[0][1] - ext[2][1]) / rows
    left_x= ext[0][0]
    bottom_y = ext[2][1]
    return dpx,dpy,left_x,bottom_y
    


# getting data from averaging windows
def createSHPfromDictionary(outputGridfn, data_dict):
    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn, geom_type=ogr.wkbPolygon)

    # create attribute fields
    for el in data_dict:
        # print(data_dict[el]);
        if el == 'id':
            outLayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger));
        else:
            outLayer.CreateField(ogr.FieldDefn(el, ogr.OFTReal));

    # feature definition (needed to address attribute able data)
    featureDefn = outLayer.GetLayerDefn()

    feature_counter = 0;

    # create grid cells
    for idx in range(0, len(data_dict[el]), 1):
        # data_dict={"id":TAB_id,"X_left":TAB_X_left,"X_right":TAB_X_right,\
        #           "Y_top":TAB_Y_top,"Y_bottom":TAB_Y_bottom,base_filename:TAB_raster_value};
        # data_dict['X_left'][idx]
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_top'][idx])
        ring.AddPoint(data_dict['X_right'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_bottom'][idx])
        ring.AddPoint(data_dict['X_left'][idx], data_dict['Y_top'][idx])
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)

        # add new geom to layer
        outFeature = ogr.Feature(featureDefn)
        # outFeature.SetGeometry(poly)
        # outLayer.CreateFeature(outFeature)

        # Setting field data
        for el in data_dict:
            outFeature.SetField(el, float(
                data_dict[el][feature_counter]));  # conversion to FLOAT, cause GDAL dislike NUMPY ARRAY

        # creating of feature MUST be AFTER adding the data
        outFeature.SetGeometry(poly)
        outLayer.CreateFeature(outFeature)
        #
        # outFeature.SetField("id", 0);
        feature_counter = feature_counter + 1;

        outFeature = None

    # Save and close DataSources
    outDataSource = None

#TODO compute averaging windows (returns dict)
#send it a processor function as an arguement
def averaging_windows(func):
    def wrapper(img,win_size=100,
                      spec_field='spec',
                      gdal_obj=None,extent=None,dpxy=None):
        
        if extent!=None and dpxy!=None:
            if dpxy!=None:
                dpx, dpy = dpxy[0],dpxy[1]
            if extent!=None:
                left_x, top_y = extent[0],extent[3]
        elif gdal_obj!=None:
            dpx, dpy, left_x, bottom_y = get_resolution_m(gdal_obj)
        else:
            print('no coordinates were given')
            return None
        if func == None:
            print('no function was passed')
            return None

        data_dict = {'id':[],'X_left':[],
                     'Y_top':[],'X_right':[],
                     'Y_bottom':[],'X_centroid':[],
                     'Y_centroid':[],spec_field:[]}

        win_size = int(win_size // ((dpx+dpy)/2))
        #win_size = win_size

        h, c = np.shape(img)
        pbar_window = ProgressBar()
        id = 0
        print('c=',c)
        print('win_size=',win_size)
        for cc in range(c // win_size + 1):
            for rr in range(h // win_size + 1):
                pbar_window.doProgress(id, (c // win_size + 1)*(h // win_size + 1))
                sub_matrix = img[rr * win_size:(rr + 1) * win_size,
                             cc * win_size:(cc + 1) * win_size]

                #adding geometric fields
                data_dict['id'].append(id)
                data_dict['X_left'].append(np.round((left_x+(cc * win_size)*dpx),2))
                data_dict['X_right'].append(np.round((left_x+(cc+1) * win_size*dpx),2))
                data_dict['Y_top'].append(np.round((top_y - (rr * win_size) * dpy),2))
                data_dict['Y_bottom'].append(np.round((top_y - (rr + 1) * win_size * dpy),2))
                data_dict['X_centroid'].append(np.round(((left_x+(cc * win_size)*dpx) + win_size/2 * dpx),2))
                data_dict['Y_centroid'].append(np.round(((top_y - (rr * win_size) * dpy) - win_size/2 * dpx),2))

                #adding param field
                data_dict[spec_field].append(func(sub_matrix,dpxy=[dpx,dpy]))
                id += 1

        return data_dict
    return wrapper

#TODO функция возвращает фрактальную размерность Минковского для log(1/n) и log(N)
@averaging_windows
def get_minkowski(img,win_size=10,
                      spec_field='spec',
                      gdal_obj=None,extent=None,dpxy=None):
    box_size_list, box_count_list = box_count_fn(img)
    box_size_list_log, box_count_list_log = box_size_num_log(box_size_list, box_count_list)
    try:
         val = np.polyfit(box_size_list_log, box_count_list_log, 1)[0]
         if np.isnan(val):
             val = 0
    except TypeError:
        print('no boxes were found')
        return 0
    return val

@averaging_windows
def get_pixels_sum(img,win_size=10,
                      spec_field='spec',
                      gdal_obj=None,extent=None,dpxy=None):
    return np.sum(img)*((dpxy[0]+dpxy[1])/2)

@averaging_windows
def get_average_val(img,win_size=10,
                      spec_field='spec',
                      gdal_obj=None,extent=None,dpxy=None):
    dpx, dpy = dpxy[0], dpxy[1]
    return np.sum(img)/np.size(img)

#launches when library runs as script

#create rose diagram
def build_rose_diag(lines,txt='Rose diagram'):
    #compute angles and length
    azimuths = []
    lengths = []
    widths = []
    w = np.pi / 10
    for line in lines:
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        length = (dx**2 + dy**2)**0.5
        if dx != 0 and dy != 0:
            angle = np.arctan(dy/dx)
            if dx>0 and dy>0:
                azimuth = np.pi/2 - angle
            elif dx>0 and dy<0:
                azimuth = np.pi / 2 + angle
            elif dx<0 and dy<0:
                azimuth = np.pi + angle
            elif dx<0 and dy>0:
                azimuth = (3/4)*np.pi + angle
        elif dx == 0 and dy != 0:
            azimuth = 0
        elif dx != 0 and dy == 0:
            azimuth = np.pi/2
        else:
            angle = np.nan
        #angles.append(angle+np.pi/2) #we add pi/2 because turning diagram doesnt turn the angles
        azimuths.append(azimuth) #we add pi/2 because turning diagram doesnt turn the angles
        lengths.append(length)
        widths.append(w)

    # duplicate angles with opposite direction and append it to data
    [azimuths.append(ang - np.pi) for ang in azimuths.copy()]
    [lengths.append(r) for r in lengths.copy()]
    [widths.append(w) for w in widths.copy()]

    ax = plt.subplot(111, projection='polar')
    ax.set_title(txt)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bars = ax.bar(azimuths, lengths, width=widths, bottom=0.0,edgecolor='b')

    # Use custom colors and opacity
    for r, bar in zip(lengths, bars):
        #bar.set_facecolor(plt.cm.viridis(r / 10.))
        bar.set_facecolor('b')
        bar.set_alpha(0.5)
    plt.show()

def get_line_azimuth_length(lines):
    azimuths = []
    lengths = []
    for line in lines:
        dx = line[1][0] - line[0][0]
        dy = line[1][1] - line[0][1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        if dx != 0 and dy != 0:
            angle = np.arctan(dy / dx)
            if dx > 0 and dy > 0:
                azimuth = np.pi / 2 - angle
            elif dx > 0 and dy < 0:
                azimuth = np.pi / 2 + angle
            elif dx < 0 and dy < 0:
                azimuth = np.pi + angle
            elif dx < 0 and dy > 0:
                azimuth = (3 / 4) * np.pi + angle
        elif dx == 0 and dy != 0:
            azimuth = 0
        elif dx != 0 and dy == 0:
            azimuth = np.pi / 2
        else:
            angle = np.nan
        # angles.append(angle+np.pi/2) #we add pi/2 because turning diagram doesnt turn the angles
        azimuths.append(azimuth)  # we add pi/2 because turning diagram doesnt turn the angles
        lengths.append(length)
        return lengths,azimuths


def get_line_azimuth_length2(line):
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if dx != 0 and dy != 0:
        angle = np.arctan(dy / dx)
        if dx > 0 and dy > 0:
            azimuth = np.pi / 2 - angle
        elif dx > 0 and dy < 0:
            azimuth = np.pi / 2 + angle
        elif dx < 0 and dy < 0:
            azimuth = np.pi + angle
        elif dx < 0 and dy > 0:
            azimuth = (3 / 4) * np.pi + angle
    elif dx == 0 and dy != 0:
        azimuth = 0
    elif dx != 0 and dy == 0:
        azimuth = np.pi / 2

    return length,np.rad2deg(azimuth)

class ProgressBar(QWidget):

    def __init__(self):
        super().__init__()

        # creating progress bar
        self.pbar = QProgressBar(self)

        # create label
        self.label1 = QLabel('Processing...', self)
        self.label1.resize(140,10)
        self.label1.move(30, 25)

        # setting its geometry
        self.pbar.setGeometry(30, 40, 200, 25)
        self.pbar_val=0 #initial value
        # creating push button
        # self.btn = QPushButton('Start', self)

        # changing its position
        # self.btn.move(40, 80)

        # adding action to push button
        # self.btn.clicked.connect(self.doAction)

        # setting window geometry
        self.setGeometry(300, 300, 280, 80)

        # setting window action
        self.setWindowTitle("Line vectorization")
        self.setWindowModality(Qt.ApplicationModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setWindowFlags(Qt.FramelessWindowHint)

        # set in the center of screen
        sizeObject = QDesktopWidget().screenGeometry(-1)
        # print(" Screen size : "  + str(sizeObject.height()) + "x"  + str(sizeObject.width()))
        self.move(int(sizeObject.width() / 2) - 140, int(sizeObject.height() / 2) - 40)

        # self.pbar.hide()
        # self.pbar.show()
        print('this is progress bar window!')
        # showing all the widgets
        self.show()
        #self.doAction()

    # when button is pressed this method is being called
    def doAction(self):
        # setting for loop to set value of progress bar
        for i in range(101):
            qApp.processEvents()  # обработка событий
            # slowing down the loop
            time.sleep(0.05)
            # setting value to progress bar
            self.pbar.setValue(i)
            # print(self.pbar.value())
        self.close()

    def doProgress(self, cur_val, max_val):
        #qApp.processEvents()  # обработка событий
        time.sleep(0.01)
        pbar_val = int((cur_val / max_val) * 100)
        self.pbar.setValue(pbar_val)
        # set value for label
        self.label1.setText(f'Processing {pbar_val} %')
        qApp.processEvents()

def read_dem_geotiff(fname):
    grid = Grid.from_raster(fname)
    dem = grid.read_raster(fname)
    # plotting
    srtm_gdal_object = gdal.Open(fname)
    return grid, dem, srtm_gdal_object


def enchance_srtm(grid, dem):
    # Fill pits in DEM
    pit_filled_dem = grid.fill_pits(dem)
    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)
    return inflated_dem


def elevation_to_flow(inflated_dem, grid):
    # Elevation to flow direction
    # Determine D8 flow directions from DEM
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # Compute flow directions
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    # Calculate flow accumulation
    acc = grid.accumulation(fdir, dirmap=dirmap)
    return fdir, acc


def detectFlowOrders(grid, fdir, acc, dem, accuracy='mean'):
    if accuracy == 'max':
        threshold = 0
    if accuracy == 'mean':
        threshold = 50
    if accuracy == 'min':
        threshold = 90
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    stream_order = grid.stream_order(fdir, acc > threshold, dirmap=dirmap)
    # remove rivers within sea level
    stream_order[dem == 0] = 0
    return stream_order


def detectFlowNetwork(srtm, accuracy):
    min_area_streams = 50  # analysis parameters

    if accuracy == 'min':
        flood_steps = 9
    if accuracy == 'mean':
        flood_steps = 10
    if accuracy == 'max':
        flood_steps = 11
    # imshape

    r, c = np.shape(srtm)

    # gradual flood

    flows = np.zeros([r, c], dtype=float)
    mins = np.min(srtm)
    maxs = np.max(srtm)
    grad_flood_step = int((maxs - mins) / flood_steps)

    # ВНИМАНИЕ!!!! ДЛЯ ТОЧНОГО ДЕТЕКТИРОВАНИЯ НИКАКОГО РАЗМЫТИЯ
    # for i in range(mins,maxs,100): #
    for i in range(mins, maxs, grad_flood_step):  # flood relief and skeletize it

        thinned = skm.thin(np.int16(srtm <= i))  # надо использовать истончение, а не скелетизацию

        if (i > mins):
            thinned[srtm < (i - 100)] = 0

        flows = flows + thinned
    flows = np.int16(flows > 0)

    # remove orphan streams (area less than 10)
    flows_label = skms.label(np.uint(flows), background=None, return_num=False,
                             connectivity=2)
    for i in range(1, np.max(flows_label), 1):
        if np.sum(flows_label == i) <= min_area_streams:
            flows[flows_label == i] = 0

    # close and thin to remove small holes)

    strel = skm.disk(1)
    flows = skm.closing(flows, strel)
    flows = np.int16(skm.skeletonize_3d(flows))  # need to convert into int8, cause closing returns BOOL

    return flows

#TODO heatmap
#made with https://www.geodose.com/2018/01/creating-heatmap-in-python-from-scratch.html
def get_probability_matrix(x,y,xminmax,yminmax,grid_size,h,aoi_extent=None):
    #GETTING X,Y MIN AND MAX
    x_min=min(x)
    x_max=max(x)
    y_min=min(y)
    y_max=max(y)


    #CONSTRUCT GRID
    if type(grid_size) == int:
        grid_size = [grid_size, grid_size]

    if aoi_extent is None:
        x_grid=np.arange(xminmax[0],xminmax[1],grid_size[0])
        y_grid=np.arange(yminmax[0],yminmax[1],grid_size[1])
    else:
        width = int((aoi_extent[1] - aoi_extent[0]) / grid_size[0])
        heigth = int((aoi_extent[3] - aoi_extent[2]) / grid_size[1])
        x_grid = np.linspace(xminmax[0], xminmax[1], width)
        y_grid = np.linspace(yminmax[0], yminmax[1], heigth)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    #GRID CENTER POINT (arrays)
    xc=x_mesh+(grid_size[0]/2)
    yc=y_mesh+(grid_size[1]/2)

    #FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
    def kde_quartic(d,h):
        dn=d/h
        P=(15/16)*(1-dn**2)**2
        return P

    #PROCESSING
    intensity_list=[]
    pbar_window = ProgressBar()
    for j in range(len(xc)):
        pbar_window.doProgress(j, len(xc))
        intensity_row=[]
        for k in range(len(xc[0])):
            kde_value_list=[]
            for i in range(len(x)):
                #CALCULATE DISTANCE
                d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2)
                #print(d)
                if d<=h:
                    p=kde_quartic(d,h)
                else:
                    p=0
                kde_value_list.append(p)
            #SUM ALL INTENSITY VALUE
            p_total=sum(kde_value_list)
            intensity_row.append(p_total)
        intensity_list.append(intensity_row)

    return np.array(intensity_list)


def get_shp_extent(path_to_file):
    driver = ogr.GetDriverByName('ESRI Shapefile');
    data_source = driver.Open(path_to_file,0); # 0 means read-only. 1 means writeable.
    layer = data_source.GetLayer();
    x_min, x_max, y_min, y_max = layer.GetExtent();
    return x_min, x_max, y_min, y_max;


if __name__=='__main__':
    print('script was activated')
    
    today = date.today()
    if today>exp_date:
        print('application is obsolete! Please, contact developer')
        sys.exit()
    
    #open image
    #img=io.imread('srtm5.tif')
    gdal_object = gdal.Open('srtm5.tif')
    band = gdal_object.GetRasterBand(1)
    rasterData = band.ReadAsArray()
    gt=gdal_object.GetGeoTransform()
    
    cols = gdal_object.RasterXSize
    rows = gdal_object.RasterYSize
    
    #convert to BW
    imgBW=gray2binary(rasterData,'flow')
    
    #do hough transform
    lines=detectPLineHough(imgBW,'medium');
    
    fig4=plt.figure()
    plt.imshow(rasterData, cmap=cm.gray)
    for line in lines:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show() 
    
    #unite lines into faults
    
    faults=uniteLines2(lines)
    """  
    fig5=plt.figure()
    plt.imshow(rasterData, cmap=cm.gray)
    for fault in faults:
        p0, p1 = fault
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show() 
    #try to export
    saveLinesShpFile(lines,'test_export_lines.shp',gdal_object)
    saveLinesShpFile(faults,'test_export_Faults.shp',gdal_object)
    #reproject HOWTO
    #https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    """
    k,b=lineKB(faults,1)
    kb=np.round(np.transpose([k,b]),1)
    kb_r=np.unique(kb,axis=0)
    '''
    k_r=np.round(k,2)
    b_r=np.round(b,2)
    uniq_k_r=np.unique(k_r)
    uniq_b_r=np.unique(b_r)
    '''
       
    
    
            
    fig=plt.figure()
    plt.imshow(rasterData, cmap=cm.gray)
    for fault in faults:
                p0, p1 = fault
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show() 

    

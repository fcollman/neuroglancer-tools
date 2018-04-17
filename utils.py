import numpy as np
import tifffile
import os
from collections import OrderedDict
import json
import sys
import neuroglancer
import pandas as pd


def get_shader(minval,maxval,color=(1.0,1.0,1.0)):
    '''get a neuroglancer shader javascript string

    Parameters
    ----------
    minval: float
        minimum value of lookup function
    maxval: float
        maximum value of lookup function
    color: tuple
        (r,g,b) tuple of colormap, values [0,1]
    
    Returns
    -------
    str
        javascript string suitable to go into layer['shader']
    '''
    redval,greenval,blueval = color 
    shader_template='''
        void emitThresholdColorRGB(vec3 rgb) {{  
            float uMin = {};
            float uMax = {};
            vec3 uColor = vec3({}, {}, {});
            rgb = rgb * uColor; 
            emit(
                vec4(
                    min( max( (rgb.r - uMin) / uMax, 0.0) , 1.0 ),
                    min( max( (rgb.g - uMin) / uMax, 0.0) , 1.0 ),
                    min( max( (rgb.b - uMin) / uMax, 0.0) , 1.0 ),
                    uOpacity
                )
            );
        }}

        void main() {{
          emitThresholdColorRGB(
            vec3(
            toNormalized(getDataValue(0)),
            toNormalized(getDataValue(1)),
            toNormalized(getDataValue(2))
              )
           );
         }}'''.format(minval,maxval,redval,greenval,blueval)
    return shader_template


def set_layer_props(s,l, color=(0.0,1.0,0.0),
                    opacity=1.0,
                    blend='additive',
                    minval=0.0,
                    maxval=.25,
                    shift=(0,0,0)):
    '''  function to set properties of a neuroglancer layer

    Parameters
    ----------
    s: neuroglancer.ViewerState
        neuroglancer viewer state (from viewer.txn())
    l: str
        key for layer you want to set such that layer=s.layers[l]
    color: tuple
        (r,g,b) tuple of color for layer values [0,1] Default (0,1,0)
    opacity: float
        opacity of layer (default=1.0)
    minval: float
        minimum value of lookup function (default=0.0)
    maxval: float
        maximum value of lookup function (default=1.0)
    shift: tuple
        (dx,dy,dz) shift of layer in global coordinates (default=(0,0,0))

    Returns:
    None
    '''
    l=s.layers[l].layer
    (shiftx,shifty,shiftz)=shift
    matrix = [[1,0,0,shiftx*1000],[0,1,0,shifty*1000],[0,0,1,shiftz*1000],[0,0,0,1]]

    l._json_data['color']=color
    l._json_data['blend']=blend
    l._json_data['opacity']=opacity
    l._json_data['max'] = maxval
    l._json_data['transform']=matrix
    l._json_data['shader']=get_shader(minval,maxval,color)


def plot_celldf(s,cell_df,layer_name='cell_pts',
                xyzcol=('x_pos','y_pos','z_pos'),
                desc_col='ID'):
    '''function for adding a dataframe of cell positions to the neuroglancer viewer

    Parameters
    ----------
    s: neuroglancer.ViewerState
        neuroglancer state from viewer.txn()
    cell_df: pd.DataFrame
        pandas DataFrame that contains columns (ID,x_pos,y_pos,z_pos)
        where ID Is the name of the cell, and (x,y,z)_pos are the position of the cells
        in the global voxel coordinate system of the viewer (usually microns).
    layer_name: str
        name of layer to put these points on (default='cell_pts')
    xyzcol: tuple
        x,y,z column names in data frame default=('x_pos','y_pos','z_pos')
    desc_col: str
        column name for description in data frame default='ID'
    Returns
    -------
    None
    '''
    
    layer_name
    s.layers.append(name=layer_name,
                    layer=neuroglancer.PointAnnotationLayer(),
                    )
    cds = []
    for k,row in cell_df.iterrows():
        cd = OrderedDict({'point':[row[xyzcol[0]],row[xyzcol[1]],row[xyzcol[2]]],
                                   'type':'point',
                                   'id':row.get(desc_col,''),
                                   'description':row.get(desc_col,'')})
        cds.append(cd)

    d={'type':'annotation',
      'annotations':cds
      }
    s.layers[layer_name].layer._json_data.update(d)



def fit(A, B):
    """function to fit this transform given the corresponding sets of points A & B

    Parameters
    ----------
    A : numpy.array
        a Nx3 matrix of source points
    B : numpy.array
        a Nx3 matrix of destination points

    Returns
    -------
    numpy.array
        a 12x1 matrix with the best fit parameters
        ordered M00,M01,M02,M10,M11,M12,M20,M21,M22,B0,B1,B2
    """
    if not all([A.shape[0] == B.shape[0], A.shape[1] == B.shape[1] == 3]):
        raise EstimationError(
            'shape mismatch! A shape: {}, B shape {}'.format(
                A.shape, B.shape))

    N = A.shape[0]  # total points

    M = np.zeros((3 * N, 12))
    Y = np.zeros((3 * N, 1))
    for i in range(N):
        M[3 * i, :] =     [A[i, 0], A[i, 1], A[i, 2],\
                                 0,       0,       0,\
                                 0,       0,       0,\
                                 1,       0,       0]
        M[3 * i + 1, :] = [      0,       0,       0,\
                           A[i, 0], A[i, 1], A[i, 2],\
                                 0,       0,       0,\
                                 0,       1,       0]
        M[3 * i + 2, :] = [      0,       0,       0,\
                                 0,       0,       0,\
                           A[i, 0], A[i, 1], A[i, 2],\
                                 0,       0,       1]
        Y[3 * i]     = B[i, 0]
        Y[3 * i + 1] = B[i, 1]
        Y[3 * i + 2] = B[i, 2]

    (Tvec, residuals, rank, s) = np.linalg.lstsq(M, Y)
    print(residuals,rank)

    M=np.zeros((4,4))
    M[0:3,0:3]=np.reshape(Tvec[0:9],(3,3))
    M[0:3,3]=Tvec[9:,0]
    M[3,3]=1
    return M

def tform(M,x):
    xh = np.ones((x.shape[0],4))
    xh[:,:3]=x
    y=np.dot(M,xh.T).T[:,:3]
    return y
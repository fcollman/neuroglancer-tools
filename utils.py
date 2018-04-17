import numpy as np
import tifffile
import os
from collections import OrderedDict
import json
import sys
import neuroglancer
import pandas as pd

def load_zseries(exp):
    fname = os.path.join(exp['Environment']['data_root'],exp['ZSeries']['Frames'][0]['Images'][0])
    data=np.float32(tifffile.imread(fname))
    if len(data.shape)==2:
        Z=len(exp['ZSeries']['Frames'])
        C=len(exp['ZSeries']['Frames'][0]['Images'])
        N=exp['Environment']['PixelsPerLine']
        data = np.zeros((C,Z,N,N),dtype=np.float32)
        for z,d in enumerate(exp['ZSeries']['Frames']):
           
            for c,fname in enumerate(d['Images']):
                filepath = os.path.join(exp['Environment']['data_root'],fname)
                data[c,z,:,:]=tifffile.imread(filepath)
        
    data = data/(1.0*(2**12))
    return data

def sync_mp_ids(mp_json): # Will create an ordered dictionary with TSeries_X_Y: {x,y,z,connection}

    all_mps = OrderedDict()
    all_mps['Stim_Locations'] = OrderedDict()
    all_mps['Filled_Cells'] = OrderedDict()

    with open(mp_json) as mp_file:
        file_data = json.load(mp_file, object_pairs_hook=OrderedDict)

        for elem in file_data:
            if 'MarkPoints' in file_data[elem]:
                for k, mp_dict in file_data[elem]['MarkPoints'].items():
                    all_mps['Stim_Locations'][k] = OrderedDict([('x',mp_dict['x_pos']), ('y',mp_dict['y_pos']), ('z',mp_dict['z_pos'])])

            if elem == 'Headstages':
                for hs in file_data[elem]:
                    all_mps['Filled_Cells'][hs] = OrderedDict()

                    all_mps['Filled_Cells'][hs]['x'] = file_data[elem][hs]['x_pos']
                    all_mps['Filled_Cells'][hs]['y'] = file_data[elem][hs]['y_pos']
                    all_mps['Filled_Cells'][hs]['z'] = file_data[elem][hs]['z_pos']
                    all_mps['Filled_Cells'][hs]['angle'] = file_data[elem][hs]['angle']

                    for tseries in file_data[elem][hs]['Connections']:
                        try:
                            all_mps['Stim_Locations'][tseries][hs] = file_data[elem][hs]['Connections'][tseries]
                        except KeyError: #mean its a repeat and not included 
                            continue

        return all_mps

def display_stim_locs(mp_json): #Displays

    stim_locs = sync_mp_ids(mp_json)
    print(json.dumps(stim_locs, indent=2))


# UNCOMMENT IF YOU WANT TO RUN AND PRINT STIM LOCATIONS
#if __name__ == "__main__":
    #mp_json = sys.argv[1] #name of json file
    #display_stim_locs(mp_json)

    ## testing purposes

def get_voxel_size(experiment):
    '''get voxel size from a ophys json tseries/zseries

    Parameters
    ----------
    experiment: dict
        ophys json
    
    Returns
    -------
    tuple
       xpix,ypix,zpix tuple of voxel sizes in nm
    '''
    env= experiment['Environment']
    xpix_size=env['XAxis_umPerPixel']*1000
    ypix_size=env['YAxis_umPerPixel']*1000
    zpix_size=env['ZAxis_umPerPixel']*1000
    return xpix_size,ypix_size,zpix_size

def get_voxel_offset(experiment):
    '''get voxel offset of image from a ophys json tseries/zseries

    Parameters
    ----------
    experiment: dict
        ophys json
    
    Returns
    -------
    tuple
       x,y,z tuple of voxel offset in nm
    '''
    env= experiment['Environment']
    zvals=np.array([f['ZAxis'] for f in experiment['ZSeries']['Frames']])
    return (env['XAxis']*1000,env['YAxis']*1000,np.min(zvals)*1000)


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


def plot_zseries(s,exp_key,exp_json,data_d):
    '''function to add ZSeries to neuroglancer viewer
    
        Parameters
        ----------
        s: neuroglancer.ViewerState
            neuroglancer viewerstate from viewer.txn()   
        exp_key: str
            experiment key to get data from exp_json and data_d
        exp_json: dict
            dictionary of ophys metadata
        data_d: dict
            dictionary containing numpy C,Z,Y,X arrays of image data
        
        Returns
        -------
        None
    '''
    exp=exp_json[exp_key]
    series_name=exp_key.split('-')[-1]
    data = data_d[exp_key]
    s.layers.append(name='%s_ch0'%series_name,
        layer = neuroglancer.LocalVolume(
            data=data[0,:,:,:],
            offset=get_voxel_offset(exp),
            voxel_size=get_voxel_size(exp)
        ))
    s.layers.append(name='%s_ch1'%series_name,
        layer = neuroglancer.LocalVolume(
            data=data[1,:,:,:],
            offset=get_voxel_offset(exp),
            voxel_size=get_voxel_size(exp)
        ))

    set_layer_props(s,'%s_ch0'%series_name,color=(1.0,0.0,0.0))
    set_layer_props(s,'%s_ch1'%series_name,color=(0.0,1.0,0.0))

def plot_tseries(s,exp_key,exp_json,data_d,color=(1.0,0.0,1.0),maxval=0.125):
    exp=exp_json[exp_key]
    series_name=exp_key.split('-')[-1]
    data = data_d[exp_key]
    s.layers.append(name='%s_ch1'%series_name,
        layer = neuroglancer.LocalVolume(
            data=data[1,:,:,:],
            offset=get_voxel_offset(exp),
            voxel_size=get_voxel_size(exp)
        ))
    set_layer_props(s,'%s_ch1'%series_name,color=color,maxval=maxval)
def get_stim_points_as_df(exp_json,tseries_keys):
    ds=[]
    for exp_key in tseries_keys:
        exp = exp_json[exp_key]
        ds+=exp['MarkPoints'].values()

    return pd.DataFrame(ds)


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
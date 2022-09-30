# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *

import __main__


import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
import numpy as np
import random
import time
def random_property(fy ,fu , epoc):    #材料随机性
    a_y_list = []
    a_u_list = []

    random.seed(1)
    for i in range(50):
        j = epoc * 50 + i
        random.seed(j)
        a_y = random.uniform(0.9, 1.1)
        a_u = random.uniform(0.9, 1.1)
        a_y = round(a_y, 3)
        a_u = round(a_u, 3)
        a_y_list.append(a_y)
        a_u_list.append(a_u)
        real_fy = list(np.array(a_y_list)* fy)
        real_fu = list(np.array(a_u_list) * fu)
    return real_fy, real_fu

def high_temp_constitutive_relation_EC3(fy,fu,E):   #EC3本构，为了隐式计算顺利，进行了最终值的修改
    elastic =np.array([[E, 0.3, 20],[E, 0.3, 100], [0.9 * E, 0.3, 200],[0.8 * E, 0.3, 300],[0.7 * E, 0.3, 400],
               [0.6 * E, 0.3, 500], [0.31 * E, 0.3, 600], [0.13 * E, 0.3, 700], [0.09 * E, 0.3, 800],[0.0675 * E, 0.3, 900],[0.045 * E, 0.3, 1000]])
    fyt = np.array([[fy,20],[fy,100],[fy,200],[fy,300],[fy,400],[0.78*fy,500],[0.47*fy,600],[0.23*fy,700],[0.11*fy,800],[0.06*fy,900],[0.04*fy,1000]])
    fpt = np.array([[fy,20],[fy,100],[0.807*fy,200],[0.613*fy,300],[0.42*fy,400],[0.36*fy,500],[0.18*fy,600],[0.075*fy,700],[0.05*fy,800],[0.0375*fy,900],[0.025*fy,1000]])
    fut = np.array([[fu,20],[fu,100],[fu,200],[fu,300],[fy,400],[0.78*fy,500],[0.47*fy,600],[0.23*fy,700],[0.11*fy,800],[0.06*fy,900],[0.04*fy,1000]])
    sigma_pt = np.zeros((fpt.shape[0],2))
    c = np.zeros((fpt.shape[0],2))
    a = np.zeros((fpt.shape[0],2))
    b = np.zeros((fpt.shape[0],2))
    for i in range(fpt.shape[0]):
        sigma_pt[i] = [fpt[i][0]/elastic[i][0],fpt[i][1]]
        c[i] = [np.power(fyt[i][0]-fpt[i][0], 2)/((0.02-sigma_pt[i][0])*elastic[i][0]-2*(fyt[i][0]-fpt[i][0])),fpt[i][1]]
        b[i] = [np.sqrt(c[i][0]*(0.02-sigma_pt[i][0])*elastic[i][0] + np.power(c[i][0],2)),fpt[i][1]]
        a[i] = [np.sqrt((0.02-sigma_pt[i][0])*(0.02-sigma_pt[i][0] + c[i][0] / elastic[i][0])),fpt[i][1]]


    for j in range(5):

        x = [1,(sigma_pt[j][0]+(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+2*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+3*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+4*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+5*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+6*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+7*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+8*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+9*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
             (sigma_pt[j][0]+10*(0.02-sigma_pt[j][0])/10)/sigma_pt[j][0],
              0.03/sigma_pt[j][0], 0.04/sigma_pt[j][0],0.1/sigma_pt[j][0], 0.15/sigma_pt[j][0], 0.2/sigma_pt[j][0]]
        norm_sigma = np.array([m *sigma_pt[j][0] for m in x])
        norm_stress =np.array([norm_sigma[0]*elastic[j][0],
                      fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[1])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[2])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[3])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[4])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[5])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[6])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[7])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[8])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[9])**2),
                       fpt[j][0]-c[j][0]+b[j][0]/a[j][0]*np.sqrt(a[j][0]**2-(0.02-norm_sigma[10])**2),
                       50*(fut[j][0]-fyt[j][0])*norm_sigma[11] + 2*fyt[j][0] -fut[j][0], 50*(fut[j][0]-fyt[j][0])*norm_sigma[12] + 2*fyt[j][0] -fut[j][0],
                       fut[j][0],fut[j][0],1000000])
        real_sigma = np.log(1+norm_sigma)

        real_stress = norm_stress * (1+norm_sigma)
        no_elastic_stress = real_sigma - real_stress/elastic[j][0]
        no_elastic_stress [0] = 0
        real_stress[15] = 10000000
        t = np.ones(norm_stress.shape[0]) * fyt[j][1]
        result = np.transpose(np.array([real_stress,no_elastic_stress,t]))
        if j == 0:
            final_result = result
        else:
            final_result = np.append(final_result,result,axis = 0)
        #np.append(final_result,result)
    for n in range(5,11):
        x = [1, (sigma_pt[n][0] + (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 2 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 3 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 4 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 5 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 6 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 7 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 8 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 9 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             (sigma_pt[n][0] + 10 * (0.02 - sigma_pt[n][0]) / 10) / sigma_pt[n][0],
             0.03 / sigma_pt[n][0], 0.04 / sigma_pt[n][0], 0.1 / sigma_pt[n][0], 0.15 / sigma_pt[n][0], 0.2 / sigma_pt[n][0]]
        norm_sigma = np.array([m * sigma_pt[n][0] for m in x])
        norm_stress = np.array([norm_sigma[0] * elastic[n][0],
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[1]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[2]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[3]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[4]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[5]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[6]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[7]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[8]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[9]) ** 2),
                                fpt[n][0] - c[n][0] + b[n][0] / a[n][0] * np.sqrt(a[n][0] ** 2 - (0.02 - norm_sigma[10]) ** 2),
                                fyt[n][0], fyt[n][0],
                                fyt[n][0], fyt[n][0], 1000000])
        real_sigma = np.log(1 + norm_sigma)
        real_stress = norm_stress * (1 + norm_sigma)
        no_elastic_stress = real_sigma - real_stress / elastic[n][0]
        no_elastic_stress[0] = 0
        if n==5:

            real_stress[15] = 100000000
        elif n==6:
            real_stress[15] = 100000000
        else :
            real_stress[15] = real_stress[14]-1000000
        t = np.ones(norm_stress.shape[0]) * fyt[n][1]
        result = np.transpose(np.array([real_stress, no_elastic_stress, t]))
        if n == 0:
            final_result = result
        else:
            final_result = np.append(final_result,result,axis = 0)
    final_result = tuple([tuple(e) for e in final_result])
    elastic =  tuple([tuple(p) for p in elastic])
    return final_result, elastic

def random_load(qmax,epoc):   #随机荷载，考虑左右均不相同
    l = np.zeros(8)
    r = np.zeros(8)
    L = np.zeros((50, 8))
    R = np.zeros((50, 8))
    for j in range(50):
        for i in range(8):
            z = j * 8 + i + 400 * epoc
            random.seed(z)
            if i == 0:
                l[i] = round(random.uniform(0.2,0.3),2)  #保留两位小数
                r[i] = round(random.uniform(0.2,0.3),2)
            elif i == 7:
                l[i] = round(random.uniform(0.2, 0.3),2)
                r[i] = round(random.uniform(0.2, 0.3),2)
            else:
                l[i] = round(random.uniform(0.4, 0.6),2)
                r[i] = round(random.uniform(0.4, 0.6),2)
        L[j,:] = l * qmax
        R[j,:] = r * qmax
    return L,R

def random_fire(epoc):
    fire_span_string = ["S1","S2","S3","S4","S5"]
    fire_bay_string = ["By1","By2"]
    final_span_string = []
    final_bay_string = []
    for i in range(50):
        j = epoc * 50 + i
        np.random.seed(j)
        final_span = np.random.randint(0,5)
        final_bay = np.random.randint(0,2)
        final_span_string.append(fire_span_string[final_span])
        final_bay_string.append(fire_bay_string[final_bay])
    return final_span_string, final_bay_string

t = np.ones(361)
def air_temp_field(beta,in_plane_fire):   #beta:升温指数 in_plane_fire：平面内火源工况
    tg_max = 1200

    for i in range(361):
        t[i] =i * 10
    #print(t)
    tg = tg_max*(1-0.8*np.exp(-beta*t)-0.2*np.exp(-0.1*beta*t))+20
    #print(tg)
    if in_plane_fire == "S1":
        reduce_factor = np.array([1,0.84,0.69,0.64,0.61])
    elif in_plane_fire == "S2":
        reduce_factor = np.array([0.84,1,0.84,0.69,0.64])
    elif in_plane_fire == "S3":
        reduce_factor = np.array([0.69, 0.84, 1, 0.84, 0.69])
    elif in_plane_fire == "S4":
        reduce_factor = np.array([0.64, 0.69, 0.84, 1, 0.84])
    else:
        reduce_factor = np.array([0.61, 0.64, 0.69, 0.84, 1])
    final_tg_1 = tg * reduce_factor[0]
    final_tg_2 = tg * reduce_factor[1]
    final_tg_3 = tg * reduce_factor[2]
    final_tg_4 = tg * reduce_factor[3]
    final_tg_5 = tg * reduce_factor[4]
    final_tg = np.hstack((final_tg_1,final_tg_2,final_tg_3,final_tg_4,final_tg_5)).reshape(5,-1)
    return(final_tg)

def steel_temp_field(air_temp_field):    #空气温度场
    t = np.ones(361)
    for i in range(361):
        t[i] = i * 10
    chord_temp = np.round(air_temp_field * (1-np.exp(-0.0008*t))+20,1)
    web_temp = np.round(air_temp_field * (1-np.exp(-0.001*t))+20,1)
    #print(chord_temp)
    final_chord_temp = np.zeros((5, 361, 2))
    final_web_temp = np.zeros((5, 361, 2))
    for k in range(5):
        chord_temp_single = chord_temp[k]
        chord_temp_single = np.transpose(np.array([t/10,chord_temp_single]))
        #print(chord_temp_single.shape[0],chord_temp_single.shape[1])
        web_temp_single = web_temp[k]
        web_temp_single = np.transpose(np.array([t/10,web_temp[k]]))
        #print(final_chord_temp)
        final_chord_temp[k,:,:] = chord_temp_single
        final_web_temp[k,:,:] = web_temp_single
    #final_result = tuple([tuple(e) for e in final_web_temp[0,:,:]])
    return final_chord_temp, final_web_temp

def random_beta(epoc):   #空气升温曲线，随机选择
    beta = [0.0004,0.0008,0.0018,0.002]
    final_beta = []
    for i in range(50):
        j = epoc * 50 + i
        np.random.seed(j)
        index = np.random.randint(0,4)
        final_beta.append(beta[index])
    return final_beta

epoc = 1
fy = 235000000.0
fu = 294000000.0
E = 210000000000.0
model_fy, model_fu = random_property(fy ,fu , epoc)
model_load_L, model_load_R = random_load(-23000,epoc)
model_fire_span, model_fire_bay = random_fire(epoc)
model_beta = random_beta(epoc)
#print(model_fy)
#print(model_fu)
#print(model_load_L)
#print(model_load_R)
#print(model_fire_span)
#print(model_fire_bay)
#print(model_beta)

for num_t in range(50):   #计算一组，50个工况
    model_name = "Model-" + str(num_t+1)
    fire_span_string = model_fire_span[num_t]
    fire_bay_string = model_fire_bay[num_t]
    load_L = model_load_L[0]
    load_R = model_load_R[0]
    #print(load_L)
    #print(load_R)
    #计算升温曲线
    cv = air_temp_field(model_beta[num_t], fire_span_string)
    chord_temp, web_temp = steel_temp_field(cv)
    chord_temp_tuple_1 = tuple([tuple(e) for e in chord_temp[0, :, :]])     #将数组均转化为元组的形式
    web_temp_tuple_1 = tuple([tuple(e) for e in web_temp[0, :, :]])
    chord_temp_tuple_2 = tuple([tuple(e) for e in chord_temp[1, :, :]])
    web_temp_tuple_2 = tuple([tuple(e) for e in web_temp[1, :, :]])
    chord_temp_tuple_3 = tuple([tuple(e) for e in chord_temp[2, :, :]])
    web_temp_tuple_3 = tuple([tuple(e) for e in web_temp[2, :, :]])
    chord_temp_tuple_4 = tuple([tuple(e) for e in chord_temp[3, :, :]])
    web_temp_tuple_4 = tuple([tuple(e) for e in web_temp[3, :, :]])
    chord_temp_tuple_5 = tuple([tuple(e) for e in chord_temp[4, :, :]])
    web_temp_tuple_5 = tuple([tuple(e) for e in web_temp[4, :, :]])
    ####结束
    mdb.Model(name=model_name,
              modelType=STANDARD_EXPLICIT)
    iges = mdb.openIges('G:/jinyu/liangce-trapzoid-LSTM/rinho/span-24-6-many.igs',    #导入的模型已上传
        msbo=False, trimCurve=DEFAULT, topology=WIRE, scaleFromFile=OFF)
    mdb.models[model_name].PartFromGeometryFile(name='span-24-6-many',
        geometryFile=iges, combine=False, stitchTolerance=1.0,
        dimensionality=THREE_D, type=DEFORMABLE_BODY, topology=WIRE,
        convertToAnalytical=1, stitchEdges=1)



    mdb.models[model_name].Material(name='steel')
    mdb.models[model_name].materials['steel'].Density(table=((7850.0, ), ))

    yield_stress, model_elastic = high_temp_constitutive_relation_EC3(model_fy[num_t], model_fu[num_t], E)    #求该次建模的本构

    mdb.models[model_name].materials['steel'].Elastic(temperatureDependency=ON,
        table= model_elastic)
    mdb.models[model_name].materials['steel'].Plastic(temperatureDependency=ON,
        table=yield_stress )
    mdb.models[model_name].materials['steel'].plastic.RateDependent(
        type=YIELD_RATIO, table=((1.0, 0.0), (1.0, 0.0001), (1.125368322,
        0.001), (1.26645386, 0.01), (1.425227055, 0.1), (1.603905379, 1.0), (
        1.637613047, 1.5), (1.672029115, 2.25), (1.707168472, 3.375), (
        1.743046316, 5.0625), (1.779678169, 7.59375), (1.817079877, 11.390625),
        (1.85526762, 17.0859375), (1.894257915, 25.62890625), (1.934067631,
        38.44335938), (1.974713987, 57.66503906), (2.016214567, 86.49755859), (
        2.058587323, 129.7463379), (2.101850585, 194.6195068), (2.117547727,
        225.0)))
    mdb.models[model_name].materials['steel'].Expansion(table=((1.4e-05, ), ))
    mdb.models[model_name].PipeProfile(name='NO1', r=0.0795, t=0.0045)
    mdb.models[model_name].PipeProfile(name='NO2', r=0.07, t=0.0045)
    mdb.models[model_name].PipeProfile(name='NO3', r=0.035, t=0.002)
    mdb.models[model_name].PipeProfile(name='NO4', r=0.0605, t=0.0035)
    mdb.models[model_name].PipeProfile(name='NO5', r=0.051, t=0.0025)
    mdb.models[model_name].PipeProfile(name='NO8', r=0.0415, t=0.002)
    mdb.models[model_name].PipeProfile(name='XG', r=0.0445, t=0.003)
    mdb.models[model_name].PipeProfile(name='bottom chord sc', r=0.035, t=0.0025)
    mdb.models[model_name].PipeProfile(name='cc1-11-brace', r=0.03, t=0.0025)
    mdb.models[model_name].PipeProfile(name='cc1-11-chord', r=0.0445, t=0.003)
    mdb.models[model_name].PipeProfile(name='top chord sc', r=0.03, t=0.0025)
    mdb.models[model_name].BeamSection(name='NO1', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO1', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='NO2', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO2', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='NO3', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO3', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='NO4', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO4', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='NO5', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO5', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='NO8', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='NO8', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='XG', integration=DURING_ANALYSIS,
        poissonRatio=0.0, profile='XG', material='steel',
        temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='bottom chord sc',
        integration=DURING_ANALYSIS, poissonRatio=0.0,
        profile='bottom chord sc', material='steel', temperatureVar=LINEAR,
        consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='cc1-11-brace',
        integration=DURING_ANALYSIS, poissonRatio=0.0, profile='cc1-11-brace',
        material='steel', temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='cc1-11-chord',
        integration=DURING_ANALYSIS, poissonRatio=0.0, profile='cc1-11-chord',
        material='steel', temperatureVar=LINEAR, consistentMassMatrix=False)
    mdb.models[model_name].BeamSection(name='top chord sc',
        integration=DURING_ANALYSIS, poissonRatio=0.0, profile='top chord sc',
        material='steel', temperatureVar=LINEAR, consistentMassMatrix=False)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.3153,
                                                    farPlane=125.781, width=49.2353, height=47.8708, cameraPosition=(
            0.240356, -73.0509, 15.1222), cameraUpVector=(0.0562292, 0.474874,
                                                          0.878255), cameraTarget=(-0.654214, 25.1401, -0.192823),
                                                    viewOffsetX=-0.528077, viewOffsetY=-1.74193)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=63.0488,
                                                    farPlane=126.07, width=48.2658, height=46.9281, cameraPosition=(
            7.55921, -70.2327, 25.0586), cameraUpVector=(-0.004305, 0.567848,
                                                         0.823122), cameraTarget=(-1.08826, 25.3655, -0.6908),
                                                    viewOffsetX=-0.517678, viewOffsetY=-1.70763)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.254,
                                                    farPlane=124.967, width=49.1884, height=47.8252, cameraPosition=(
            -0.485303, -70.001, 27.2517), cameraUpVector=(0.0156775, 0.585827,
                                                          0.810285), cameraTarget=(-0.698438, 25.3422, -0.792257),
                                                    viewOffsetX=-0.527573, viewOffsetY=-1.74027)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=63.6729,
                                                    farPlane=125.719, width=48.7436, height=47.3927, cameraPosition=(
            -5.17989, -70.029, 26.9506), cameraUpVector=(0.0311781, 0.582184,
                                                         0.812459), cameraTarget=(-0.465497, 25.2948, -0.761911),
                                                    viewOffsetX=-0.522802, viewOffsetY=-1.72453)
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.5476,
                                                    farPlane=125.656, width=49.6833, height=48.3064, cameraPosition=(
            0.483052, -75.6019, 2.22343), cameraTarget=(0.483052, 21, 2.22343))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#7f39e7 #39ff3ff8 #2fe3b818 #60ffb87f #76edbff #f07f3c3e #fbee3bf8',
        ' #3018ffda #73fe10c #ffae5b9f #ca1fecf0 #7fbf81a5 #1cd10f6 #e707fe70',
        ' #1bf1386c #c0eb8dee #6700a8e7 #5738097b #2d8d6f1e #171ba1 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.4501,
                                                    farPlane=128.642, cameraPosition=(-8.60281, -75.0453, -2.74295),
                                                    cameraUpVector=(0.00653055, -0.0522548, 0.998612))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.5273,
                                                    farPlane=127.68, cameraPosition=(0.459703, -75.5953, 3.31851),
                                                    cameraUpVector=(-0.000201353, 0.0113722, 0.999935), cameraTarget=(
            0.477831, 21.0003, 2.21993))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#0:6 #400 #48000000 #0:5 #100000 #4000200 #0', ' #1200 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.5812,
                                                    farPlane=124.834, cameraPosition=(7.52938, -56.4554, 59.516),
                                                    cameraUpVector=(0.0123815, 0.595379, 0.80335),
                                                    cameraTarget=(0.477904,
                                                                  21.0005, 2.22051))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.5782,
                                                    farPlane=122.792, cameraPosition=(0.89348, -50.6856, 66.9738),
                                                    cameraUpVector=(-0.0183681, 0.67006, 0.742079),
                                                    cameraTarget=(0.470633,
                                                                  21.0068, 2.22868))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=71.3893,
                                                    farPlane=121.933, cameraPosition=(-6.21556, -43.1022, 74.1827),
                                                    cameraUpVector=(-0.0139836, 0.747146, 0.664513), cameraTarget=(
            0.464511, 21.0133, 2.23489))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=77.6919,
                                                    farPlane=115.716, cameraPosition=(-3.92844, -15.0779, 91.7375),
                                                    cameraUpVector=(-0.0318809, 0.927429, 0.372637), cameraTarget=(
            0.465919, 21.0306, 2.2457))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.0288,
                                                    farPlane=121.38, width=97.0378, height=94.3484, cameraPosition=(
            1.16533, -5.5668, 95.8252), cameraTarget=(5.55969, 30.5417, 6.3334))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=76.3958,
                                                    farPlane=122.613, cameraPosition=(-0.19964, 20.712, 102.265),
                                                    cameraUpVector=(-0.0154431, 0.994737, 0.101295),
                                                    cameraTarget=(5.55824,
                                                                  30.5695, 6.34021))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.1726,
                                                    farPlane=115.836, width=50.8758, height=49.4658, cameraPosition=(
            -1.01854, 7.15686, 100.823), cameraTarget=(4.73934, 17.0144, 4.89809))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#80008218 #8200c007 #901803c7 #9c000380 #f0112400 #f804300 #4118003',
        ' #87c60001 #d0400ee3 #a020 #20800002 #80003c40 #2a001 #18e00182',
        ' #e002c113 #19005200 #8a14100 #80814000 #90300061 #448 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.1726,
                                                    farPlane=115.836, width=79.4934, height=77.2903, cameraPosition=(
            -2.61999, 9.57444, 100.975), cameraTarget=(3.13789, 19.4319, 5.0504))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.5675,
                                                    farPlane=136.462, cameraPosition=(-1.73841, -70.2184, 44.4202),
                                                    cameraUpVector=(-0.00317658, 0.425318, 0.905039), cameraTarget=(
            3.16361, 17.1043, 3.4006))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.2152,
                                                    farPlane=133.251, cameraPosition=(-5.68914, -53.2568, 69.2693),
                                                    cameraUpVector=(0.0479037, 0.670607, 0.740265),
                                                    cameraTarget=(3.0098,
                                                                  17.7646, 4.36802))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#7f804400 #44000000 #40044420 #3004400 #8800000 #80c1 #4004',
        ' #210024 #28801010 #510440 #1560130d #40421a #fe304f08 #8000d',
        ' #c0480 #26142011 #905e0418 #2846b684 #42429080 #28e016 ]',), )
    region = p.Set(edges=edges, name='top chord')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO1', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.replace(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.0143,
                                                    farPlane=128.46, width=40.7006, height=39.5726, cameraPosition=(
            -4.94295, -52.739, 69.936), cameraTarget=(3.75599, 18.2824, 5.03468))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.7419,
                                                    farPlane=128.268, cameraPosition=(-7.00715, -62.4885, 57.0253),
                                                    cameraUpVector=(0.0245337, 0.544583, 0.838348),
                                                    cameraTarget=(3.67126,
                                                                  17.8822, 4.50469))
    session.viewports['Viewport: 1'].view.setProjection(projection=PERSPECTIVE)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.3593,
                                                    farPlane=125.865, width=30.908, height=30.0514, cameraPosition=(
            -30.9977, -50.1987, 63.8763), cameraUpVector=(0.105294, 0.619104,
                                                          0.778218), cameraTarget=(2.73955, 18.3595, 4.77076))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.1104,
                                                    farPlane=129.113, width=57.6937, height=56.0948,
                                                    viewOffsetX=-2.62494,
                                                    viewOffsetY=-1.12694)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#ffc1effc #efc7eccf #f87cefef #ffb3efd9 #fefffec9 #ffede7e3 #bef3ee6f',
        ' #ffff99ff #f9f83fff #b8fbfef1 #77fe3bff #ccc0ffff #ff3fffdd #dffcb19f',
        ' #ee2fdffb #7f957ffb #ffffff7e #ffdff7d4 #def7d8ff #3def5f ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.6328,
                                                    farPlane=119.595, cameraPosition=(-17.8412, -40.4066, 73.7575),
                                                    cameraUpVector=(0.13948, 0.737992, 0.660237), cameraTarget=(0, 21,
                                                                                                                1.35))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.6525,
                                                    farPlane=126.575, width=82.8055, height=80.5106, cameraPosition=(
            -13.6271, -35.9167, 78.6036), cameraTarget=(4.21413, 25.4899, 6.1961))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=76.6034,
                                                    farPlane=125.231, cameraPosition=(5.201, -8.43546, 96.7219),
                                                    cameraUpVector=(-0.0495082, 0.933899, 0.354094),
                                                    cameraTarget=(4.40931,
                                                                  25.7748, 6.38392))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.2243,
                                                    farPlane=125.693, cameraPosition=(-10.2974, -5.04854, 96.8069),
                                                    cameraUpVector=(-0.02261, 0.94688, 0.32079), cameraTarget=(3.7466,
                                                                                                               25.9196,
                                                                                                               6.38755))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=80.083,
                                                    farPlane=123.267, cameraPosition=(-7.02618, 10.5299, 101.2),
                                                    cameraUpVector=(0.00836489, 0.985837, 0.167497),
                                                    cameraTarget=(3.87219,
                                                                  26.5177, 6.55624))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#1 #10000300 #820010 #0:4 #6600 #60000 #700000c',
        ' #1c000 #0:5 #1 #8 #0 #1080 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.9286,
                                                    farPlane=123.813, cameraPosition=(0.234253, 40.2942, 102.346),
                                                    cameraUpVector=(0.0310457, 0.991543, -0.126009),
                                                    cameraTarget=(4.23447,
                                                                  28.0029, 6.61343))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=80.0526,
                                                    farPlane=123.552, cameraPosition=(-0.453256, 5.31004, 100.851),
                                                    cameraUpVector=(0.0749368, 0.974231, 0.212741),
                                                    cameraTarget=(4.18635,
                                                                  25.5545, 6.50879))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=89.9594,
                                                    farPlane=113.645, width=27.1337, height=26.3817, cameraPosition=(
            3.29653, 13.1521, 102.718), cameraTarget=(7.93613, 33.3965, 8.37598))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#0 #10 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#0:15 #80000 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=86.0933,
                                                    farPlane=117.511, width=66.2445, height=64.4085, cameraPosition=(
            -9.66451, 5.41714, 100.421), cameraTarget=(-5.0249, 25.6616, 6.07878))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=84.7319,
                                                    farPlane=118.478, cameraPosition=(10.9852, 18.9228, 101.269),
                                                    cameraUpVector=(-0.00928765, 0.996809, 0.0792873), cameraTarget=(
            -3.96992, 26.3516, 6.12211))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#0:2 #4000000 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=77.1808,
                                                    farPlane=126.029, width=82.8056, height=80.5107, cameraPosition=(
            22.3089, 11.8394, 98.9361), cameraTarget=(7.35382, 19.2682, 3.7892))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#200000:2 #3000000 #c0002 #1000016 #1808 #1180 #0',
        ' #4008000 #0 #80000000 #23160000 #800002 #20010c40 #11000000',
        ' #0:2 #200000 #21002700 #20 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.1318,
                                                    farPlane=127.648, cameraPosition=(10.1377, -24.1979, 90.4467),
                                                    cameraUpVector=(0.0184771, 0.902073, 0.431188),
                                                    cameraTarget=(6.75441,
                                                                  17.4934, 3.37112))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#1e1002 #181020 #11000 #401024 #120 #120014 #410c0010',
        ' #0 #2014000 #40040102 #8000400 #10290000 #400020 #24220',
        ' #d02004 #80628004 #80 #823 #80000 #20000 ]',), )
    region = p.Set(edges=edges, name='Set-2')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO2', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.9368,
                                                    farPlane=131.843, width=129.384, height=125.798, cameraPosition=(
            7.44305, -21.4068, 91.8877), cameraTarget=(4.05976, 20.2845, 4.81216))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.replace(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=52.8013,
                                                    farPlane=145.809, cameraPosition=(-1.63193, -69.122, 43.2109),
                                                    cameraUpVector=(0.0149707, 0.415305, 0.909559),
                                                    cameraTarget=(3.73618,
                                                                  18.5831, 3.07654))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=51.9862,
                                                    farPlane=146.622, cameraPosition=(1.76175, -76.274, 21.634),
                                                    cameraUpVector=(0.0188818, 0.197801, 0.98006),
                                                    cameraTarget=(3.82859,
                                                                  18.3884, 2.48901))
    session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=56.3655,
                                                    farPlane=136.838, width=118.661, height=115.372, cameraPosition=(
            16.7101, -75.6019, -6.3968), cameraTarget=(16.7101, 21, -6.3968))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=38.4952,
                                                    farPlane=130.7, cameraPosition=(-32.5813, -49.8611, 36.9736),
                                                    cameraUpVector=(0.205365, 0.403007, 0.891858),
                                                    cameraTarget=(16.7101,
                                                                  21, -6.3968))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=36.982,
                                                    farPlane=120.651, cameraPosition=(-52.9453, -20.718, 44.1307),
                                                    cameraUpVector=(0.422778, 0.337421, 0.841074),
                                                    cameraTarget=(19.5997,
                                                                  16.8646, -7.4124))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=39.5136,
                                                    farPlane=110.443, cameraPosition=(-56.0166, 5.31161, 50.184),
                                                    cameraUpVector=(0.58302, 0.236371, 0.777314), cameraTarget=(20.2927,
                                                                                                                10.991,
                                                                                                                -8.77834))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=58.5163,
                                                    farPlane=91.4405, width=31.1063, height=30.2442, cameraPosition=(
            -59.2684, 0.597892, 45.5215), cameraTarget=(17.0409, 6.27728,
                                                        -13.4408))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=54.8518,
                                                    farPlane=90.8382, cameraPosition=(-58.9029, 10.5638, 45.8665),
                                                    cameraUpVector=(0.614781, 0.214483, 0.758974),
                                                    cameraTarget=(16.9356,
                                                                  3.40315, -13.5403))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=49.2021,
                                                    farPlane=96.4879, width=73.7335, height=71.69, cameraPosition=(
            -55.7019, 30.0221, 47.6075), cameraTarget=(20.1366, 22.8614, -11.7993))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=38.9214,
                                                    farPlane=103.695, cameraPosition=(-35.7412, -11.602, 53.882),
                                                    cameraUpVector=(0.403236, 0.581634, 0.706472),
                                                    cameraTarget=(13.6269,
                                                                  36.4363, -13.8456))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=40.2923,
                                                    farPlane=102.121, cameraPosition=(-4.5211, -24.0553, 56.4414),
                                                    cameraUpVector=(0.250302, 0.702942, 0.665748),
                                                    cameraTarget=(2.55275,
                                                                  40.8537, -14.7534))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=39.5439,
                                                    farPlane=103.432, cameraPosition=(15.8971, -21.9124, 56.408),
                                                    cameraUpVector=(0.161155, 0.766721, 0.621424),
                                                    cameraTarget=(-4.7292,
                                                                  40.0895, -14.7415))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=31.2114,
                                                    farPlane=107.556, cameraPosition=(-26.016, -38.6704, 25.5837),
                                                    cameraUpVector=(0.154598, 0.265814, 0.951547),
                                                    cameraTarget=(9.99481,
                                                                  45.9766, -3.91295))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=32.0785,
                                                    farPlane=106.812, cameraPosition=(18.6872, -39.7157, 29.5915),
                                                    cameraUpVector=(0.0214459, 0.382789, 0.923587),
                                                    cameraTarget=(-7.54177,
                                                                  46.3867, -5.48518))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=36.5378,
                                                    farPlane=103.58, cameraPosition=(-0.697922, -36.5823, 41.3779),
                                                    cameraUpVector=(0.0284835, 0.532445, 0.845985), cameraTarget=(
            0.0389371, 45.1613, -10.0943))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=46.5832,
                                                    farPlane=93.5345, width=30.2013, height=29.3642, cameraPosition=(
            -6.25539, -38.3437, 38.501), cameraTarget=(-5.51853, 43.3999,
                                                       -12.9712))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=43.3033,
                                                    farPlane=101.792, cameraPosition=(-36.3255, -37.4065, 24.9913),
                                                    cameraUpVector=(0.0506166, 0.354611, 0.933643),
                                                    cameraTarget=(5.87411,
                                                                  43.0448, -7.85279))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=40.2637,
                                                    farPlane=94.8553, cameraPosition=(50.5902, -17.4363, 24.3556),
                                                    cameraUpVector=(-0.179568, 0.309582, 0.933763),
                                                    cameraTarget=(-22.9438,
                                                                  36.4235, -7.642))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=34.4306,
                                                    farPlane=100.688, width=73.7335, height=71.6901, cameraPosition=(
            54.2621, -9.40655, 29.4332), cameraTarget=(-19.2719, 44.4532,
                                                       -2.56439))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=31.2377,
                                                    farPlane=102.948, cameraPosition=(36.3861, -17.7515, 43.4081),
                                                    cameraUpVector=(-0.30628, 0.442554, 0.842816),
                                                    cameraTarget=(-11.5874,
                                                                  48.0406, -8.57194))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=33.8753,
                                                    farPlane=98.5449, cameraPosition=(8.65544, -21.5869, 52.2178),
                                                    cameraUpVector=(-0.161226, 0.652893, 0.740092),
                                                    cameraTarget=(0.609462,
                                                                  49.7275, -12.4467))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=31.3415,
                                                    farPlane=98.7229, cameraPosition=(-7.69219, -23.0329, 49.5376),
                                                    cameraUpVector=(0.106405, 0.620186, 0.777205),
                                                    cameraTarget=(8.11341,
                                                                  50.3913, -11.2164))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=27.8386,
                                                    farPlane=99.3186, cameraPosition=(-11.3935, -27.8013, 41.5952),
                                                    cameraUpVector=(0.184276, 0.474617, 0.860687), cameraTarget=(9.9102,
                                                                                                                 52.7061,
                                                                                                                 -7.36076))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#410124 #1012000 #20008008 #808041 #2a9200 #90200020 #88200000',
        ' #81092 #200004 #204810 #40080090 #40800080 #1050000 #80008000',
        ' #10060 #520 #45000840 #40080040 #8010012 #100201 ]',), )
    region = p.Set(edges=edges, name='vertical brace')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO3', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=35.13,
                                                    farPlane=92.0273, width=47.1895, height=45.8816, cameraPosition=(
            -12.5231, -32.706, 33.038), cameraTarget=(8.78057, 47.8014, -15.918))
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.1974,
                                                    farPlane=126.006, width=51.8438, height=50.4069, cameraPosition=(
            3.81652, -75.6019, -1.15278), cameraTarget=(3.81652, 21, -1.15278))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.0802,
                                                    farPlane=128.604, cameraPosition=(19.7207, -67.9015, 33.1327),
                                                    cameraUpVector=(-0.0572661, 0.3503, 0.934885),
                                                    cameraTarget=(3.81652,
                                                                  21, -1.15278))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#2000 #80 #400000 #120000 #8 #0:2 #48',
                                        ' #80000 #0 #2000 #21 #0 #40000000 #0', ' #10000 #0:2 #4800000 ]',), )
    region = p.Set(edges=edges, name='brace connected to support')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO4', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=64.0372,
                                                    farPlane=126.868, cameraPosition=(-2.82917, -69.2221, 32.7049),
                                                    cameraUpVector=(0.0475666, 0.347824, 0.936353),
                                                    cameraTarget=(3.87736,
                                                                  21.0036, -1.15163))
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.7226,
                                                    farPlane=125.481, width=48.6035, height=47.2565, cameraPosition=(
            6.39838, -75.6019, 2.31548), cameraTarget=(6.39838, 21, 2.31548))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.9711,
                                                    farPlane=126.899, cameraPosition=(-2.19672, -66.3916, 42.574),
                                                    cameraUpVector=(0.164153, 0.399367, 0.901976))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=63.5089,
                                                    farPlane=129.361, width=75.943, height=73.8383, cameraPosition=(
            -3.92508, -65.7835, 43.5251), cameraTarget=(4.67003, 21.6081, 3.26654))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=60.3478,
                                                    farPlane=132.317, cameraPosition=(-8.33298, -62.9061, 48.2121),
                                                    cameraUpVector=(0.192204, 0.437645, 0.878365),
                                                    cameraTarget=(4.67766,
                                                                  21.6031, 3.25843))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.8577,
                                                    farPlane=124.807, width=31.1063, height=30.2442, cameraPosition=(
            -9.6038, -65.4036, 43.1493), cameraTarget=(3.40684, 19.1057, -1.80436))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#880 #20000040 #202000 #212008 #840 #90000 #30000000',
        ' #0 #1102000 #200 #2000800 #8004 #80 #1002000',
        ' #0 #48 #4 #0 #40000:2 ]',), )
    region = p.Set(edges=edges, name='brace-5-7')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO5', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.8577,
                                                    farPlane=124.807, width=48.6035, height=47.2565, cameraPosition=(
            -11.8671, -62.3008, 48.3272), cameraTarget=(1.14358, 22.2084, 3.37354))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.9415,
                                                    farPlane=126.315, cameraPosition=(0.248887, -68.7574, 35.8602),
                                                    cameraUpVector=(0.128723, 0.332071, 0.93443), cameraTarget=(1.10968,
                                                                                                                22.2265,
                                                                                                                3.40842))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.6957,
                                                    farPlane=126.863, cameraPosition=(-11.0921, -63.5083, 46.202),
                                                    cameraUpVector=(0.153633, 0.424155, 0.892462),
                                                    cameraTarget=(1.16555,
                                                                  22.2006, 3.35747))
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.2985,
                                                    farPlane=123.905, width=38.8828, height=37.8052, cameraPosition=(
            -0.701148, -75.6019, 0.29481), cameraTarget=(-0.701148, 21, 0.29481))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=66.3873,
                                                    farPlane=126.352, cameraPosition=(-22.7537, -65.9236, 36.2098),
                                                    cameraUpVector=(-0.00756498, 0.383495, 0.923512), cameraTarget=(
            -0.701148, 21, 0.29481))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=66.0385,
                                                    farPlane=125.553, cameraPosition=(35.7925, -56.0809, 45.5712),
                                                    cameraUpVector=(0.0406089, 0.520641, 0.85281),
                                                    cameraTarget=(-0.84224,
                                                                  20.9763, 0.27225))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.8701,
                                                    farPlane=122.051, cameraPosition=(-0.712879, -59.2087, 54.0344),
                                                    cameraUpVector=(0.0168135, 0.557167, 0.83023),
                                                    cameraTarget=(-0.535141,
                                                                  21.0026, 0.201054))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.1776,
                                                    farPlane=123.913, cameraPosition=(-21.8174, -55.3342, 55.4151),
                                                    cameraUpVector=(0.0569643, 0.57476, 0.816337),
                                                    cameraTarget=(-0.394064,
                                                                  20.9767, 0.191825))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.2772,
                                                    farPlane=124.976, cameraPosition=(-30.4418, -55.213, 51.4135),
                                                    cameraUpVector=(0.0685651, 0.537644, 0.840379), cameraTarget=(
            -0.344113, 20.976, 0.215001))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=67.4515,
                                                    farPlane=124.769, cameraPosition=(-26.9306, -56.3022, 51.7232),
                                                    cameraUpVector=(0.0644599, 0.538014, 0.840468), cameraTarget=(
            -0.361477, 20.9814, 0.21347))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.561,
                                                    farPlane=123.485, cameraPosition=(-15.8269, -56.8612, 55.2824),
                                                    cameraUpVector=(0.0568867, 0.569182, 0.820241), cameraTarget=(
            -0.418286, 20.9843, 0.195261))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#40 #8c60c00 #800 #810 #6444080 #60440002 #2c22068',
        ' #108000 #8 #b88a1081 #60 #c000100 #81050 #41010',
        ' #201808 #40000882 #22000022 #17000110 #480c #10900 ]',), )
    region = p.Set(edges=edges, name='inner brace')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='NO8', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.1369,
                                                    farPlane=121.909, width=38.8828, height=37.8052, cameraPosition=(
            -16.593, -57.6985, 53.885), cameraTarget=(-1.1844, 20.147, -1.20219))
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(width=31.1063, height=30.2442,
                                                    cameraPosition=(-0.0681159, -75.6019, 0.683449), cameraTarget=(
            -0.0681159, 21, 0.683449))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#7f39e7 #39ff3ff8 #2fe3b818 #60ffb87f #76edbff #f07f3c3e #fbee3bf8',
        ' #3018ffda #73fe10c #ffae5b9f #ca1fecf0 #7fbf81a5 #1cd10f6 #e707fe70',
        ' #1bf1386c #c0eb8dee #6700a8e7 #5738097b #2d8d6f1e #171ba1 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.5592,
                                                    farPlane=122.645, width=19.908, height=19.3563, cameraPosition=(
            0.267923, -75.6019, 0.593669), cameraTarget=(0.267923, 21, 0.593669))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#0:6 #400 #48000000 #0:5 #100000 #4000200 #0', ' #1200 ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.5592,
                                                    farPlane=122.645, width=31.1063, height=30.2442, cameraPosition=(
            -0.130783, -75.6019, -0.294158), cameraTarget=(-0.130783, 21,
                                                           -0.294158))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.1087,
                                                    farPlane=120.35, cameraPosition=(-13.2802, -56.5388, 55.8017),
                                                    cameraUpVector=(-0.0411821, 0.590225, 0.806188), cameraTarget=(
            -0.130783, 21, -0.294159))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=74.0949,
                                                    farPlane=115.252, cameraPosition=(4.13585, -40.2166, 74.2252),
                                                    cameraUpVector=(0.0130339, 0.775323, 0.63143),
                                                    cameraTarget=(-0.381784,
                                                                  20.7648, -0.559678))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.5492,
                                                    farPlane=118.798, width=60.7544, height=59.0706, cameraPosition=(
            8.62219, -43.2189, 71.506), cameraTarget=(4.10455, 17.7625, -3.27881))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#18 #c000 #180186 #c000300 #c0000000 #c000000 #0',
        ' #7860000 #80000cc3 #2000 #0 #80003840 #0 #18c00180',
        ' #6002c002 #19000000 #a00100 #80004000 #10300001 #448 ]',), )
    region = p.Set(edges=edges, name='top chord sc')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='top chord sc', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.087,
                                                    farPlane=121.26, width=48.6035, height=47.2565, cameraPosition=(
            3.72506, -40.0468, 74.3884), cameraTarget=(-0.792572, 20.9346,
                                                       -0.396438))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.replace(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(session.views['Bottom'])
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.2985,
                                                    farPlane=123.905, width=38.8828, height=37.8052, cameraPosition=(
            0.541521, -75.6019, 4.03321), cameraTarget=(0.541521, 21, 4.03321))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#ffc1effc #efc7eccf #f87cefef #ffb3efd9 #fefffec9 #ffede7e3 #bef3ee6f',
        ' #ffff99ff #f9f83fff #b8fbfef1 #77fe3bff #ccc0ffff #ff3fffdd #dffcb19f',
        ' #ee2fdffb #7f957ffb #ffffff7e #ffdff7d4 #def7d8ff #3def5f ]',), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.remove(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.5845,
                                                    farPlane=124.921, cameraPosition=(4.40107, -60.953, 55.0306),
                                                    cameraUpVector=(0.000461414, 0.528351, 0.849026))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=77.2225,
                                                    farPlane=122.125, cameraPosition=(-4.52664, -42.8884, 76.4441),
                                                    cameraUpVector=(-0.00861507, 0.745965, 0.665929), cameraTarget=(
            0.347078, 21.3934, 4.49959))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=78.7745,
                                                    farPlane=121.001, cameraPosition=(-1.28114, -38.3709, 80.3872),
                                                    cameraUpVector=(-0.0147558, 0.784517, 0.619932), cameraTarget=(
            0.447099, 21.5326, 4.62111))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=74.3425,
                                                    farPlane=125.434, width=48.6035, height=47.2565, cameraPosition=(
            -4.65926, -47.2321, 73.3041), cameraTarget=(-2.93102, 12.6714,
                                                        -2.46195))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#0 #10000300 #820010 #c0000 #0:3 #6000 #60000',
        ' #600000c #0 #3160000 #0 #10c40 #11000000 #0', ' #1 #0 #2600 #1000 ]',
    ), )
    region = p.Set(edges=edges, name='bottom chord sc')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='bottom chord sc', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=74.3425,
                                                    farPlane=125.434, width=75.943, height=73.8383, cameraPosition=(
            -0.34115, -40.2735, 78.9044), cameraTarget=(1.38709, 19.63, 3.13831))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=71.0053,
                                                    farPlane=128.619, cameraPosition=(-7.97504, -36.8382, 80.9718),
                                                    cameraUpVector=(0.00895987, 0.808083, 0.589001),
                                                    cameraTarget=(1.13595,
                                                                  19.743, 3.20632))
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)
    session.viewports['Viewport: 1'].partDisplay.displayGroup.replace(leaf=leaf)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=61.8621,
                                                    farPlane=135.225, cameraPosition=(-17.7263, -65.5598, 45.0056),
                                                    cameraUpVector=(0.0222817, 0.445123, 0.895192),
                                                    cameraTarget=(0.926211,
                                                                  19.1252, 2.43273))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=74.6694,
                                                    farPlane=122.418, width=15.9264, height=15.485, cameraPosition=(
            -30.4892, -60.2265, 50.0227), cameraTarget=(-11.8366, 24.4585,
                                                        7.44982))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.0123,
                                                    farPlane=126.371, cameraPosition=(-48.3315, -55.793, 47.0298),
                                                    cameraUpVector=(0.0505966, 0.423505, 0.90448),
                                                    cameraTarget=(-12.1882,
                                                                  24.5459, 7.39084))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.1969,
                                                    farPlane=128.187, width=19.908, height=19.3563, cameraPosition=(
            -47.8565, -56.8596, 45.3011), cameraTarget=(-11.7132, 23.4793,
                                                        5.66215))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=78.7029,
                                                    farPlane=128.795, cameraPosition=(-65.2581, -39.4451, 56.0076),
                                                    cameraUpVector=(0.118609, 0.549964, 0.826724), cameraTarget=(-12.42,
                                                                                                                 24.1866,
                                                                                                                 6.09704))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=77.6943,
                                                    farPlane=129.803, width=31.1063, height=30.2442, cameraPosition=(
            -60.3212, -40.4224, 59.9881), cameraTarget=(-7.48307, 23.2093,
                                                        10.0776))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=77.7205,
                                                    farPlane=128.506, cameraPosition=(-32.7971, -45.5859, 73.5203),
                                                    cameraUpVector=(0.142261, 0.636111, 0.75837),
                                                    cameraTarget=(-5.58702,
                                                                  22.8536, 11.0098))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=78.9812,
                                                    farPlane=127.245, width=31.1063, height=30.2442, cameraPosition=(
            -22.8904, -46.937, 76.3533), cameraTarget=(4.3197, 21.5025, 13.8428))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=82.297,
                                                    farPlane=123.578, cameraPosition=(-67.6581, 1.72432, 77.373),
                                                    cameraUpVector=(0.519775, 0.447769, 0.727555),
                                                    cameraTarget=(1.49274,
                                                                  24.5753, 13.9072))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.5576,
                                                    farPlane=122.317, width=31.1063, height=30.2442, cameraPosition=(
            -73.6329, 8.71259, 73.3792), cameraTarget=(-4.48201, 31.5636, 9.91339))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=80.9641,
                                                    farPlane=123.79, cameraPosition=(-71.8057, 3.89906, 73.4409),
                                                    cameraUpVector=(0.512982, 0.442826, 0.735361),
                                                    cameraTarget=(-4.36955,
                                                                  31.2673, 9.91719))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.2599,
                                                    farPlane=131.494, width=86.4066, height=84.0119, cameraPosition=(
            -73.3932, 2.99128, 71.3645), cameraTarget=(-5.95704, 30.3595, 7.84083))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=61.6917,
                                                    farPlane=136.333, cameraPosition=(-66.9681, -24.315, 59.5374),
                                                    cameraUpVector=(0.344785, 0.426048, 0.836425),
                                                    cameraTarget=(-5.5946,
                                                                  28.8192, 7.17365))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.0292,
                                                    farPlane=125.995, width=44.2402, height=43.0141, cameraPosition=(
            -64.2645, -38.4295, 48.3841), cameraTarget=(-2.89098, 14.7047,
                                                        -3.97968))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=71.9015,
                                                    farPlane=121.762, cameraPosition=(-50.2017, -26.8823, 69.3887),
                                                    cameraUpVector=(0.490406, 0.577542, 0.652646),
                                                    cameraTarget=(-2.54864,
                                                                  14.9858, -3.46835))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#0 #80000014 #4000001 #90000000 #6 #1b00 #19100',
        ' #80000600 #200 #1000000 #1c000 #20000000 #2 #20000000',
        ' #0 #80000 #4000 #8 #1000000 ]',), )
    region = p.Set(edges=edges, name='cc1-11-chord')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='cc1-11-chord', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.3099,
                                                    farPlane=121.353, width=52.7808, height=51.3181, cameraPosition=(
            -52.5209, -25.2206, 68.8267), cameraTarget=(-4.86783, 16.6475,
                                                        -4.03032))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.138,
                                                    farPlane=126.782, cameraPosition=(-58.3094, -32.507, 59.6889),
                                                    cameraUpVector=(0.436698, 0.497478, 0.749541),
                                                    cameraTarget=(-4.88157,
                                                                  16.6302, -4.05201))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#0 #8 #8000000 #60000000 #1 #2400 #e00',
                                        ' #78000900 #100 #0 #160000 #0 #4 #6100000',
                                        ' #e000200 #800000 #b200 #100000 ]',), )
    region = p.Set(edges=edges, name='Set-10')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='cc1-11-brace', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.1379,
                                                    width=65.976, height=64.1476, cameraPosition=(-57.6452, -26.3387,
                                                                                                  65.0007),
                                                    cameraTarget=(-4.2174, 22.7985, 1.25977))
    session.viewports['Viewport: 1'].view.setValues(session.views['Top'])
    session.viewports['Viewport: 1'].view.setValues(session.views['Back'])
    session.viewports['Viewport: 1'].view.setValues(width=56.6963, height=55.125,
                                                    cameraPosition=(-0.620762, 22.5951, -95.2519),
                                                    cameraTarget=(-0.620762,
                                                                  22.5951, 1.35))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=62.4546,
                                                    farPlane=127.826, cameraPosition=(32.2344, -53.147, -48.8058),
                                                    cameraUpVector=(-0.0911772, 0.522019, -0.848047), cameraTarget=(
            -0.620763, 22.5951, 1.35))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.1684,
                                                    farPlane=119.367, cameraPosition=(-4.70299, -32.6028, 78.7517),
                                                    cameraUpVector=(-0.303169, -0.775374, -0.55397), cameraTarget=(
            -0.0532365, 22.2795, -0.609861))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.8395,
                                                    farPlane=115.445, cameraPosition=(47.3677, 33.0086, 81.8165),
                                                    cameraUpVector=(-0.438642, -0.813963, 0.380864), cameraTarget=(
            -1.34271, 20.6547, -0.685758))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=61.1165,
                                                    farPlane=127.083, cameraPosition=(48.5382, 86.8995, 47.7837),
                                                    cameraUpVector=(-0.0783441, -0.534187, 0.841728), cameraTarget=(
            -1.37329, 19.2466, 0.203481))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=66.0699,
                                                    farPlane=122.214, cameraPosition=(26.6857, 75.5858, 73.2595),
                                                    cameraUpVector=(0.00436591, -0.796933, 0.604052), cameraTarget=(
            -0.792213, 19.5474, -0.473942))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.7941,
                                                    farPlane=114.49, width=23.2228, height=22.5792, cameraPosition=(
            29.2278, 82.4409, 67.1022), cameraTarget=(1.74992, 26.4025, -6.63126))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=74.0561,
                                                    farPlane=113.307, cameraPosition=(11.7204, 84.6966, 69.6972),
                                                    cameraUpVector=(0.0968734, -0.796742, 0.596504),
                                                    cameraTarget=(2.20743,
                                                                  26.3435, -6.69907))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.8797,
                                                    farPlane=114.484, width=36.2857, height=35.28,
                                                    cameraPosition=(11.1367,
                                                                    82.3176, 71.587),
                                                    cameraTarget=(1.62375, 23.9645, -4.80923))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=69.25,
                                                    farPlane=119.732, cameraPosition=(34.1276, 83.0601, 64.2197),
                                                    cameraUpVector=(-0.0614486, -0.742159, 0.667401), cameraTarget=(
            0.907057, 23.9413, -4.57957))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.7635,
                                                    farPlane=117.133, cameraPosition=(-4.46333, 88.5369, 66.8099),
                                                    cameraUpVector=(0.097109, -0.733417, 0.672807),
                                                    cameraTarget=(1.76922,
                                                                  23.8189, -4.63744))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.4106,
                                                    farPlane=114.486, width=29.0285, height=28.224, cameraPosition=(
            -6.94065, 89.186, 66.0058), cameraTarget=(-0.7081, 24.468, -5.44153))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.3355,
                                                    farPlane=115.24, cameraPosition=(32.2894, 77.0476, 68.2521),
                                                    cameraUpVector=(-0.214202, -0.747676, 0.628569), cameraTarget=(
            -1.81632, 24.8109, -5.50499))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=57.0695,
                                                    farPlane=128.506, width=118.118, height=114.844, cameraPosition=(
            38.0184, 61.4352, 76.6601), cameraTarget=(3.91263, 9.19855, 2.90299))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=45.1928,
                                                    farPlane=137.747, cameraPosition=(70.3862, 71.477, 33.0572),
                                                    cameraUpVector=(-0.156272, -0.261843, 0.952375),
                                                    cameraTarget=(2.58202,
                                                                  8.78574, 4.69547))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=59.8159,
                                                    farPlane=135.831, cameraPosition=(86.6079, 25.3599, 47.7163),
                                                    cameraUpVector=(-0.437239, -0.132965, 0.889462),
                                                    cameraTarget=(1.67187,
                                                                  11.3732, 3.87299))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=62.299,
                                                    farPlane=148.63, cameraPosition=(44.1554, -49.761, 66.2186),
                                                    cameraUpVector=(-0.752875, 0.128167, 0.645563),
                                                    cameraTarget=(1.1419,
                                                                  10.4354, 4.10397))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=60.7955,
                                                    farPlane=153.913, cameraPosition=(39.1287, -71.3091, 40.2253),
                                                    cameraUpVector=(-0.608291, 0.0858393, 0.789059), cameraTarget=(
            0.719494, 8.62466, 1.9197))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=61.363,
                                                    farPlane=156.975, cameraPosition=(34.3479, -82.7823, 2.893),
                                                    cameraUpVector=(-0.414381, -0.109417, 0.903502), cameraTarget=(
            0.240677, 7.47557, -1.81929))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=63.4332,
                                                    farPlane=155.641, cameraPosition=(59.3349, -57.2819, -47.4832),
                                                    cameraUpVector=(0.051994, -0.474352, 0.878799),
                                                    cameraTarget=(3.11697,
                                                                  10.411, -7.61817))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.2816,
                                                    farPlane=154.471, cameraPosition=(12.4313, -61.0945, -74.5528),
                                                    cameraUpVector=(0.203464, -0.629861, 0.749585),
                                                    cameraTarget=(-2.42175,
                                                                  9.96079, -10.8148))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=66.5514,
                                                    farPlane=158.475, cameraPosition=(-15.5455, -89.3183, -15.2138),
                                                    cameraUpVector=(0.0333843, -0.13545, 0.990222),
                                                    cameraTarget=(-6.34874,
                                                                  5.99913, -2.48562))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.9034,
                                                    farPlane=161.653, cameraPosition=(-46.6408, -82.4919, -7.85855),
                                                    cameraUpVector=(0.0979489, -0.110208, 0.98907),
                                                    cameraTarget=(-10.7461,
                                                                  6.96449, -1.44547))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=72.1675,
                                                    farPlane=159.265, cameraPosition=(-94.5731, -29.5774, 45.0498),
                                                    cameraUpVector=(0.330846, 0.223751, 0.916775),
                                                    cameraTarget=(-17.9819,
                                                                  14.9524, 6.54151))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.7328,
                                                    farPlane=152.883, cameraPosition=(-62.7012, -17.3279, 89.016),
                                                    cameraUpVector=(0.724753, 0.301902, 0.619345),
                                                    cameraTarget=(-12.7173,
                                                                  16.9758, 13.8039))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.612,
                                                    farPlane=156.078, cameraPosition=(-49.3957, 80.4706, 87.7246),
                                                    cameraUpVector=(0.915148, 0.1813, 0.360047), cameraTarget=(-10.6564,
                                                                                                               32.1241,
                                                                                                               13.6039))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=70.3605,
                                                    farPlane=164.43, cameraPosition=(-37.765, 128.203, 30.9778),
                                                    cameraUpVector=(0.876706, 0.149858, 0.457088),
                                                    cameraTarget=(-8.72444,
                                                                  40.0528, 4.17779))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=73.6277,
                                                    farPlane=158.238, cameraPosition=(-20.7662, 108.557, 74.5327),
                                                    cameraUpVector=(0.91436, -0.139556, 0.380092),
                                                    cameraTarget=(-5.71363,
                                                                  36.5731, 11.8922))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=87.7593,
                                                    farPlane=144.107, width=60.4762, height=58.8001, cameraPosition=(
            -15.6398, 109.622, 74.5406), cameraTarget=(-0.587254, 37.6382,
                                                       11.9001))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=84.4951,
                                                    farPlane=143.158, cameraPosition=(-82.3558, 65.3646, 66.2819),
                                                    cameraUpVector=(0.627966, 0.0284084, 0.777722),
                                                    cameraTarget=(-11.7114,
                                                                  30.2587, 10.5231))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=84.1191,
                                                    farPlane=143.283, cameraPosition=(-106.453, 56.0502, 20.7781),
                                                    cameraUpVector=(0.168369, -0.0562732, 0.984116), cameraTarget=(
            -15.3577, 28.8493, 3.63767))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=86.4728,
                                                    farPlane=139.868, cameraPosition=(-102.214, 41.2288, 45.6309),
                                                    cameraUpVector=(0.401139, 0.00370996, 0.91601),
                                                    cameraTarget=(-14.7202,
                                                                  26.6205, 7.37503))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=(
        '[#80208201 #2200003 #93000240 #82 #31112410 #3804008 #4100083',
        ' #400001 #54408020 #8020 #a0800002 #400 #82a001 #200002',
        ' #80000111 #5200 #8010000 #a10000 #a0000160 #a0 ]',), )
    region = p.Set(edges=edges, name='XG')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.SectionAssignment(region=region, sectionName='XG', offset=0.0,
                        offsetType=MIDDLE_SURFACE, offsetField='',
                        thicknessAssignment=FROM_SECTION)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=82.517,
                                                    farPlane=148.442, cameraPosition=(28.9181, 115.696, 60.8193),
                                                    cameraUpVector=(0.155833, -0.574983, 0.803188),
                                                    cameraTarget=(4.47743,
                                                                  37.5225, 9.5986))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=87.4423,
                                                    farPlane=142.043, cameraPosition=(17.5388, 88.9666, 92.1409),
                                                    cameraUpVector=(0.178299, -0.814172, 0.55257),
                                                    cameraTarget=(2.61731,
                                                                  33.1531, 14.7186))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=94.1327,
                                                    farPlane=133.905, cameraPosition=(-15.1144, 48.8266, 110.9),
                                                    cameraUpVector=(0.207034, -0.945517, 0.251266),
                                                    cameraTarget=(-2.54499,
                                                                  26.8072, 17.6843))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=103.818,
                                                    farPlane=124.22, width=10.1462, height=9.86503, cameraPosition=(
            -16.0811, 31.2871, 114.913), cameraTarget=(-3.5117, 9.26771, 21.6971))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=101.181,
                                                    farPlane=122.908, cameraPosition=(3.6622, 43.4495, 112.226),
                                                    cameraUpVector=(0.27581, -0.90961, 0.310707),
                                                    cameraTarget=(-0.495888,
                                                                  11.1255, 21.2866))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=87.9189,
                                                    farPlane=136.171, width=94.4941, height=91.8753, cameraPosition=(
            6.31043, 46.1305, 111.152), cameraTarget=(2.15235, 13.8065, 20.2126))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=68.6333,
                                                    farPlane=145.956, cameraPosition=(47.8016, 82.0636, 76.5135),
                                                    cameraUpVector=(0.127455, -0.729121, 0.672412),
                                                    cameraTarget=(7.87077,
                                                                  18.7589, 15.4386))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.3022,
                                                    farPlane=146.328, cameraPosition=(51.1503, 90.4472, 63.8521),
                                                    cameraUpVector=(0.12037, -0.617776, 0.777087),
                                                    cameraTarget=(8.20447,
                                                                  19.5943, 14.1769))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=65.303,
                                                    farPlane=144.002, cameraPosition=(27.3067, 97.7649, 68.1428),
                                                    cameraUpVector=(0.118867, -0.586336, 0.801299),
                                                    cameraTarget=(6.12865,
                                                                  20.2314, 14.5504))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=75.0423,
                                                    farPlane=134.263, width=139.38, height=58.8002, cameraPosition=(
            18.2876, 102.949, 64.2066), cameraTarget=(-2.89047, 25.4157, 10.6142))
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:19 #3fffff ]',), )
    region = p.Set(edges=edges, name='Set-12')
    p = mdb.models[model_name].parts['span-24-6-many']
    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(2.0, 1.0,
                                                                         -1.0))
    session.viewports['Viewport: 1'].partDisplay.setValues(renderBeamProfiles=ON)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=82.7551,
                                                    farPlane=126.545, width=23.3841, height=9.86504, cameraPosition=(
            22.6609, 110.259, 51.9036), cameraTarget=(1.48282, 32.7252, -1.68882))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=81.6924,
                                                    farPlane=125.969, cameraPosition=(-43.8315, 97.5415, 56.8934),
                                                    cameraUpVector=(0.185973, -0.584789, 0.789579),
                                                    cameraTarget=(-3.63017,
                                                                  31.7473, -1.30512))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=76.1385,
                                                    farPlane=131.523, width=111.504, height=47.0402, cameraPosition=(
            -33.6176, 98.6811, 62.6604), cameraTarget=(6.58372, 32.8869, 4.46187))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.3349,
                                                    farPlane=138.768, cameraPosition=(52.976, 88.7884, 72.2116),
                                                    cameraUpVector=(-0.421647, -0.552019, 0.719367),
                                                    cameraTarget=(12.6117,
                                                                  32.1983, 5.12676))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.0495,
                                                    farPlane=147.953, cameraPosition=(78.5435, 102.421, 26.097),
                                                    cameraUpVector=(-0.107627, -0.277695, 0.954621),
                                                    cameraTarget=(15.9383,
                                                                  33.9721, -0.873194))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=86.4366,
                                                    farPlane=144.566, width=89.2034, height=37.6321, cameraPosition=(
            77.8224, 101.595, 29.8675), cameraTarget=(15.2172, 33.146, 2.89733))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=84.4633,
                                                    farPlane=142.231, cameraPosition=(28.0182, 119.028, 51.0821),
                                                    cameraUpVector=(-0.231609, -0.415302, 0.879706),
                                                    cameraTarget=(7.06809,
                                                                  35.9984, 6.3685))
    session.viewports['Viewport: 1'].view.setProjection(projection=PARALLEL)
    session.viewports['Viewport: 1'].view.setProjection(projection=PERSPECTIVE)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=87.6499,
                                                    farPlane=137.237, width=80.9361, height=34.1444, cameraPosition=(
            17.9286, 105.94, 72.9515), cameraUpVector=(-0.141009, -0.640872,
                                                       0.754585), cameraTarget=(5.57758, 34.065, 9.5992))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=88.3541,
                                                    farPlane=136.532, width=87.0254, height=36.7133,
                                                    viewOffsetX=-5.48089,
                                                    viewOffsetY=4.31368)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=89.2117,
                                                    farPlane=144.683, width=87.8702, height=37.0697, cameraPosition=(
            70.4948, 93.3766, 60.5341), cameraUpVector=(-0.322167, -0.442492,
                                                        0.836905), cameraTarget=(15.2016, 34.149, 7.93342),
                                                    viewOffsetX=-5.53409, viewOffsetY=4.35555)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=87.8002,
                                                    farPlane=145.117, width=86.4799, height=36.4832, cameraPosition=(
            67.8053, 98.886, 55.5413), cameraUpVector=(-0.323808, -0.389528,
                                                       0.862216), cameraTarget=(14.648, 34.7619, 6.60781),
                                                    viewOffsetX=-5.44653, viewOffsetY=4.28664)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=101.178,
                                                    farPlane=141.387, width=99.6567, height=42.0421, cameraPosition=(
            95.3321, 39.7938, 74.2256), cameraUpVector=(-0.511516, -0.487662,
                                                        0.707487), cameraTarget=(22.3819, 26.4897, 12.3115),
                                                    viewOffsetX=-6.27641, viewOffsetY=4.93979)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=101.836,
                                                    farPlane=144.686, width=100.305, height=42.3157, cameraPosition=(
            91.5864, 1.6638, 81.8059), cameraUpVector=(-0.660942, -0.509056,
                                                       0.551377), cameraTarget=(23.2757, 18.7137, 15.6618),
                                                    viewOffsetX=-6.31725, viewOffsetY=4.97194)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=99.4582,
                                                    farPlane=149.652, width=97.9633, height=41.3277, cameraPosition=(
            72.4462, -31.4572, 88.2886), cameraUpVector=(-0.831074, -0.421415,
                                                         0.36294), cameraTarget=(20.5757, 10.8475, 18.6334),
                                                    viewOffsetX=-6.16975, viewOffsetY=4.85585)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=103.806,
                                                    farPlane=139.776, width=102.246, height=43.1343, cameraPosition=(
            78.1769, 14.195, 94.8751), cameraUpVector=(-0.788891, -0.275393,
                                                       0.549372), cameraTarget=(21.4463, 21.3887, 17.0165),
                                                    viewOffsetX=-6.43945, viewOffsetY=5.06812)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=99.7044,
                                                    farPlane=147.619, width=98.206, height=41.4301, cameraPosition=(
            56.3522, -33.529, 97.4045), cameraUpVector=(-0.920885, -0.157236,
                                                        0.356717), cameraTarget=(18.8588, 10.4689, 20.0062),
                                                    viewOffsetX=-6.18501, viewOffsetY=4.86787)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=101.442,
                                                    farPlane=145.352, width=99.9172, height=42.1519, cameraPosition=(
            90.7243, -6.00377, 81.4152), cameraUpVector=(-0.706182, 0.12717,
                                                         0.696517), cameraTarget=(26.6299, 15.689, 12.4704),
                                                    viewOffsetX=-6.29277, viewOffsetY=4.95268)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=98.6454,
                                                    farPlane=152.568, width=97.1627, height=40.9899, cameraPosition=(
            99.1818, -32.1423, 58.4933), cameraUpVector=(-0.535696, 0.121696,
                                                         0.835596), cameraTarget=(28.7319, 9.60423, 7.248),
                                                    viewOffsetX=-6.11929, viewOffsetY=4.81614)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=98.6892,
                                                    farPlane=152.048, width=97.2058, height=41.0081, cameraPosition=(
            91.2222, -36.0626, 66.789), cameraUpVector=(-0.612848, 0.119605,
                                                        0.781096), cameraTarget=(27.2949, 8.61864, 9.78952),
                                                    viewOffsetX=-6.12201, viewOffsetY=4.81828)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=102.516,
                                                    farPlane=142.798, width=100.975, height=42.5984, cameraPosition=(
            71.0746, -5.69798, 98.2947), cameraUpVector=(-0.835743, 0.117922,
                                                         0.536309), cameraTarget=(22.4508, 16.124, 17.7244),
                                                    viewOffsetX=-6.35942, viewOffsetY=5.00513)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=107.559,
                                                    farPlane=137.755, width=46.287, height=19.5271,
                                                    viewOffsetX=-2.62005,
                                                    viewOffsetY=6.90917)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=99.9087,
                                                    farPlane=144.39, width=42.9948, height=18.1382, cameraPosition=(
            68.0376, 89.8226, 76.6388), cameraUpVector=(-0.787791, -0.0446315,
                                                        0.614323), cameraTarget=(20.1319, 37.1168, 11.3763),
                                                    viewOffsetX=-2.43369, viewOffsetY=6.41775)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=100.15,
                                                    farPlane=144.149, width=43.0989, height=18.1821, cameraPosition=(
            66.5969, 91.32, 76.487), cameraUpVector=(-0.673511, -0.249423,
                                                     0.695825), cameraTarget=(18.6912, 38.6142, 11.2245),
                                                    viewOffsetX=-2.43958, viewOffsetY=6.43327)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=99.6572,
                                                    farPlane=145.331, width=42.8868, height=18.0926, cameraPosition=(
            79.0633, 88.9205, 66.5232), cameraUpVector=(-0.640144, -0.13113,
                                                        0.756981), cameraTarget=(21.2478, 37.4631, 8.71703),
                                                    viewOffsetX=-2.42758, viewOffsetY=6.40161)
    session.viewports['Viewport: 1'].partDisplay.setValues(renderBeamProfiles=OFF)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=97.8004,
                                                    farPlane=147.193, width=82.2026, height=34.6787,
                                                    viewOffsetX=-3.07735,
                                                    viewOffsetY=6.90326)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=94.1417,
                                                    farPlane=147.629, width=79.1274, height=33.3814, cameraPosition=(
            5.29114, 131.079, 51.7035), cameraUpVector=(0.147927, -0.476495,
                                                        0.866643), cameraTarget=(-1.25537, 46.1547, 6.12793),
                                                    viewOffsetX=-2.96223, viewOffsetY=6.64501)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=91.1202,
                                                    farPlane=152.134, width=76.5878, height=32.31,
                                                    cameraPosition=(49.3898,
                                                                    130.896, 19.9141),
                                                    cameraUpVector=(0.164581, -0.305427, 0.937885),
                                                    cameraTarget=(7.79109, 46.1075, -0.39807), viewOffsetX=-2.86716,
                                                    viewOffsetY=6.43173)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=92.9792,
                                                    farPlane=151.191, width=78.1503, height=32.9692, cameraPosition=(
            71.6697, 113.067, 38.2384), cameraUpVector=(0.0502743, -0.4782,
                                                        0.876811), cameraTarget=(13.4316, 44.0474, 3.93514),
                                                    viewOffsetX=-2.92565, viewOffsetY=6.56294)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=94.4812,
                                                    farPlane=148.465, width=79.4128, height=33.5018, cameraPosition=(
            33.8083, 120.48, 62.9139), cameraUpVector=(-0.178685, -0.534511,
                                                       0.826057), cameraTarget=(6.64459, 45.4383, 8.48105),
                                                    viewOffsetX=-2.97291, viewOffsetY=6.66896)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=93.2238,
                                                    farPlane=150.609, width=78.356, height=33.0559, cameraPosition=(
            63.0778, 114.005, 49.3791), cameraUpVector=(-0.0893596, -0.478951,
                                                        0.873282), cameraTarget=(12.3222, 44.2386, 5.92173),
                                                    viewOffsetX=-2.93335, viewOffsetY=6.58021)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=95.3382,
                                                    farPlane=147.931, width=80.1332, height=33.8057, cameraPosition=(
            39.6439, 113.113, 70.7033), cameraUpVector=(-0.157091, -0.607592,
                                                        0.778559), cameraTarget=(7.66218, 44.4939, 10.6987),
                                                    viewOffsetX=-2.99988, viewOffsetY=6.72946)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=97.2059,
                                                    farPlane=145.065, width=81.7031, height=34.468,
                                                    cameraPosition=(3.6817,
                                                                    117.921, 74.4097),
                                                    cameraUpVector=(-0.14797, -0.644203, 0.750405),
                                                    cameraTarget=(0.0232615, 45.034, 11.116), viewOffsetX=-3.05865,
                                                    viewOffsetY=6.86129)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=94.3378,
                                                    farPlane=148.987, width=79.2924, height=33.451, cameraPosition=(
            45.3572, 114.355, 65.3885), cameraUpVector=(-0.145021, -0.571168,
                                                        0.807921), cameraTarget=(8.67046, 44.6203, 9.50309),
                                                    viewOffsetX=-2.9684, viewOffsetY=6.65884)
    session.viewports['Viewport: 1'].view.setValues(width=99.3187, height=41.8995,
                                                    viewOffsetX=-4.47785, viewOffsetY=6.57001)
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models[model_name].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[model_name].parts['span-24-6-many']
    a.Instance(name='span-24-6-many-1', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)


    mdb.models[model_name].StaticStep(name='Step-1', previous='Initial',
        maxNumInc=100000, initialInc=0.1, minInc=1e-08, nlgeom=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    mdb.models[model_name].ImplicitDynamicsStep(name='Step-2', previous='Step-1',
        timePeriod=360.0, maxNumInc=1000000000, application=QUASI_STATIC,
        initialInc=0.1, minInc=1e-08, nohaf=OFF, amplitude=RAMP, alpha=DEFAULT,
        initialConditions=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-2')
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'PE', 'U', 'RF', 'CF'), timeInterval=1.0)
    mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValuesInStep(
        stepName='Step-2', timeInterval=1.0)

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#9000 #8000000 #280062 ]', ), )
    a.Set(vertices=verts1, name='boundary-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0 #8 #0:2 #ac000004 #0:2 #408 ]', ), )
    a.Set(vertices=verts1, name='boundart-R')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    a = mdb.models[model_name].rootAssembly
    region = a.sets['boundary-L']
    mdb.models[model_name].DisplacementBC(name='BC-1', createStepName='Initial',
        region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET, distributionType=UNIFORM, fieldName='',
        localCsys=None)

    a = mdb.models[model_name].rootAssembly
    region = a.sets['boundart-R']
    mdb.models[model_name].DisplacementBC(name='BC-2', createStepName='Initial',
        region=region, u1=SET, u2=SET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET,
        amplitude=UNSET, distributionType=UNIFORM, fieldName='',
        localCsys=None)


    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#8 #10c00280 #0:3 #d00000 ]', ), )
    a.Set(vertices=verts1, name='load-B1_L')
    mdb.models[model_name].rootAssembly.sets.changeKey(fromName='load-B1_L',
        toName='load-B1-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0 #20 #0:3 #80000d04 #40 #800000 ]',
        ), )
    a.Set(vertices=verts1, name='load-B1-R')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#160 #2002800 #1 #0:2 #4000 #20 ]', ),
        )
    a.Set(vertices=verts1, name='load-B2-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0 #1 #0:3 #a001060 #0 #3000000 ]', ),
        )
    a.Set(vertices=verts1, name='load-B2-R')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#2600 #60050000 #0:4 #410 ]', ), )
    a.Set(vertices=verts1, name='load-B3-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#4000 #180000 #8 #b8 #0:2 #800 ]', ), )
    a.Set(vertices=verts1, name='load-B4-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#200000 #0 #10 #d45 #0:2 #20000 ]', ),
        )
    a.Set(vertices=verts1, name='load-B5-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0:2 #500 #1000 #b000 #0 #4050000 ]',
        ), )
    a.Set(vertices=verts1, name='load-B6-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#0:2 #1400800 #2000 #800 #0 #81280000 ]', ), )
    a.Set(vertices=verts1, name='load-B7-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0:2 #4066000 #0 #540 #0:2 #8000 ]', ),
        )
    a.Set(vertices=verts1, name='load-B8-L')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#80000000 #0:3 #40000000 #30000000 #206 #1000 ]', ), )
    a.Set(vertices=verts1, name='load-B3-R')
    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#40000000 #0:3 #11400008 #0 #1000 #120000 ]', ), )
    a.Set(vertices=verts1, name='load-B4-R')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#3fc00000 ]', ), )
    a.Set(vertices=verts1, name='load-B5-R')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0:3 #60000000 #12 #0 #78000000 ]', ),
        )
    a.Set(vertices=verts1, name='load-B6-R')
    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0:3 #e00000 #20 #0:2 #10002044 ]', ),
        )
    a.Set(vertices=verts1, name='load-B7-R')

    a = mdb.models[model_name].rootAssembly
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#0:3 #2018000 #0:3 #ac000080 ]', ), )
    a.Set(vertices=verts1, name='load-B8-R')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #40000000 #0:8 #1400 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S1-C')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #40000000 #0:8 #3400 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S1-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#4 #20000000 #8 #0:7 #2000 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S1-B')

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#19 #9000c31c #c9a01d7 #0:4 #ffc66f00 #d0060fe3 #700200c',
        ' #17c000 #0:4 #19880000 #8a0f301 #8 #0 #14c0 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#200 #200002 #12000200 #0:5 #408000 #8020 #0:6',
        ' #10000 #0 #80000000 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#80208000 #2000001 #81000000 #fc0c0382 #f1112417 #f807b08 #4119f83',
        ' #1 #4000000 #0 #a0800002 #a3163c40 #82a001 #1ef10dc2',
        ' #ff02c113 #5200 #0 #80a14000 #31302761 #28 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#4 #20000000 #8 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S1-B')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:2 #400000 #0:7 #18400000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S1-C')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#7400 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S1-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#20000 #0 #40000000 #0 #8 #0:5 #1000000 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S1-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #20 #0:6 #1 #300 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S1-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #20 #0:6 #1 #300 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S1-C')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #4 #0:7 #38 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S1-C')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #3500000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S1-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:3 #20000 #0:7 #600000 #0:5 #20000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S1-C')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#100 #0 #200000 #0:7 #80000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S1-B')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#800 #0 #20008000 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S1-B')

    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#10000 #0:3 #1000 #0:5 #2000000 ]', ),
        )
    a.Set(edges=edges1, name='fire-B4-S1-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #1 #200 #0:7 #80 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S1-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #40 #0:6 #40000000 #4 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S1-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #a00000 #0:7 #40000000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S1-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #18000 #0:7 #800000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S1-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#2 #1000 #20 #0:5 #20000000 #0 #200000', ' #0:5 #80000 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0 #80000 #50000 #0:6 #400000 #4000000 #0:5', ' #100000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:2 #5400 #0:6 #150000 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#c0000 #4000000 #0:10 #4c00 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #800100 #0:7 #200028 #0:4 #400000 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #20 #0:2 #14 #0:4 #2 #100000',
        ' #0:4 #2000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:11 #10014200 #0:2 #40000 #0:2 #8000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #5400 #0:10 #2000 #0:2 #40000 #0',
        ' #4 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S2-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #2c00 #0:7 #5000 #800 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#e0 #70000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #1000000 #2800 #0:6 #2a0000 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #8000000 #0:2 #8c800 #0:7 #1000 ]',
        ), )
    a.Set(edges=edges1, name='fire-B4-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #6200c0 #0:7 #10 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #18 #0:2 #8 #0:5 #d0000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:11 #c008180 #0:2 #10000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:3 #2800 #0 #80000000 #0:8 #1800 #0:2', ' #80000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S2-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #10 #2 #0:6 #60000 #20 #0',
        ' #2000 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #100000 #0:13 #400000 #410 #4 #0',
        ' #4000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:9 #40000000 #101 #0:6 #282 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#100000 #0:9 #20c #0 #400000 #0:4 #1000 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#1800000 #0:3 #8000000 #0 #c0000 #0:6 #1 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:12 #6000000 #4 #580000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:13 #80028 #800000 #0:2 #8000000 #8000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S3-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #1 #4000 #0:4 #80000 #0 #200',
        ' #400 #0:4 #2 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S3-C')

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:12 #6 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:14 #200 #0:2 #100000 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #c #18000001 #0:6 #40 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:9 #800000 #0:6 #820 #10 #0 #a00 ]',
        ), )
    a.Set(edges=edges1, name='fire-B2-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0 #c00000 #0:7 #80000000 #0:6 #40000000 #140 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #2040000 #0:5 #f0 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#400000 #0:3 #4000000 #400000 #2000000 #0:5 #40', ' #0 #20 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #60 #0:5 #1000000 #0 #200048 ]',
        ), )
    a.Set(edges=edges1, name='fire-B6-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:13 #40010 #0:3 #14000000 #10000 #1 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #70000002 #2000 #0:10 #40000000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S3-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #200000 #10000 #0:6 #6000000 #88 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #10000 #1000 #0:6 #a0200000 #0:3 #200000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #800000 #140 #0:6 #90000000 #1 ]',
        ), )
    a.Set(edges=edges1, name='fire-B3-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:15 #2005 #0 #c00 #0 #10 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#1e000000 #0:5 #41000000 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #20000 #0:6 #78000000 #0 #4 ]', ),
        )
    a.Set(edges=edges1, name='fire-B6-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #10 #0:7 #20000 #80 #0:3 #40400080 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #4 #0:11 #20000000 #21000 #28000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S4-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #180000 #0 #20000000 #0:6 #1000006 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S4-B')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #180000 #0 #20000000 #0:6 #1000006 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S4-B')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #8000 #202000 #10 #0:5 #40000000 #0:3', ' #100 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #1000000 #880 #0:6 #26000000 ]',
        ), )
    a.Set(edges=edges1, name='fire-B3-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:15 #1e2 #0 #1000000 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #18e20000 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #c0000 #0:7 #80000000 #0 #c00 #0',
        ' #2000000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #20 #0:7 #3000 #0:4 #4006 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:18 #40818 #110000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S4-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #a0 #0:6 #8000000 #0:6 #100000 ]',
        ), )
    a.Set(edges=edges1, name='fire-B1-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #84000 #0:6 #40000 #400000 ]', ),
        )
    a.Set(edges=edges1, name='fire-B2-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #60 #2000000 #400 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #c #0:7 #20010 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#60000000 #0:14 #18000 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #108000 #0:6 #80000000 #40000000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #c0 #0:7 #4000 #0:4 #800000 ]', ),
        )
    a.Set(edges=edges1, name='fire-B7-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:18 #6080000 #80000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S5-C')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #40 #0:5 #1000 ]', ), )
    a.Set(edges=edges1, name='fire-B1-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #80 #100000 ]', ), )
    a.Set(edges=edges1, name='fire-B2-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #10 #0 #200 ]', ), )
    a.Set(edges=edges1, name='fire-B3-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #2 #0:7 #8 ]', ), )
    a.Set(edges=edges1, name='fire-B4-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #a0000000 ]', ), )
    a.Set(edges=edges1, name='fire-B5-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #210000 ]', ), )
    a.Set(edges=edges1, name='fire-B6-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:13 #1008000 ]', ), )
    a.Set(edges=edges1, name='fire-B7-S5-B')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:18 #8000000 #40000 ]', ), )
    a.Set(edges=edges1, name='fire-B8-S5-B')


    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #400 #0:7 #20000000 ]', ), )
    leaf = dgm.LeafFromGeometry(edgeSeq=edges1)

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #80000300 #c980017 #0:7 #17c000 ]',
        ), )
    a.Set(edges=edges1, name='fire-A1-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#200 #0 #12000000 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#8000 #0:9 #800000 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S1')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:2 #80000000 #0 #2000 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #2 #400 #0:5 #20000000 ]', ), )
    a.Set(edges=edges1, name='fire-A5-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #80 #0:6 #80000000 #0 #1 ]', ), )
    a.Set(edges=edges1, name='fire-A6-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #fc0c0000 #7 #0:6 #a3000800 #6 ]',
        ), )
    a.Set(edges=edges1, name='fire-A7-S1')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#19 #1000c000 #201c0 #0:5 #c0000000 #6002000 ]', ), )
    a.Set(edges=edges1, name='fire-A1-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:2 #200 #0:6 #8000 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #2000000 #0:10 #2000 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #110000 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:12 #28000 ]', ), )
    a.Set(edges=edges1, name='fire-A5-S2')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #2 #0:10 #800000 ]', ), )
    a.Set(edges=edges1, name='fire-A6-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #1 #0:4 #163000 #0 #40 #6002c000',
        ' #0:2 #210000 #0 #8 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:3 #300 #0:7 #163440 #0 #40 #c000',
        ' #0:2 #214000 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S2')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #fe000000 #c3 #100000c #0:5 #800000 #f301', ' #8 #0 #1400 ]', ),
        )
    a.Set(edges=edges1, name='fire-A1-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #fe000000 #e3 #100000c #0:5 #800000 #f301', ' #8 #0 #1400 ]', ),
        )
    a.Set(edges=edges1, name='fire-A1-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:16 #10000 #0 #80000000 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:10 #2 #0:3 #10 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S4')

    del mdb.models[model_name].rootAssembly.sets['fire-A2-S4']

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:10 #2 #0:3 #10 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #800000 #0:7 #2 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#200000 #0:4 #800000 #0:7 #2 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S3')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #1000000 #0:5 #2 #0:3 #10 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S3')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0 #200000 #0:14 #10000 #0 #80000000 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S3')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #10000000 #1000000 #0:6 #800000 ]', ), )
    a.Set(edges=edges1, name='fire-A5-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:4 #20000000 #2000000 #80 ]', ), )
    a.Set(edges=edges1, name='fire-A6-S3')


    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #c0000000 #c000000 #19f00 #0:4 #160000 #0',
        ' #100040 #6e020300 #0:2 #100000 #0 #8 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #c0000000 #c000008 #19e00 #0:6 #100c00 #7f020300',
        ' #0:2 #100000 #0 #8 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #c0000000 #c000008 #19f01 #0:6 #100c00 #7f020300',
        ' #0:2 #100000 #0 #8 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:4 #c0000000 #c000000 #19f01 #0:6 #100c00 #7f020200',
        ' #0:2 #100000 #0 #8 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S3')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #1866000 #60000 #0:6 #19000000 #8800000 #0:2', ' #80 ]', ), )
    a.Set(edges=edges1, name='fire-A1-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:7 #1c66000 #60000 #0:6 #19000000 #8800000 #0:2', ' #80 ]', ), )
    a.Set(edges=edges1, name='fire-A1-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:8 #400000 #20 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:14 #80000000 #1000 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:6 #4100000 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:15 #4200 ]', ), )
    a.Set(edges=edges1, name='fire-A5-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:14 #1 #0:3 #40 ]', ), )
    a.Set(edges=edges1, name='fire-A6-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:13 #c10180 #100 #0:2 #80000000 #302621 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #8 #0:7 #c10180 #100 #0:2 #80000000', ' #302621 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S4')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0 #1c #0:5 #f00 #10000f00 #0:7 #200000', ' #0:2 #40 ]', ), )
    a.Set(edges=edges1, name='fire-A1-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #2 #0:6 #8000 ]', ), )
    a.Set(edges=edges1, name='fire-A2-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0 #1 #0:13 #1000 ]', ), )
    a.Set(edges=edges1, name='fire-A3-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#80000000 #0:5 #4000000 ]', ), )
    a.Set(edges=edges1, name='fire-A4-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:7 #1 #0:10 #20000000 ]', ), )
    a.Set(edges=edges1, name='fire-A5-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#0:5 #4000 #0:12 #100 ]', ), )
    a.Set(edges=edges1, name='fire-A6-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #3f00 #0:7 #26000000 #0:4 #1000000 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #3f00 #0:7 #3e000000 #2 #0:3 #11000000 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S5')
    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #3f00 #0:7 #3e200000 #2 #0:3 #11000000 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0 #1c #0:5 #f00 #10000f00 #0:6 #80000', ' #200000 #0:2 #40 ]', ), )
    a.Set(edges=edges1, name='fire-A1-S5')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#0:5 #3f00 #0:7 #3e200000 #2 #0:3 #11000000 ]', ), )
    a.Set(edges=edges1, name='fire-A7-S5')

    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)

    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B1-L']
    mdb.models[model_name].ConcentratedForce(name='Load-1', createStepName='Step-1',
        region=region, cf3=load_L[0], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B1-R']
    mdb.models[model_name].ConcentratedForce(name='Load-2', createStepName='Step-1',
        region=region, cf3=load_R[0], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B2-L']
    mdb.models[model_name].ConcentratedForce(name='Load-3', createStepName='Step-1',
        region=region, cf3=load_L[1], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B2-R']
    mdb.models[model_name].ConcentratedForce(name='Load-4', createStepName='Step-1',
        region=region, cf3=load_R[1], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B3-L']
    mdb.models[model_name].ConcentratedForce(name='Load-5', createStepName='Step-1',
        region=region, cf3=load_L[2], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B3-R']
    mdb.models[model_name].ConcentratedForce(name='Load-6', createStepName='Step-1',
        region=region, cf3=load_R[2], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B4-L']
    mdb.models[model_name].ConcentratedForce(name='Load-7', createStepName='Step-1',
        region=region, cf3=load_L[3], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B4-R']
    mdb.models[model_name].ConcentratedForce(name='Load-8', createStepName='Step-1',
        region=region, cf3=load_R[3], distributionType=UNIFORM, field='',
        localCsys=None)

    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B5-L']
    mdb.models[model_name].ConcentratedForce(name='Load-9', createStepName='Step-1',
        region=region, cf3=load_L[4], distributionType=UNIFORM, field='',
        localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B5-R']
    mdb.models[model_name].ConcentratedForce(name='Load-10',
        createStepName='Step-1', region=region, cf3=load_R[4],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B6-L']
    mdb.models[model_name].ConcentratedForce(name='Load-11',
        createStepName='Step-1', region=region, cf3=load_L[5],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B6-R']
    mdb.models[model_name].ConcentratedForce(name='Load-12',
        createStepName='Step-1', region=region, cf3=load_R[5],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B7-L']
    mdb.models[model_name].ConcentratedForce(name='Load-13',
        createStepName='Step-1', region=region, cf3=load_L[6],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B7-R']
    mdb.models[model_name].ConcentratedForce(name='Load-14',
        createStepName='Step-1', region=region, cf3=load_R[6],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B8-L']
    mdb.models[model_name].ConcentratedForce(name='Load-15',
        createStepName='Step-1', region=region, cf3=load_L[7],
        distributionType=UNIFORM, field='', localCsys=None)
    a = mdb.models[model_name].rootAssembly
    region = a.sets['load-B8-R']
    mdb.models[model_name].ConcentratedForce(name='Load-16',
        createStepName='Step-1', region=region, cf3=load_R[7],
        distributionType=UNIFORM, field='', localCsys=None)

    mdb.models[model_name].TabularAmplitude(name='steel-field-chord-1', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=chord_temp_tuple_1)

    mdb.models[model_name].TabularAmplitude(name='steel-field-brace-1', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=web_temp_tuple_1)

    mdb.models[model_name].TabularAmplitude(name='steel-field-chord-2', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=chord_temp_tuple_2)

    mdb.models[model_name].TabularAmplitude(name='steel-field-brace-2', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=web_temp_tuple_2)

    mdb.models[model_name].TabularAmplitude(name='steel-field-chord-3', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=chord_temp_tuple_3)

    mdb.models[model_name].TabularAmplitude(name='steel-field-brace-3', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=web_temp_tuple_3)

    mdb.models[model_name].TabularAmplitude(name='steel-field-chord-4', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=chord_temp_tuple_4)

    mdb.models[model_name].TabularAmplitude(name='steel-field-brace-4', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=web_temp_tuple_4)

    mdb.models[model_name].TabularAmplitude(name='steel-field-chord-5', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=chord_temp_tuple_5)

    mdb.models[model_name].TabularAmplitude(name='steel-field-brace-5', timeSpan=STEP,
        smooth=SOLVER_DEFAULT, data=web_temp_tuple_5)

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#80208219 #9220c31f #9f9a03d7 #fc0c0382 #f1112417 #f807f08 #4119f83',
        ' #ffc66f01 #d4468fe3 #700a02c #a097c002 #a3163c40 #82a007 #3ef10dc2',
        ' #ff02c313 #19885200 #8a1f301 #80b14008 #b1302761 #14e8 ]', ), )
    a.Set(edges=edges1, name='whole-XG')
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    set1 = mdb.models[model_name].rootAssembly.sets['whole-XG']
    leaf = dgm.LeafFromSets(sets=(set1, ))

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#7f9e5402 #44181020 #40055420 #3405424 #8800120 #1280d5 #410c4014',
        ' #210024 #2a815010 #40550542 #1d60170d #1069421a #fe704f28 #a422d',
        ' #dc2484 #a676a015 #905e0498 #2846bea7 #424a9080 #2ae016 ]', ), )
    a.Set(edges=edges1, name='whole-chord')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#7f9e7402 #441810a0 #40455420 #3525424 #8800128 #1280d5 #410c4014',
        ' #21006c #2a895010 #40550542 #1d60370d #1069423b #fe704f28 #400a422d',
        ' #dc2484 #a677a015 #905e0498 #2846bea7 #46ca9080 #2ae016 ]', ), )
    a.Set(edges=edges1, name='whole-chord')

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=(
        '[#4109e4 #29c72c40 #2020a808 #a1a859 #66edac0 #f06d0022 #bae22068',
        ' #189092 #130200c #b8aa5a91 #420808f0 #4c808184 #10d10d0 #8104b010',
        ' #211868 #40000dea #67000866 #57080150 #805481e #150b01 ]', ), )
    a.Set(edges=edges1, name='whole-brace')
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    leaf = dgm.Leaf(leafType=DEFAULT_MODEL)

    a = mdb.models[model_name].rootAssembly
    e1 = a.instances['span-24-6-many-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#ffffffff:19 #3fffff ]', ), )
    v1 = a.instances['span-24-6-many-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#ffffffff:8 ]', ), )
    region = a.Set(vertices=verts1, edges=edges1, name='whole')
    mdb.models[model_name].Temperature(name='Predefined Field-1',
        createStepName='Step-1', region=region, distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(20.0,
        ))
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-2')
    mdb.models[model_name].predefinedFields['Predefined Field-1'].resetToInitial(
        stepName='Step-2')

    #赋予温度
    if fire_span_string == "S1":
        if fire_bay_string == "By1":

            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S1']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,
                ), amplitude='steel-field-brace-1')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S5']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-5')

            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S1']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,
                ), amplitude='steel-field-brace-1')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-8',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-9',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-10',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S5']
            mdb.models[model_name].Temperature(name='Predefined Field-11',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-5')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S1-C']
            mdb.models[model_name].Temperature(name='Predefined Field-12',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,
                ), amplitude='steel-field-chord-1')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S2-C']
            mdb.models[model_name].Temperature(name='Predefined Field-13',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-chord-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S3-C']
            mdb.models[model_name].Temperature(name='Predefined Field-14',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-chord-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S4-C']
            mdb.models[model_name].Temperature(name='Predefined Field-15',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-chord-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S5-C']
            mdb.models[model_name].Temperature(name='Predefined Field-16',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-chord-5')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S1-B']
            mdb.models[model_name].Temperature(name='Predefined Field-17',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0,
                ), amplitude='steel-field-brace-1')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S2-B']
            mdb.models[model_name].Temperature(name='Predefined Field-18',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S3-B']
            mdb.models[model_name].Temperature(name='Predefined Field-19',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S4-B']
            mdb.models[model_name].Temperature(name='Predefined Field-20',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-B4-S5-B']
            mdb.models[model_name].Temperature(name='Predefined Field-21',
                createStepName='Step-2', region=region, distributionType=UNIFORM,
                crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                ), amplitude='steel-field-brace-5')
        elif fire_bay_string == "By2":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            B = ["B3","B4","B5"]
            field_num = 7
            #magnitudes = [1,0.84,0.69,0.64,0.61]
            type = ["B","C"]
            type_new = ["steel-field-chord","steel-field-brace"]
            for s_num in range(5):
                for B_num in range(3):
                    for type_num in range(2):
                        field_num = field_num + 1
                        a = mdb.models[model_name].rootAssembly
                        region = a.sets["fire-"+B[B_num]+"-"+S[s_num]+"-"+type[type_num]]
                        if type[type_num] == "C":


                            mdb.models[model_name].Temperature(name="Predefined Field-"+str(field_num),
                            createStepName='Step-2', region=region, distributionType=UNIFORM,
                            crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                            ), amplitude='steel-field-chord-'+str(s_num+1))
                        else:
                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-brace-' + str(s_num+1))
    elif fire_span_string == "S2":
        if fire_bay_string == "By1":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            #magnitudes = [0.84, 1, 0.84, 0.69, 0.64]
            field_num = 7
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for type_num in range(2):
                  field_num = field_num+1
                  a = mdb.models[model_name].rootAssembly
                  region = a.sets["fire-B4" + "-" + S[s_num] + "-" + type[type_num]]
                  if type[type_num] == "C":
                      mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                        createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                         ), amplitude='steel-field-chord-' + str(s_num + 1))
                  else:
                      mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                        createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                         ), amplitude='steel-field-brace-' + str(s_num + 1))
        elif  fire_bay_string == "By2":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            B = ["B3", "B4", "B5"]
            field_num = 7
            #magnitudes = [0.84, 1, 0.84, 0.69, 0.64]
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for B_num in range(3):
                    for type_num in range(2):
                        field_num = field_num + 1
                        a = mdb.models[model_name].rootAssembly
                        region = a.sets["fire-" + B[B_num] + "-" + S[s_num] + "-" + type[type_num]]
                        if type[type_num] == "C":

                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-chord-' + str(s_num+1))
                        else:
                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-brace-' + str(s_num+1))
    elif fire_span_string == "S3":
        if fire_bay_string == "By1":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            #magnitudes = [0.69, 0.84, 1, 0.84, 0.69]
            field_num = 7
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for type_num in range(2):
                    field_num = field_num + 1
                    a = mdb.models[model_name].rootAssembly
                    region = a.sets["fire-B4" + "-" + S[s_num] + "-" + type[type_num]]
                    if type[type_num] == "C":
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-chord-'+str(s_num+1))
                    else:
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-brace-' + str(s_num+1))
        elif fire_bay_string == "By2":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            B = ["B3", "B4", "B5"]
            field_num = 7
            #magnitudes = [0.69, 0.84, 1, 0.84, 0.69]
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for B_num in range(3):
                    for type_num in range(2):
                        field_num = field_num + 1
                        a = mdb.models[model_name].rootAssembly
                        region = a.sets["fire-" + B[B_num] + "-" + S[s_num] + "-" + type[type_num]]
                        if type[type_num] == "C":

                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-chord-'+str(s_num+1))
                        else:
                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-brace-' + str(s_num+1))
    elif fire_span_string == "S4":
        if fire_bay_string == "By1":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            #magnitudes = [0.64, 0.69, 0.84, 1, 0.84]
            field_num = 7
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for type_num in range(2):
                    field_num = field_num + 1
                    a = mdb.models[model_name].rootAssembly
                    region = a.sets["fire-B4" + "-" + S[s_num] + "-" + type[type_num]]
                    if type[type_num] == "C":
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-chord-'+str(s_num+1))
                    else:
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-brace-'+str(s_num+1))
        elif fire_bay_string == "By2":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            B = ["B3", "B4", "B5"]
            field_num = 7
            #magnitudes = [0.64, 0.69, 0.84, 1, 0.84]
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for B_num in range(3):
                    for type_num in range(2):
                        field_num = field_num + 1
                        a = mdb.models[model_name].rootAssembly
                        region = a.sets["fire-" + B[B_num] + "-" + S[s_num] + "-" + type[type_num]]
                        if type[type_num] == "C":

                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-chord-' + str(s_num+1))
                        else:
                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-brace-' + str(s_num+1))
    elif fire_span_string == "S5":
        if fire_bay_string == "By1":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            #magnitudes = [0.61, 0.64, 0.69, 0.84, 1]
            field_num = 7
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for type_num in range(2):
                    field_num = field_num + 1
                    a = mdb.models[model_name].rootAssembly
                    region = a.sets["fire-B4" + "-" + S[s_num] + "-" + type[type_num]]
                    if type[type_num] == "C":
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-chord-'+str(s_num+1))
                    else:
                        mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                          createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                          crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                           ), amplitude='steel-field-brace-' + str(s_num+1))
        elif fire_bay_string == "By2":
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-2',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-3',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A3-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-4',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S2']
            mdb.models[model_name].Temperature(name='Predefined Field-5',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-2')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S3']
            mdb.models[model_name].Temperature(name='Predefined Field-6',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-3')
            a = mdb.models[model_name].rootAssembly
            region = a.sets['fire-A4-S4']
            mdb.models[model_name].Temperature(name='Predefined Field-7',
                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                               ), amplitude='steel-field-brace-4')
            S = ["S1", "S2", "S3", "S4", "S5"]
            B = ["B3", "B4", "B5"]
            field_num = 7
            #magnitudes = [0.61, 0.64, 0.69, 0.84, 1]
            type = ["B", "C"]
            type_new = ["steel-field-chord", "steel-field-brace"]
            for s_num in range(5):
                for B_num in range(3):
                    for type_num in range(2):
                        field_num = field_num + 1
                        a = mdb.models[model_name].rootAssembly
                        region = a.sets["fire-" + B[B_num] + "-" + S[s_num] + "-" + type[type_num]]
                        if type[type_num] == "C":

                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-chord-' + str(s_num+1))
                        else:
                            mdb.models[model_name].Temperature(name="Predefined Field-" + str(field_num),
                                                              createStepName='Step-2', region=region, distributionType=UNIFORM,
                                                              crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1,
                                                                                                                               ), amplitude='steel-field-brace-' + str(s_num+1))


    #赋予温度结束




    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#ffffffff:19 #3fffff ]', ), )
    p.seedEdgeByNumber(edges=pickedEdges, number=10, constraint=FINER)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=99.6349,
        farPlane=143.69, width=14.05, height=6.37279, viewOffsetX=-3.73745,
        viewOffsetY=3.70784)
    p = mdb.models[model_name].parts['span-24-6-many']
    p.generateMesh()
    session.viewports['Viewport: 1'].view.setValues(nearPlane=91.7158,
        farPlane=151.61, width=128.481, height=58.2762, viewOffsetX=13.2519,
        viewOffsetY=9.37381)
    elemType1 = mesh.ElemType(elemCode=B31, elemLibrary=STANDARD)
    p = mdb.models[model_name].parts['span-24-6-many']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:19 #3fffff ]', ), )
    pickedRegions =(edges, )
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, ))
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON,
        engineeringFeatures=ON, mesh=OFF)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=OFF)

    session.viewports['Viewport: 1'].view.setValues(nearPlane=83.9478,
        farPlane=159.378, width=230.286, height=104.18, viewOffsetX=23.5483,
        viewOffsetY=15.6378)
    a1 = mdb.models[model_name].rootAssembly
    a1.regenerate()
    a = mdb.models[model_name].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    mdb.Job(name="basic" + model_name, model=model_name, description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF,
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='',
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1,
        numGPUs=0)
mdb.saveAs(pathName='G:/jinyu/liangce-trapzoid-LSTM/abaqus cae/batch1.cae')
#time.sleep(30)
#for c_t in range(2):
#    mdb.jobs["basic"+"Model-" + str(c_t+1)].submit(consistencyChecking=OFF)
#    mdb.jobs["basic" + "Model-" + str(5-(c_t+1))].submit(consistencyChecking=OFF)
#    time.sleep(180)
#    mdb.jobs["basic"+"Model-" + str(c_t+1)].kill()
#    mdb.jobs["basic" + "Model-" + str(5-(c_t+1))].kill()
#    time.sleep(60)
#mdb.jobs['basic-model'].submit(consistencyChecking=OFF)
mdb.saveAs(pathName='G:/jinyu/liangce-trapzoid-LSTM/abaqus cae/Batch1')
#for c_t in range(25):   #批量计算，建议另一个程序中执行
#    mdb.jobs["basic"+"Model-" + str(c_t+1)].submit(consistencyChecking=OFF)
#    mdb.jobs["basic" + "Model-" + str(51-(c_t + 1))].submit(consistencyChecking=OFF)
#    mdb.jobs["basic" + "Model-" + str(51-(c_t + 1))].waitForCompletion()

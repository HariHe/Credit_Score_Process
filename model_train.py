"""
Created on Fri Mar  9 14:31:42 2018

@author: miao01.he

这是模型训练中常用的一些函数和功能合集
"""


import pandas as pd
import math
from sklearn.model_selection import train_test_split
import collections
import numpy as np
from sklearn import linear_model
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.ensemble import GradientBoostingClassifier
from imp import reload
from sklearn.externals import joblib
from pandas.core.frame import DataFrame
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bk
from sklearn.model_selection import train_test_split
import iv as iv1
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import (OneHotEncoder, LabelEncoder, OneHotEncoder, RobustScaler)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import svm
import time
import csv
import os
# from smote_oversample import Smote
from sklearn.naive_bayes import GaussianNB
import xlsxwriter

from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.utils.multiclass import type_of_target

from collections import Counter


def split_x_y(data,y_name): #暂时没有用
    '''
    加载数据
    param：
        data:DataFrame
             数据集
        y_name:数据集中y变量的列名
    '''
    y = data[y_name]
    x = data.drop([y_name],axis = 1) #axis = 1 表示按列
    return x,y

def woe(data,elements_value_counts_dict,Y_value_counts_dict,i,Y):
    '''
    用于计算离散型变量的WOE值

    :param data: DataFrame
                 包含一列离散型变量及Y变量
    :param elements_value_counts_dict: dict
                 需要计算的变量的value_counts字典
    :param Y_value_counts_dict: dict
                 Y变量的value_counts字典
    :param i: string
                 需要计算的变量名
    :param Y: string
                 Y变量名
    :return: dict
                 map字典。{"元素值1":woe1,"元素值2":woe2,...}
    '''
    mapping_dict = {}
    total_bad_count = Y_value_counts_dict[1]
    total_good_count = Y_value_counts_dict[0]
    if total_bad_count == 0:total_bad_count+=1
    if total_good_count == 0:total_good_count+=1
    for k in elements_value_counts_dict.keys():
        bad_count = len(data[data[i]==k][data[Y] == 1]) #该元素Y=1的个数
        good_count = elements_value_counts_dict[k] - bad_count
        if bad_count == 0:bad_count+=1
        if good_count == 0:good_count+=1
        if bad_count*total_good_count/total_bad_count/good_count <=  0:
            woe_value = math.log(0.01,math.e)
        else:
            woe_value = math.log(bad_count*total_good_count/total_bad_count/good_count ,math.e)
        mapping_dict[k] = woe_value
    return mapping_dict


def string2numeric(data,Y,output_mapping = 0,stringColumeName = []):
    '''
    将数据中非数字型变量转化成数字型,并调用WOE函数转化成WOE值
    param：
        data: DataFrame
                需要转化的数据集
        Y: string
                Y变量的名称
        output_mapping ：Boolean
                是否输出映射关系；1为输出，0不输出，默认为0
        exclude_column_name：list
                指定不需要转化的列名，例如user_id。默认为空。
    :return DataFrame
                转化后的数据集
    '''
    Y_value_counts_dict = collections.Counter(data[Y].tolist()) #统计Y变量中元素个数
    if len(Y_value_counts_dict) == 2:
        string2numeric_columeName_list = [] #存放被映射的列名
        for i in stringColumeName:
            elements_value_counts_dict = collections.Counter(data[i].tolist()) #统计该列中元素个数
            #建立string2numeric的映射
            map_dict = woe(data[[i,Y]],elements_value_counts_dict,Y_value_counts_dict,i,Y) #计算该列中各元素的woe值,以dict返回
            string2numeric_columeName_list.append(i)
            data[i] = data[i].map(map_dict)
            if output_mapping == 1 : print("列名：",i,"  映射关系：",map_dict)
        print("string2numeric过程中被映射的列名：",string2numeric_columeName_list)
    else: print("ERROR：Y变量中元素种类不等于2！")
    return data


def string2numeric_hm(data_train,Y,data_apply,output_mapping = 0,stringColumeName = []):
    '''
    将数据中非数字型变量转化成数字型,并调用WOE函数转化成WOE值
    param：
        data_train: DataFrame
                用该数据集生成woe计算方法
        Y_train: string
                data_train 中Y变量名称
        data_apply: DataFrame
                应用woe的数据集.也应该包括转换列。
        output_mapping ：Boolean
                是否输出映射关系；1为输出，0不输出，默认为0
        exclude_column_name：list
                指定不需要转化的列名，例如user_id。默认为空。
    :return DataFrame
                转化后的数据集
    '''
    Y_value_counts_dict = collections.Counter(data_train[Y].tolist()) #统计Y变量中元素个数
    if len(Y_value_counts_dict) == 2:
        string2numeric_columeName_list = [] #存放被映射的列名
        for i in stringColumeName:
            elements_value_counts_dict = collections.Counter(data_train[i].tolist()) #统计该列中元素个数
            #建立string2numeric的映射
            map_dict = woe(data_train[[i,Y]],elements_value_counts_dict,Y_value_counts_dict,i,Y) #计算该列中各元素的woe值,以dict返回
            string2numeric_columeName_list.append(i)
            data_apply[i] = data_apply[i].map(map_dict) #应用该方法
            if output_mapping == 1 : print("列名：",i,"  映射关系：",map_dict)
        print("string2numeric过程中被映射的列名：",string2numeric_columeName_list)
    else: print("ERROR：Y变量中元素种类不等于2！")
    data_apply = data_apply.fillna(0)
    return data_apply


def stratified_sampling(data,Y,random_state_1 = 0.7,random_state_0 = 0.7):
    '''
    按Y变量进行分层抽样(Y变量取值必须为0,1),默认每层按70%抽样
    :param data:DataFrame
                所需分层抽样的数据集
    :param Y:String
                Y变量名
    :param random_state_1:
                Y=1 的抽样占比
    :param random_state_0:
                Y=0 的抽样占比
    :return: DataFrame
                抽样后的数据集
    '''
    if random_state_1 == 1:  #如果是1则不采样
        data_Y_1 = data[data[Y] == 1]
        data_Y_0 = data[data[Y] == 0]
        train_0, test_0 = train_test_split(data_Y_0, train_size=random_state_0, random_state=1)
        new_data = pd.concat([data_Y_1, train_0])
    else:
        data_Y_1 = data[data[Y] == 1]
        data_Y_0 = data[data[Y] == 0]
        train_1,test_1 = train_test_split(data_Y_1,train_size = random_state_1,random_state=1)
        train_0,test_0 = train_test_split(data_Y_0,train_size = random_state_0,random_state=1)
        new_data = pd.concat([train_1,train_0])
    return new_data

def drop_column_byNan(data,nan_ratio = 0.5):
    '''
    删除缺失值大于指定比例的列
    :param data: DataFrame
    :param nan_ratio: 缺失值的占比,默认值是50%
    :return: DataFrame
    '''
    row_num = len(data)
    column_notnull_dict = dict(data.count())  # pandas中count()可以统计非空值的个数
    for i in column_notnull_dict.keys():
        a = column_notnull_dict[i] / row_num
        b = 1-nan_ratio
        if a < b:
            data = data.drop([i],axis = 1)
    return data

def is_num(num):
    '''
    判断是否为数字
    :param num:
    :return:
    '''
    try:
        float(num)
        return True
    except ValueError:
        return False

def fill_na(data,method = 0):
    '''
    缺失值填充，更多方式查看pandas文档中缺失值处理相关章节
    :param data: DataFrame
    :param method: 默认值0
                "mean":均值填充
                int/float:实际值填充
    :return:DataFrame
    '''
    if method == "mean":
        filled_data = data.fillna(data.mean())
        return filled_data
    elif is_num(method):
        filled_data = data.fillna(method)
        return filled_data
    else:
        print("输入的参数method有误，请查看函数说明后再运行！")



def fitmodel (learning_rate,n_estimators,max_depth,subsample,dev_x,dev_y,val_x,val_y,oft_x,oft_y):
    "按需求拟合参数GBDT"
    clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample)
    clf.fit(dev_x, dev_y)
    dev_score = clf.predict_proba(dev_x)
    val_score = clf.predict_proba(val_x)
    oft_score = clf.predict_proba(oft_x)
    #print("AUC: DEV - %f , VAL - %f " %(roc_dev,roc_val), file=f)
    print("With Parameter: learning_rate=%f, estimaters=%d, max_depth=%d, subsample=%f " %(learning_rate,n_estimators,max_depth,subsample))
    print("DEV Sample - ")
    dev_ks, dev_auc, dev_gain = ks_report(dev_y.values,dev_score.T[1],bins = 20,pos_label = 1)
    print("VAL Sample - ")
    val_ks, val_auc, val_gain = ks_report(val_y.values,val_score.T[1],bins = 20,pos_label = 1)
    result = [dev_ks,dev_auc,val_ks,val_auc]
    return result


# 输出KS 报告
def ks_report_bal(y_real,y_score,balance, bins = 20,pos_label = 1):
    '''
    输出KS报告，包含最大的KS值，每个分箱的KS值
        :param y_real: Serice     原始的y值
        :param y_score: np数组    预测y=1的概率值，注意：只有一列
        :param bins: 分箱的个数
        :param pos_label: y变量中正类的值，0或1
        :return:
    '''
    #下面用来输出各分箱中的KS值
    ks_range_list = []
    y = pd.Series(data = y_real ,name="y")
    score = pd.Series(data = y_score,name="score")
    bal = pd.Series(data = balance,name="bal")
    pd_data = pd.concat([score,y,bal],axis=1)  #将两个Series按列合并
    pd_data_sorted = pd_data.sort_values(["score"],axis=0,ascending=False)  #按score列进行整体排序
    total_y_value_counts_dict = collections.Counter(pd_data_sorted["y"].tolist()) #统计全部y变量中各元素个数
    total_y_value_sum_1 = pd_data_sorted['bal'][pd_data_sorted['y']==1].sum() #统计全部y=1中bal 求和
    total_y_value_sum_0 = pd_data_sorted['bal'][pd_data_sorted['y']==0].sum() #统计全部y=1中bal 求和
    length = len(pd_data_sorted)
    bin_len = math.floor(length/bins)
    for i in range(1,bins+1):
        cumulative_length = bin_len*i
        cumulative_df = pd_data_sorted[0:cumulative_length]
        y_value_counts_dict = collections.Counter(cumulative_df["y"].tolist()) #统计各分段中y变量中各元素个数
        y_value_sum_1 = cumulative_df['bal'][cumulative_df['y']==1].sum()
        y_value_sum_0 = cumulative_df['bal'][cumulative_df['y']==0].sum()
        cumulative_badratio  = float(y_value_counts_dict[1])/total_y_value_counts_dict[1]
        cumulative_goodratio = float(y_value_counts_dict[0])/total_y_value_counts_dict[0]
        cumulative_bad_bal_ratio  = y_value_sum_1/total_y_value_sum_1
        cumulative_good_bal_ratio  = y_value_sum_0/total_y_value_sum_0
        temp = [y_value_counts_dict[1],
                total_y_value_counts_dict[1],
                cumulative_badratio,
                y_value_counts_dict[0],
                total_y_value_counts_dict[0],
                cumulative_goodratio,
                abs(cumulative_badratio-cumulative_goodratio),
                y_value_sum_1,
                total_y_value_sum_1,
                cumulative_bad_bal_ratio,
                y_value_sum_0,
                total_y_value_sum_0,
                cumulative_good_bal_ratio]
        ks_range_list.append(temp)
    # print(pd.DataFrame(ks_range_list,columns=["cum_bad%","cum_good%","KS_value"]))
    gains = pd.DataFrame(ks_range_list,columns=["cum_bad#",
                                                "tot_bad#",
                                                "cum_bad%",
                                                "cum_good#",
                                                "tot_good#",
                                                "cum_good%",
                                                "KS_value", 
                                                "cum_bad_bal$",
                                                "tot_bad_bal$",
                                                "cum_bad_bal%",
                                                "cum_good_bal$",
                                                "tot_good_bal$",
                                                "cum_good_bal%"])
    #下面用来计算最大的KS值
    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label)
    max_ks = 0.0
    for i in range(0,len(fpr)):
        D = abs(tpr[i] - fpr[i])
        if D > max_ks:
            max_ks = D
    # print("MAX_KS:",max_ks)
    #输出AUC的值
    auc = roc_auc_score(y_real,y_score)
    # print("AUC值：",auc)
    return (max_ks, auc, gains)


# 输出KS 报告
def ks_report(y_real,y_score,bins = 20,pos_label = 1):
    '''
    输出KS报告，包含最大的KS值，每个分箱的KS值
        :param y_real: Serice     原始的y值
        :param y_score: np数组    预测y=1的概率值，注意：只有一列
        :param bins: 分箱的个数
        :param pos_label: y变量中正类的值，0或1
        :return:
    '''
    #下面用来输出各分箱中的KS值
    ks_range_list = []
    y = pd.Series(data = y_real ,name="y")
    score = pd.Series(data = y_score,name="score")
    pd_data = pd.concat([score,y],axis=1)  #将两个Series按列合并
    pd_data_sorted = pd_data.sort_values(["score"],axis=0,ascending=False)  #按score列进行整体排序
    total_y_value_counts_dict = collections.Counter(pd_data_sorted["y"].tolist()) #统计全部y变量中各元素个数
    length = len(pd_data_sorted)
    bin_len = math.floor(length/bins)
    for i in range(1,bins+1):
        cumulative_length = bin_len*i
        cumulative_df = pd_data_sorted[0:cumulative_length]
        alone_df = pd_data_sorted[bin_len*(i-1):cumulative_length]
        avg_score = np.average(alone_df['score'])
        y_value_counts_dict = collections.Counter(cumulative_df["y"].tolist()) #统计各分段中y变量中各元素个数
        cumulative_badratio  = float(y_value_counts_dict[1])/total_y_value_counts_dict[1]
        cumulative_goodratio = float(y_value_counts_dict[0])/total_y_value_counts_dict[0]
        temp = [y_value_counts_dict[1], 
                y_value_counts_dict[0], 
                cumulative_badratio,
                cumulative_goodratio,
                avg_score,
                abs(cumulative_badratio-cumulative_goodratio)]
        ks_range_list.append(temp)
    # print(pd.DataFrame(ks_range_list,columns=["cum_bad_num","cum_good_num","cum_bad%","cum_good%","KS_value"]))
    # 加入"cum_bad_num","cum_good_num",20180710 Hemiao 
    gains = pd.DataFrame(ks_range_list,columns=["cum_bad_num","cum_good_num","cum_bad%","cum_good%","avg_score","KS_value"])
    #下面用来计算最大的KS值
    fpr, tpr, thresholds = roc_curve(y_real, y_score, pos_label)
    max_ks = 0.0
    for i in range(0,len(fpr)):
        D = abs(tpr[i] - fpr[i])
        if D > max_ks:
            max_ks = D
    # print("MAX_KS:",max_ks)
    #输出AUC的值
    auc = roc_auc_score(y_real,y_score)
    # print("AUC值：",auc)
    return (max_ks, auc, gains)

# 读入csv文件，导入字段。导入数据在类的buffer数据中
import csv

class readCSV(object):
    def __init__(self,path="Demo.csv"):
        #创建一个属性用来保存要操作CSV的文件
        self.path=path
        try:
            #打开一个csv文件，并赋予读的权限
            self.csvHand=open(self.path,"r")
            #调用csv的reader函数读取csv文件
            self.readcsv=csv.reader(self.csvHand)
            #创建一个list用来保存csv中的内容
            self.buffer=[]
            try:
                #把csv中内容存入list 中
                for row in self.readcsv:
                    #self.buffer.append(row)
                    self.buffer.extend(row)  #稍微修改，防止类型为list
            except Exception as e:
                print(e)
        except  Exception as e:
            print(e)
        finally:
            #关闭csv文件
            self.csvHand.close()



'''
这个函数用作hive表下载到mlp服务器本地。
20180614 hemiao
'''
def hive_data_get(hive_name, # 数坊表名
                  hdfs_file='/tmp/hm/sample_yuqi_fenxi.csv',  # hdfs数据文件和位置
                  local_file='/home/mlp/notebooks/yuqifenxi/fenxi_20180614.csv'): # 本地数据文件和位置
    print("select * from {};".format(hive_name))
    a = spark.sql("select * from {}".format(hive_name)) 
    print("Step0 ----generate spark dataframe ---Done\n")

    a.coalesce(1).write.csv(path=hdfs_file, mode="overwrite", header=True)
    print("Step1 ----hdfs write ---Done\n")

    os.system("rm {}".format(local_file))
    os.system("hadoop fs -get {} ./data".format(hdfs_file))
    print("Step2 ----get to local file :{}---Done\n".format(local_file))

    os.system("mv ./data/part-*.csv {}".format(local_file))
    os.system("rm -R ./data/")

    return




'''
char_auto_woe函数用于从按月变化（或其他维度变化）的数据集中，选取比较稳定、有效的字符型变量的粗分类。并自动生成WOE和分段。
headers：需要分析的字段列表，需要是字符型变量。
df1：参考数据集
df2-df4：比较数据集，用于衡量分段稳定性
tgt：好坏标签，
rpt_path：生成报告的目录，例如'./rpt';

“稳定”：指PSI在各个月较小。
“有效”：粗分类不会导致较大的区分度下降。
在3-8段之间选取稳定和有效的平衡点。自动生成group（1、2、3、。。）和woe的mapping逻辑。

 2018/07/10 Hemiao
'''

def char_auto_woe(headers, df1, df2, df3, df4, tgt, rpt_path):

# woe需要用到的函数：

    def calc_pro(data, tgt):
        '''自动化的算概率'''
        noduplicate_list = data[headers[i]].drop_duplicates().reset_index(drop=True)
        result = pd.DataFrame(columns=[headers[i],'good_num','bad_num','bad_ratio'])
        for name_tmp in noduplicate_list:
            good_tmp = data['bulid_num'][(data[headers[i]]==name_tmp)&(data[tgt]==0)].reset_index(drop=True)
            bad_tmp = data['bulid_num'][(data[headers[i]]==name_tmp)&(data[tgt]==1)].reset_index(drop=True)
            if len(good_tmp)>0:
                good_num = good_tmp[0]
                if len(bad_tmp)>0:
                    bad_num = bad_tmp[0]
                else:
                    bad_num = 0
                bad_ratio = bad_num/(bad_num+good_num)
            else:
                good_num = 0
                if len(bad_tmp)>0:
                    bad_num = bad_tmp[0]
                    bad_ratio = bad_num/(bad_num+good_num)
                else:
                    bad_ratio = 'NULL'
            result = result.append([{headers[i]:name_tmp,'good_num':good_num,'bad_num':bad_num,'bad_ratio':bad_ratio}])
            result = result.sort_values(by = 'bad_ratio')
            result = result.reset_index(drop=True)
        return result


    def my_psi(build_data,off_data,name):
        build_distribution             = pd.DataFrame()
        build_distribution[name] = build_data[name]
        build_distribution['build_num']= build_data['good_num']+build_data['bad_num']
        off_distribution               = pd.DataFrame()
        off_distribution[name]   = off_data[name]
        off_distribution['off_num']    = off_data['good_num']+off_data['bad_num']
        psi_data = pd.merge(build_distribution,off_distribution,on=name)

        build_allnum = psi_data['build_num'].sum()
        allnum = psi_data['off_num'].sum()
        psi_value=0
        for ii in range(len(psi_data)):
            psi_value = psi_value+np.log((psi_data.loc[ii,'build_num']/build_allnum)/(psi_data.loc[ii,'off_num']/allnum)) \
            * ((psi_data.loc[ii,'build_num']/build_allnum)-(psi_data.loc[ii,'off_num']/allnum))
        return psi_value

    def my_ks(good_se,bad_se):
        '''计算ks值
        '''
        tmp_df = pd.DataFrame()
        tmp_df['good_num']         = good_se.cumsum()
        tmp_df['bad_num']          = bad_se.cumsum()
        tmp_df['good_all']         = good_se.sum()
        tmp_df['bad_all']          = bad_se.sum()
        tmp_df['good_cumul_ratio'] = tmp_df['good_num'] / tmp_df['good_all']
        tmp_df['bad_cumul_ratio']  = tmp_df['bad_num'] / tmp_df['bad_all']
        tmp_df['ks']               = np.abs(tmp_df['good_cumul_ratio']-tmp_df['bad_cumul_ratio']) 
        return tmp_df,max(tmp_df['ks'])

    def my_iv(good_se,bad_se):
        '''计算iv值
        '''
        tmp_df = pd.DataFrame()
        tmp_df['good_num']         = good_se
        tmp_df['badd_num']         = bad_se
        tmp_df['good_all']         = good_se.sum()
        tmp_df['bad_all']          = bad_se.sum()
        tmp_df['good_ratio']       = good_se / tmp_df['good_all']
        tmp_df['bad_ratio']        = bad_se / tmp_df['bad_all']
        tmp_df = tmp_df.drop(tmp_df[(tmp_df['good_ratio']==0) | (tmp_df['bad_ratio']==0)].index, axis=0)
        tmp_df=tmp_df.reset_index(drop=True)
        iv_value = 0
        for i in range(len(tmp_df)):
            iv_value = iv_value+np.log(tmp_df.loc[i,'bad_ratio']/tmp_df.loc[i,'good_ratio']) \
            * (tmp_df.loc[i,'bad_ratio']-tmp_df.loc[i,'good_ratio'])

        return iv_value

    def my_woe(data):
        headers = data.columns.values.tolist()
        name = headers[0]
        tmp = pd.DataFrame()
        tmp['new_group'] = data[name]
        bad_all = data['bad_num'].sum()
        good_all = data['good_num'].sum()
        for i in range(len(data)):
            if data.loc[i,'bad_num']!=0 or data.loc[i,'good_num']!=0:
                tmp.loc[i,'woe'] = np.log((data.loc[i,'bad_num']/bad_all)/(data.loc[i,'good_num']/good_all))
            else:
                tmp.loc[i,'woe'] = -99
        return tmp

    
    report_map = pd.DataFrame()
    report_df = pd.DataFrame(columns=['feature','重新分组的数量','old_build_ks','new_build_ks','old_build_iv','new_build_iv','old_02_psi','new_02_psi','new_03_psi'])


    map_group_re = {}
    map_woe_re = {}
    for i in range(0, len(headers)):
#     for i in range(0, 1):
        #建模样本
        tmp_201710_201801                = pd.DataFrame()
        tmp_201710_201801[headers[i]]    = df1[headers[i]]
        tmp_201710_201801[tgt] = df1[tgt]
        tmp_201710_201801['bulid_num']   = 1
        build_data                       = tmp_201710_201801.groupby([headers[i],tgt],as_index=False)['bulid_num'].sum()
    #    my_tmp_bulid = tmp_201710_201801.groupby([headers[i],tgt],as_index=False).count()
        build_bad_ratio                  = calc_pro(build_data, tgt)

        #offtime样本 2018-02
        tmp_201802                = pd.DataFrame()
        tmp_201802[headers[i]]    = df2[headers[i]]
        tmp_201802[tgt] = df2[tgt]
        tmp_201802['bulid_num']   = 1
        off_data_02               = tmp_201802.groupby([headers[i],tgt],as_index=False)['bulid_num'].sum()
        off_bad_ratio_02          = calc_pro(off_data_02, tgt)
        #2018-03
        tmp_201803                = pd.DataFrame()
        tmp_201803[headers[i]]    = df3[headers[i]]
        tmp_201803[tgt] = df3[tgt]
        tmp_201803['bulid_num']   = 1
        off_data_03               = tmp_201803.groupby([headers[i],tgt],as_index=False)['bulid_num'].sum()
        off_bad_ratio_03          = calc_pro(off_data_03, tgt)
        #2018-04
        tmp_201804                = pd.DataFrame()
        tmp_201804[headers[i]]    = df4[headers[i]]
        tmp_201804[tgt] = df4[tgt]
        tmp_201804['bulid_num']   = 1
        off_data_04               = tmp_201804.groupby([headers[i],tgt],as_index=False)['bulid_num'].sum()
        off_bad_ratio_04          = calc_pro(off_data_04, tgt)

        feature_newgroup = pd.DataFrame()
        old_map = pd.DataFrame()
        every_report_df = pd.DataFrame(columns=['feature','重新分组的数量','old_build_ks','new_build_ks','old_build_iv','new_build_iv','old_02_psi','new_02_psi','new_03_psi','new_04_psi'])     
        if len(build_bad_ratio) < 7:
            old_df,old_ks   = my_ks(build_bad_ratio['good_num'],build_bad_ratio['bad_num'])
            old_iv = my_iv(build_bad_ratio['good_num'],build_bad_ratio['bad_num'])
            old_psi_value = my_psi(build_bad_ratio,off_bad_ratio_02,name=headers[i])
            report_df = report_df.append([{'feature':headers[i],'重新分组的数量':'不变','old_build_ks':old_ks,'new_build_ks':0,\
                                           'old_build_iv':old_iv,'new_build_iv':0,'old_02_psi':old_psi_value,'new_02_psi':0,'new_03_psi':0,'new_04_psi':0}])
            woe_tmp = my_woe(build_bad_ratio)
            map_woe = dict(zip(woe_tmp['new_group'],woe_tmp['woe']))

            old_map[headers[i]] = build_bad_ratio[headers[i]]
            old_map['new_group_woe'] = old_map[headers[i]].map(map_woe)
            report_map = pd.concat([report_map,old_map],axis=1)

            dict_woe = {}
            dict_group = {}
            for k in range(len(old_map)):
                tmp_woe = {str(old_map.loc[k,headers[i]]):old_map.loc[k,'new_group_woe']}
            # 改为直接对应成数值 Hemiao
                tmp_group = {str(old_map.loc[k,headers[i]]):str(k)}                
            #    tmp_group = {str(old_map.loc[k,headers[i]]):str(old_map.loc[k,headers[i]])}
                dict_woe =dict(dict_woe, **tmp_woe)
                dict_group=dict(dict_group, **tmp_group)
            map_woe_re[headers[i]] = dict_woe
            map_group_re[headers[i]] = dict_group

        else:    
            '''=====================进行重分组===================================================
            '''  
            for bin_num in range(3,8):  #分箱数量 #分10组以上注意排序
                list_binning = []
                for j in range(len(build_bad_ratio)):
                    tmp_l = [build_bad_ratio.loc[j,'bad_ratio']]*(build_bad_ratio.loc[j,'good_num']+build_bad_ratio.loc[j,'bad_num'])
                    list_binning = list_binning +tmp_l
                try:
                    border_values = pd.qcut(list_binning,bin_num,duplicates='drop',retbins=True)[1]            #分箱的边界值
                    border_index = [0]
                    for num in border_values.tolist()[1:]:
                        border_index.append(build_bad_ratio[build_bad_ratio.bad_ratio==num].index.tolist()[0])       #分箱边界值对应的原df的index
    #  #  去掉group的前缀。. Hemiao 20180710
                    build_bad_ratio.loc[0,'new_group'] = '0' 
                    for k in range(len(border_index)-1):
                        build_bad_ratio.loc[border_index[k]+1:border_index[k+1]+1,'new_group'] = str(k)  #在原来df上加入新的一列也就是新的分组号
                    build_bad_ratio.loc[border_index[k+1]+1:len(build_bad_ratio),'new_group'] = str(k)
    #                 group_name = build_bad_ratio.groupby(['new_group'])[headers[i]].sum()         

    #                build_bad_ratio.loc[0,'new_group'] = 'group_0'  
    #                for k in range(len(border_index)-1):
    #                    build_bad_ratio.loc[border_index[k]+1:border_index[k+1]+1,'new_group'] = 'group_'+str(k)  #在原来df上加入新的一列也就是新的分组号
    #                build_bad_ratio.loc[border_index[k+1]+1:len(build_bad_ratio),'new_group'] = 'group_'+str(k)
    #                group_name = build_bad_ratio.groupby(['new_group'])[headers[i]].sum()                         #新组号和原来组的对应情况

                    build_new_group = build_bad_ratio.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
                #    build_new_group = build_new_group.sort_values(by = 'bad_ratio',axis = 0)                         #重新计算每组的好坏人的数量
                    build_new_group['bad_ratio'] = build_new_group['bad_num']/(build_new_group['good_num']+build_new_group['bad_num'])

                    map_dict = dict(zip(build_bad_ratio[headers[i]],build_bad_ratio['new_group']))

                    '''==================验证样本================================================================================
                    '''
                    off_bad_ratio_02['new_group'] = off_bad_ratio_02[headers[i]].map(map_dict)
                    off_bad_ratio_02 = off_bad_ratio_02.fillna('0')
                    off_new_group_02 = off_bad_ratio_02.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
                    off_new_group_02['bad_ratio'] = off_new_group_02['bad_num']/(off_new_group_02['good_num']+off_new_group_02['bad_num'])

                    off_bad_ratio_03['new_group'] = off_bad_ratio_03[headers[i]].map(map_dict)
                    off_bad_ratio_03 = off_bad_ratio_03.fillna('0')
                    off_new_group_03 = off_bad_ratio_03.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
                    off_new_group_03['bad_ratio'] = off_new_group_03['bad_num']/(off_new_group_03['good_num']+off_new_group_03['bad_num'])

                    off_bad_ratio_04['new_group'] = off_bad_ratio_04[headers[i]].map(map_dict)
                    off_bad_ratio_04 = off_bad_ratio_04.fillna('0')
                    off_new_group_04 = off_bad_ratio_04.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
                    off_new_group_04['bad_ratio'] = off_new_group_04['bad_num']/(off_new_group_04['good_num']+off_new_group_04['bad_num'])

                    old_df,old_ks   = my_ks(build_bad_ratio['good_num'],build_bad_ratio['bad_num'])                  #原来ks
                    new_df,new_ks = my_ks(build_new_group['good_num'],build_new_group['bad_num'])                #分组后的ks
                    old_iv = my_iv(build_bad_ratio['good_num'],build_bad_ratio['bad_num'])
                    new_iv = my_iv(build_new_group['good_num'],build_new_group['bad_num'])
    #                aold_iv = my_iv(off_bad_ratio['good_num'],off_bad_ratio['bad_num'])
    #                anew_iv = my_iv(off_new_group['good_num'],off_new_group['bad_num'])
                    '''计算psi
                    '''
                    old_psi_value = my_psi(build_bad_ratio,off_bad_ratio_02,name=headers[i])
    #                old_psi_value = my_psi(build_bad_ratio,off_bad_ratio,name=headers[i])
                    new_psi_value_02 = my_psi(build_new_group,off_new_group_02,name='new_group')
                    new_psi_value_03 = my_psi(build_new_group,off_new_group_03,name='new_group')
                    new_psi_value_04 = my_psi(build_new_group,off_new_group_04,name='new_group')

                    report_df = report_df.append([{'feature':headers[i],'重新分组的数量':len(build_new_group),'old_build_ks':old_ks,'new_build_ks':new_ks,\
                                           'old_build_iv':old_iv,'new_build_iv':new_iv,'old_02_psi':old_psi_value,'new_02_psi':new_psi_value_02,'new_03_psi':new_psi_value_03,'new_04_psi':new_psi_value_04}])
                    every_report_df = every_report_df.append([{'feature':headers[i],'重新分组的数量':len(build_new_group),'old_build_ks':old_ks,'new_build_ks':new_ks,\
                                           'old_build_iv':old_iv,'new_build_iv':new_iv,'old_02_psi':old_psi_value,'new_02_psi':new_psi_value_02,'new_03_psi':new_psi_value_03,'new_04_psi':new_psi_value_04}])
                except:
                    report_df = report_df.append([{'feature':headers[i],'重新分组的数量':'不变','old_build_ks':0,'new_build_ks':0,\
                                           'old_build_iv':0,'new_build_iv':0,'old_02_psi':0,'new_02_psi':0,'new_03_psi':0,'new_04_psi':0}])
            '''选择max的score对应的bin_num
            '''
            every_report_df = every_report_df.reset_index(drop=True)
            every_report_df['psi_sum'] = (every_report_df['new_02_psi']+every_report_df['new_03_psi']+every_report_df['new_04_psi'])/3
            every_report_df['score']   = every_report_df['new_build_iv']-every_report_df['psi_sum'] 
            every_tmp= every_report_df[every_report_df['score']==max(every_report_df['score'])].reset_index(drop=True)
            bin_num = every_tmp.loc[0,'重新分组的数量']
            list_binning = []
            for j in range(len(build_bad_ratio)):
                tmp_l = [build_bad_ratio.loc[j,'bad_ratio']]*(build_bad_ratio.loc[j,'good_num']+build_bad_ratio.loc[j,'bad_num'])
                list_binning = list_binning +tmp_l
            border_values = pd.qcut(list_binning,bin_num,duplicates='drop',retbins=True)[1]            #分箱的边界值
            border_index = [0]
            for num in border_values.tolist()[1:]:
                border_index.append(build_bad_ratio[build_bad_ratio.bad_ratio==num].index.tolist()[0])       #分箱边界值对应的原df的index

            #  去掉group的前缀。. Hemiao 20180710
            build_bad_ratio.loc[0,'new_group'] = '0'  
            for k in range(len(border_index)-1):
                build_bad_ratio.loc[border_index[k]+1:border_index[k+1]+1,'new_group'] = str(k)  #在原来df上加入新的一列也就是新的分组号
            build_bad_ratio.loc[border_index[k+1]+1:len(build_bad_ratio),'new_group'] = str(k)
            build_new_group = build_bad_ratio.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
            build_new_group['bad_ratio'] = build_new_group['bad_num']/(build_new_group['good_num']+build_new_group['bad_num'])

    #        build_bad_ratio.loc[0,'new_group'] = 'group_0'  
    #        for k in range(len(border_index)-1):
    #            build_bad_ratio.loc[border_index[k]+1:border_index[k+1]+1,'new_group'] = 'group_'+str(k)  #在原来df上加入新的一列也就是新的分组号
    #        build_bad_ratio.loc[border_index[k+1]+1:len(build_bad_ratio),'new_group'] = 'group_'+str(k)
    #        build_new_group = build_bad_ratio.groupby(['new_group'],as_index=False).agg({'good_num': 'sum', 'bad_num': 'sum'})
    #        build_new_group['bad_ratio'] = build_new_group['bad_num']/(build_new_group['good_num']+build_new_group['bad_num'])
    #        
            woe_tmp = my_woe(build_new_group)
            map_woe = dict(zip(woe_tmp['new_group'],woe_tmp['woe']))

            feature_newgroup[headers[i]]   = build_bad_ratio[headers[i]]
            feature_newgroup['new_group'] = build_bad_ratio['new_group']
            feature_newgroup['new_group_woe'] = feature_newgroup['new_group'].map(map_woe)

            report_map= pd.concat([report_map,feature_newgroup],axis=1)

            dict_woe = {}
            dict_group = {}
            for k in range(len(feature_newgroup)):
                tmp_woe = {str(feature_newgroup.loc[k,headers[i]]):feature_newgroup.loc[k,'new_group_woe']}
                #  改成map 到new_group. Hemiao 20180710
                tmp_group = {str(feature_newgroup.loc[k,headers[i]]):feature_newgroup.loc[k,'new_group']}
                dict_woe =dict(dict_woe, **tmp_woe)
                dict_group=dict(dict_group, **tmp_group)
            map_woe_re[headers[i]] = dict_woe
            map_group_re[headers[i]] = dict_group

    report_df['psi_sum'] = (report_df['new_02_psi']+report_df['new_03_psi']+report_df['new_04_psi'])/3
    report_df['score']   = report_df['new_build_iv']-report_df['psi_sum']
    report_map.to_csv("{}/woe_group_final_report.csv".format(rpt_path),encoding='gb18030',index=False)
    report_df.to_csv("{}/woe_group_trial_report.csv".format(rpt_path),encoding='gb18030',index=False)
    f = open("{}/woe_mapping.txt".format(rpt_path),'w',encoding='utf-8')  
    f.write(str(map_woe_re))  
    f.close()  
    f = open("{}/group_mapping.txt".format(rpt_path),'w',encoding='utf-8')  
    f.write(str(map_group_re))  
    f.close()
    return



class model_train(object):
    
    

    def __init__(self, numlst="numerical_varlist.csv", charlst="char_varlist.csv", tgt='target'):
        # 字符型变量列表
        self.charlist = readCSV(path=charlst).buffer
        # 数值型变量列表
        self.numlist = readCSV(path=numlst).buffer
        # X变量列表
        self.varlist = self.charlist + self.numlist
        # 变量类型列表
        self.var_type = []
        for i in self.charlist:
            self.var_type.append('string')
        for i in self.numlist:
            self.var_type.append('number')
        # 二分类target
        self.tgt = tgt
        # X+Y变量列表
        self.varlist_tot = self.varlist.copy()
        self.varlist_tot.append(self.tgt)
        return
    

    
    def read_csv_data(self, path="Demo.csv", encoding='ANSI', na_values='(NULL)'):
        '''
        读入数据，从path。
        '''
        sample = pd.read_csv(filepath_or_buffer=path,encoding=encoding, na_values=na_values, error_bad_lines=False)
        return sample

    def transform(self, sample):
        #一些预处理
        # 身份证和手机号所属地的解析
        sample['mobile_loc_prov'] = sample.mobile_location.str.split(",", expand=True)[0]
        sample['mobile_loc_city'] = sample.mobile_location.str.split(",", expand=True)[1]
        sample['idcard_loc_prov'] = sample.idcard_location.str.split(",", expand=True)[0]
        sample['idcard_loc_city'] = sample.idcard_location.str.split(",", expand=True)[1]
        return sample

    def treat(self, sample, treat_value=0):
        '''
        缺失值去处理成treat_value
        '''
        sample_trt = sample.fillna(treat_value)
        return sample_trt

    def basic_profile(self, sample, path1="contents.csv", path2="describe.csv", lielian_flag=0):
        '''
        基本的描述性统计
        '''
        self.types = sample.dtypes
        self.statistics = sample.describe(percentiles = [.01, .25, .5, .75, .99]).T
        self.types.to_csv(path1)
        self.statistics.to_csv(path2)
        # 原来的self是mt，报错？？
        if lielian_flag == 1:
            self.lielian(sample, self.charlist, path3="lielian.csv")

    def iv(self, sample_trt, path="iv.csv"):
        ''' IV,目前支持字符型和数值型
        '''
        x = sample_trt[self.varlist]
        y = sample_trt[self.tgt]
        woe, iv = iv1.WOE().woe(x,y)
        #匹配到字段--对多个分类变量的有问题。
        test = [x.columns.values,x.dtypes,iv]
        test1 = DataFrame(test).T
        test1.columns = ['变量名','类型','IV']
        # 按IV大小排序
        test1.sort_values(axis=0, ascending=False, by='IV').to_csv(path)

    def split(self, sample_trt, test_size=0.3):
        dev, val = train_test_split(sample_trt, test_size=test_size, random_state=1)
        print("DEV size --\n", dev[self.tgt].value_counts())
        print("VAL size --\n", val[self.tgt].value_counts())
        return dev, val

    def StrToNum(self, train_sample, dev, val):
        tmp = train_sample.copy()
        dev_woe = string2numeric_hm(data_train=tmp,
                              Y=self.tgt,
                              data_apply=dev,
                              stringColumeName=self.charlist)
        val_woe = string2numeric_hm(data_train=tmp,
                            Y=self.tgt,
                            data_apply=val,
                            stringColumeName=self.charlist)
        return dev_woe, val_woe

    def x_y_split(self, sample):
#        x = sample.drop(self.tgt, axis=1)
        x = sample[self.varlist]
        y = sample[self.tgt]
        return x, y

    def fraud_reports(self, df, classifier, bal, bins=100, type='GBDT', pr=1):
        df_x, df_y = self.x_y_split(df)
        df_score = classifier.predict_proba(df_x)
#         df_ks, df_auc, df_gain = ks_report(df_y.values, df_score.T[1], bins=bins)
        df_ks, df_auc, df_gain = ks_report_bal(df_y.values, df_score.T[1], df[bal].values, bins=bins)       
        #print 1% capture Rate
        if pr==1:
            print("前1%抓取: ", df_gain.iloc[0])
            print("ks: ", df_ks, "auc: ", df_auc)
        return df_gain

    def normal_reports(self, df, classifier, bins=20, type='GBDT', pr=1):
        df_x, df_y = self.x_y_split(df)
        df_score = classifier.predict_proba(df_x)
        df_ks, df_auc, df_gain = ks_report(df_y.values, df_score.T[1], bins=bins)       
        return df_ks

    def some_reports(self, dev, val, oft1, oft2, oft3, classifier, plot_vars_num='null', bins=20, type='GBDT', psi_flag=0, bivar_flag=1, filename='./output.xlsx'):
        dev_x, dev_y = self.x_y_split(dev)
        val_x, val_y = self.x_y_split(val)
        oft1_x, oft1_y = self.x_y_split(oft1)
        oft2_x, oft2_y = self.x_y_split(oft2)
        oft3_x, oft3_y = self.x_y_split(oft3)
        
        dev_score = classifier.predict_proba(dev_x)
        val_score = classifier.predict_proba(val_x)
        oft1_score = classifier.predict_proba(oft1_x)
        oft2_score = classifier.predict_proba(oft2_x)
        oft3_score = classifier.predict_proba(oft3_x)   
        
        # psi 分布图
        plt.figure(figsize=(5, 3),dpi=150)
        plt.hist(dev_score[:,1], density=True, histtype='step', bins=20, label='DEV')
        plt.hist(val_score[:,1], density=True, histtype='step', bins=20, label='VAL')
        plt.hist(oft1_score[:,1], density=True, histtype='step', bins=20, label='OFT1')
        plt.hist(oft2_score[:,1], density=True, histtype='step', bins=20, label='OFT2')
        plt.hist(oft3_score[:,1], density=True, histtype='step', bins=20, label='OFT3')
        plt.title('psi curve')
        plt.legend(loc='best')
        plt.xlabel('score')
        plt.savefig('test1.jpg', bbox_inches='tight')
        plt.show()
        

        
        workbook = xlsxwriter.Workbook(filename)
        cell_format_bold = workbook.add_format({'bold':True,'font_name':u'微软雅黑'}) 
        cell_format1=workbook.add_format({'font_name':u'微软雅黑','font_size':10})
        cell_format2=workbook.add_format({'font_name':u'微软雅黑','font_size':11,'font_color':'red'})
        
        worksheet0 = workbook.add_worksheet('log')
        worksheet0.write('A1','Model Report', cell_format_bold)
        
        # 样本好坏分布
        t1 = pd.DataFrame(columns=['num_good', 'num_bad'], index=['dev','val','oft1','oft2','oft3'])
        t = dev[self.tgt].value_counts()
        t1.loc['dev','num_good'] = t[0]
        t1.loc['dev','num_bad'] = t[1]
        t = val[self.tgt].value_counts()
        t1.loc['val','num_good'] = t[0]
        t1.loc['val','num_bad'] = t[1]
        t = oft1[self.tgt].value_counts()
        t1.loc['oft1','num_good'] = t[0]
        t1.loc['oft1','num_bad'] = t[1]
        t = oft2[self.tgt].value_counts()
        t1.loc['oft2','num_good'] = t[0]
        t1.loc['oft2','num_bad'] = t[1]
        t = oft3[self.tgt].value_counts()
        t1.loc['oft3','num_good'] = t[0]
        t1.loc['oft3','num_bad'] = t[1]
        t1['bad_rate']=t1['num_bad']/(t1['num_bad']+t1['num_good'])
        
        worksheet0.write_row('B3', list(t1.columns), cell_format2)
        worksheet0.write_column('A4', t1.index,cell_format1)
        worksheet0.write_column('B4', t1['num_good'],cell_format1)
        worksheet0.write_column('C4', t1['num_bad'],cell_format1)
        worksheet0.write_column('D4', t1['bad_rate'],cell_format1)
        
        worksheet0.write('A11','Xgboost Model Params Setting:', cell_format_bold)
        
        
        worksheet_desc = workbook.add_worksheet('describe')
        worksheet_desc.write('A1','Dev Variables Base Description', cell_format_bold)
        dev_desc=dev_x.describe().T.reset_index()
        worksheet_desc.write_row('A3', list(dev_desc.columns), cell_format_bold)
        worksheet_desc.write_column('A4', dev_desc['index'],cell_format1)
        worksheet_desc.write_column('B4', dev_desc['count'],cell_format1)
        worksheet_desc.write_column('C4', dev_desc['mean'],cell_format1)
        worksheet_desc.write_column('D4', dev_desc['std'],cell_format1)
        worksheet_desc.write_column('E4', dev_desc['min'],cell_format1)
        worksheet_desc.write_column('F4', dev_desc['25%'],cell_format1)
        worksheet_desc.write_column('G4', dev_desc['50%'],cell_format1)
        worksheet_desc.write_column('H4', dev_desc['75%'],cell_format1)
        worksheet_desc.write_column('I4', dev_desc['max'],cell_format1)
        
        worksheet = workbook.add_worksheet('demo')
        worksheet.insert_image('A1','test1.jpg')
        if psi_flag==1:
                # 计算psi
            psi_dev_val, actual_dev_val = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(val_score[:,1]), k=20)
            psi_dev_oft1, actual_dev_oft1 = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(oft1_score[:,1]), k=20)
            psi_dev_oft2, actual_dev_oft2 = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(oft2_score[:,1]), k=20)
            psi_dev_oft3, actual_dev_oft3 = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(oft3_score[:,1]), k=20)
            print('psi_val:{}'.format(psi_dev_val))
            print('psi_oft1:{}'.format(psi_dev_oft1))
            print('psi_oft2:{}'.format(psi_dev_oft2))
            print('psi_oft3:{}'.format(psi_dev_oft3))
            
            worksheet.write('H4', 'psi_val:{}'.format(psi_dev_val),cell_format1)
            worksheet.write('H5', 'psi_oft1:{}'.format(psi_dev_oft1),cell_format1)
            worksheet.write('H6', 'psi_oft2:{}'.format(psi_dev_oft2),cell_format1)
            worksheet.write('H7', 'psi_oft3:{}'.format(psi_dev_oft3),cell_format1)
            
        #　KS数据输出到 xlsx
        dev_ks, dev_auc, dev_gain = ks_report(dev_y.values, dev_score.T[1], bins=bins)
        val_ks, val_auc, val_gain = ks_report(val_y.values, val_score.T[1], bins=bins)
        oft1_ks, oft1_auc, oft1_gain = ks_report(oft1_y.values, oft1_score.T[1], bins=bins)
        oft2_ks, oft2_auc, oft2_gain = ks_report(oft2_y.values, oft2_score.T[1], bins=bins)
        oft3_ks, oft3_auc, oft3_gain = ks_report(oft3_y.values, oft3_score.T[1], bins=bins)
        
       
        
        worksheet.write('A52','DEV Sample - KS={:.2f}%'.format(dev_ks*100),cell_format2)
        worksheet.write('B52','AUC={:.2f}%'.format(dev_auc*100),cell_format2)
        worksheet.write_row('A53', list(dev_gain.columns), cell_format_bold)
        worksheet.write_column('A54', dev_gain['cum_bad_num'],cell_format1)
        worksheet.write_column('B54', dev_gain['cum_good_num'],cell_format1)
        worksheet.write_column('C54', dev_gain['cum_bad%'],cell_format1)
        worksheet.write_column('D54', dev_gain['cum_good%'],cell_format1)
        worksheet.write_column('E54', dev_gain['avg_score'],cell_format1)
        worksheet.write_column('F54', dev_gain['KS_value'],cell_format1)
        
        worksheet.write('A76','VAL Sample - KS={:.2f}%'.format(val_ks*100),cell_format2)
        worksheet.write('B76','AUC={:.2f}%'.format(val_auc*100),cell_format2)
        worksheet.write_row('A77', list(val_gain.columns), cell_format_bold )
        worksheet.write_column('A78', val_gain['cum_bad_num'],cell_format1)
        worksheet.write_column('B78', val_gain['cum_good_num'],cell_format1)
        worksheet.write_column('C78', val_gain['cum_bad%'],cell_format1)
        worksheet.write_column('D78', val_gain['cum_good%'],cell_format1)
        worksheet.write_column('E78', val_gain['avg_score'],cell_format1)
        worksheet.write_column('F78', val_gain['KS_value'],cell_format1)
        
        worksheet.write('A100','OFT1 Sample - KS={:.2f}%'.format(oft1_ks*100),cell_format2)
        worksheet.write('B100','AUC={:.2f}%'.format(oft1_auc*100),cell_format2)
        worksheet.write_row('A101', list(oft1_gain.columns), cell_format_bold )
        worksheet.write_column('A102', oft1_gain['cum_bad_num'],cell_format1)
        worksheet.write_column('B102', oft1_gain['cum_good_num'],cell_format1)
        worksheet.write_column('C102', oft1_gain['cum_bad%'],cell_format1)
        worksheet.write_column('D102', oft1_gain['cum_good%'],cell_format1)
        worksheet.write_column('E102', oft1_gain['avg_score'],cell_format1)
        worksheet.write_column('F102', oft1_gain['KS_value'],cell_format1)
        
        worksheet.write('A124','OFT2 Sample - KS={:.2f}%'.format(oft2_ks*100),cell_format2)
        worksheet.write('B124','AUC={:.2f}%'.format(oft2_auc*100),cell_format2)
        worksheet.write_row('A125', list(oft2_gain.columns), cell_format_bold )
        worksheet.write_column('A126', oft2_gain['cum_bad_num'],cell_format1)
        worksheet.write_column('B126', oft2_gain['cum_good_num'],cell_format1)
        worksheet.write_column('C126', oft2_gain['cum_bad%'],cell_format1)
        worksheet.write_column('D126', oft2_gain['cum_good%'],cell_format1)
        worksheet.write_column('E126', oft2_gain['avg_score'],cell_format1)
        worksheet.write_column('F126', oft2_gain['KS_value'],cell_format1)
        
        worksheet.write('A148','OFT3 Sample - KS={:.2f}%'.format(oft3_ks*100),cell_format2)
        worksheet.write('B148','AUC={:.2f}%'.format(oft3_auc*100),cell_format2)
        worksheet.write_row('A149', list(oft3_gain.columns),cell_format_bold )
        worksheet.write_column('A150', oft3_gain['cum_bad_num'],cell_format1)
        worksheet.write_column('B150', oft3_gain['cum_good_num'],cell_format1)
        worksheet.write_column('C150', oft3_gain['cum_bad%'],cell_format1)
        worksheet.write_column('D150', oft3_gain['cum_good%'],cell_format1)
        worksheet.write_column('E150', oft3_gain['avg_score'],cell_format1)
        worksheet.write_column('F150', oft3_gain['KS_value'],cell_format1)        
         
        print("DEV Sample - KS={:.2f}%, AUC={:.2f}%".format(dev_ks*100, dev_auc*100))
#         print(dev_gain)
        print("VAL Sample - KS={:.2f}%, AUC={:.2f}%".format(val_ks*100, val_auc*100))
#         print(val_gain)
        print("OFT1 Sample - KS={:.2f}%, AUC={:.2f}%".format(oft1_ks*100, oft1_auc*100))
#         print(oft1_gain)

        print("OFT2 Sample - KS={:.2f}%, AUC={:.2f}%".format(oft2_ks*100, oft2_auc*100))
        print("OFT3 Sample - KS={:.2f}%, AUC={:.2f}%".format(oft3_ks*100, oft3_auc*100))

        # KS 曲线
        # 加入为 0 的行
        dev_gain.index = range(1,len(dev_gain) + 1) 
        dev_gain.loc[0] = 0
        dev_gain = dev_gain.sort_index(ascending=True)
        val_gain.index = range(1,len(val_gain) + 1) 
        val_gain.loc[0] = 0
        val_gain = val_gain.sort_index(ascending=True)
        oft1_gain.index = range(1,len(oft1_gain) + 1) 
        oft1_gain.loc[0] = 0
        oft1_gain = oft1_gain.sort_index(ascending=True)
        
        oft2_gain.index = range(1,len(oft2_gain) + 1) 
        oft2_gain.loc[0] = 0
        oft2_gain = oft2_gain.sort_index(ascending=True)
        
        oft3_gain.index = range(1,len(oft3_gain) + 1) 
        oft3_gain.loc[0] = 0
        oft3_gain = oft3_gain.sort_index(ascending=True)
        plt.figure(figsize=(20, 3),dpi=100)
        p1 = plt.subplot(151)
        p2 = plt.subplot(152)
        p3 = plt.subplot(153)
        p4 = plt.subplot(154)
        p5 = plt.subplot(155)
        # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=0.2, wspace=0.3)
        p1.plot(dev_gain['cum_bad%'])
        p1.plot(dev_gain['cum_good%'])
        p1.legend(loc='best')
        p1.set_title("DEV KS = {:.2f}%".format(dev_ks * 100),fontsize=10)
        
        p2.plot(val_gain['cum_bad%'])
        p2.plot(val_gain['cum_good%'])
        p2.legend(loc='best')
        p2.set_title("VAL KS = {:.2f}%".format(val_ks * 100),fontsize=10)
        
        p3.plot(oft1_gain['cum_bad%'])
        p3.plot(oft1_gain['cum_good%'])
        p3.legend(loc='best')
        p3.set_title("OFT1 KS = {:.2f}%".format(oft1_ks * 100),fontsize=10)
        
        p4.plot(oft2_gain['cum_bad%'])
        p4.plot(oft2_gain['cum_good%'])
        p4.legend(loc='best')
        p4.set_title("OFT2 KS = {:.2f}%".format(oft2_ks * 100),fontsize=10)
        
        p5.plot(oft3_gain['cum_bad%'])
        p5.plot(oft3_gain['cum_good%'])
        p5.legend(loc='best')
        p5.set_title("OFT3 KS = {:.2f}%".format(oft3_ks * 100),fontsize=10)
        
        plt.savefig('1.jpg', bbox_inches='tight')
        plt.show()
        
        # Gains 曲线
        plt.figure(figsize=(4, 3),dpi=100)
        plt.plot(dev_gain['cum_bad%'], label='DEV')
        plt.plot(val_gain['cum_bad%'], label='VAL')
        plt.plot(oft1_gain['cum_bad%'], label='OFT1')
        plt.plot(oft2_gain['cum_bad%'], label='OFT2')
        plt.plot(oft3_gain['cum_bad%'], label='OFT3')
        plt.legend(loc='best')
        plt.title("GAINS CHART", fontsize=10)
        plt.savefig("gain.jpg")
        plt.show()
        
        worksheet.insert_image('A19','1.jpg')
        worksheet.insert_image('A36','gain.jpg')
        
        if type == 'GBDT':
            #重要性排序并输出
            self.feature_importance = pd.DataFrame({
                    '变量名':self.varlist,
                    '类型':self.var_type,
                    # '重要度':classifier.steps[1][1].feature_importances_
                    '重要度': classifier.named_steps['GBDT'].feature_importances_.tolist()
                    }).sort_values(by='重要度', ascending=False).reset_index(drop=True)
            print('最重要的变量列表:\n', self.feature_importance[:40])
        worksheet1 = workbook.add_worksheet('varible')
        worksheet1.write_row('A1', list(self.feature_importance.columns) ,cell_format_bold)
        worksheet1.write_column('A2', self.feature_importance['变量名'],cell_format1)
        worksheet1.write_column('B2', self.feature_importance['类型'],cell_format1)
        worksheet1.write_column('C2', self.feature_importance['重要度'],cell_format1)

        if bivar_flag == 1: 
            # 输出连续型变量的单变量分析, 输出10个变化图
            if plot_vars_num=='null':
                plot_vars_num=len(list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名']))
            self.bivar(df = dev, 
                       var_x = list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       tgt = self.tgt,
                       plot_need_show=True,
                       tag1 = 'dev')
            self.bivar(df = val, 
                       var_x = list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       tgt = self.tgt,
                       plot_need_show=True,
                       tag1 = 'val')
            self.bivar(df = oft1, 
                       var_x = list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       tgt = self.tgt,
                       plot_need_show=True,
                       tag1 = 'oft1')
            self.bivar(df = oft2, 
                       var_x = list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       tgt = self.tgt,
                       plot_need_show=True,
                       tag1 = 'oft2')
            self.bivar(df = oft3, 
                       var_x = list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       tgt = self.tgt,
                       plot_need_show=True,
                       tag1 = 'oft3')
            worksheet2 = workbook.add_worksheet('bivar')
            worksheet2.write('A1','DEV BIVAR')
            worksheet2.write('J1','VAL BIVAR')
            worksheet2.write('S1','OFT1 BIVAR')
            worksheet2.write('AB1','OFT2 BIVAR')
            worksheet2.write('AK1','OFT3 BIVAR')
            for plot_num in range(plot_vars_num):
                worksheet2.insert_image('A'+str(plot_num*26+3),'bivar'+str(plot_num+1) + '_dev.jpg')
                worksheet2.insert_image('J'+str(plot_num*26+3),'bivar'+str(plot_num+1) + '_val.jpg')
                worksheet2.insert_image('S'+str(plot_num*26+3),'bivar'+str(plot_num+1) + '_oft1.jpg')
                worksheet2.insert_image('AB'+str(plot_num*26+3),'bivar'+str(plot_num+1) + '_oft2.jpg')
                worksheet2.insert_image('AK'+str(plot_num*26+3),'bivar'+str(plot_num+1) + '_oft3.jpg')
                if plot_num%5==0:
                    print("{}/{} is completed".format(plot_num+1,plot_vars_num))
        

        # 输出离散型变量的单变量分析
        # self.lielian(sample = sample_trt, lielian_list=list(self.feature_importance.loc[self.feature_importance['类型']=='string', '变量名'])[:3])
        workbook.close()
        os.system("rm *.jpg")

    
    def lielian(self, sample, lielian_list, path3="lielian.csv"):
        # 字符型变量、分类变量的列联表和作图：
        # 删除已存在的csv文件
        try:
            os.remove(path3)
        except Exception as e:
            print(e)
        with bk.PdfPages("lielian.pdf") as pdf: #打开pdf文件
            for i in lielian_list: #列联表作图并保存
                try:
                    a = pd.crosstab(sample[i], sample[self.tgt], margins=True)  #生成列联表
                    #a = pd.crosstab(sample[i], sample[tgt], dropna=False, margins=True)
                    a['ratio']=a[1]/a['All']
                    df_name = DataFrame(columns=[i])
                    df_empty = DataFrame()
                    df_empty.to_csv(path3, mode='a+')   #插入空行
                    df_name.to_csv(path3, mode='a+')    #插入字段名
                    a.to_csv(path3,mode='a+')
                    plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签 
                    plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号
                    fig = plt.figure(tight_layout=True)
                    ax = a['All'].plot(kind='bar', use_index=True)
                    ax2 = ax.twinx()
                    ax2.plot(a['ratio'].values, linestyle='-', marker='o', linewidth=2.0)
                    pdf.savefig(fig)
                    plt.savefig('test1.jpg')
                except Exception as e:
                    print(e)
                    print("This Var is not Available---", i)
    
    def use_smote(self, sample_trt, Y, varlist, N=100, k=5):
        '''
        Smote 方法过采样
        '''
        #保留要用的字段，此处应该全是数值型
        sample_1 = sample_trt[varlist][sample_trt[Y] == 1]
        sample_0 = sample_trt[varlist][sample_trt[Y] == 0]
        #过采样
        s1 = sample_1.as_matrix()
        s=Smote(s1, N=N, k=k)
        s2 = s.over_sampling()
        
        #这里取所有的字段名，用于后续拼接。
        col = sample_1.columns

        #查看是否过采样改变了分布：待看
        #合并
        sample_1_oversmpl = pd.DataFrame(s2, columns=col)
        sample_oversmpl = pd.concat([sample_0, sample_1_oversmpl])

        return sample_oversmpl
    
    # 传鹏psi函数
    def calc_psi(self, dev, val, k=10):
        dev = pd.Series(dev)
        val = pd.Series(val)
        build = dev.sort_values(ascending=False).reset_index(drop=True)
        vali = val.sort_values(ascending=False).reset_index(drop=True)
        num = len(build) // k
        val_ite = 1 / k
        # 取分段的边界值，max和min分别取验证样本的最大最小值，而中间元素分别取建模样本的分界值
        sep = [vali[0]] + [build[num * (i + 1)] for i in range(k - 1)] + [vali[len(vali) - 1]]
        # 将验证样本根据建模样本的分段来分组，存入my_vali中
        my_build = []
        my_vali = []
        for j in range(len(sep) - 1):
            tmp_list = []
            for my_i in range(len(vali)):
                if j < len(sep) - 2:
                    if sep[j] >= vali[my_i] > sep[j + 1]:
                        tmp_list.append(vali[my_i])
                else:
                    if sep[j] >= vali[my_i] >= sep[j + 1]:
                        tmp_list.append(vali[my_i])
            my_vali.append(tmp_list)

        for j in range(len(sep) - 1):
            tmp_list = []
            for my_i in range(len(build)):
                if j < len(sep) - 2:
                    if sep[j] >= build[my_i] > sep[j + 1]:
                        tmp_list.append(build[my_i])
                else:
                    if sep[j] >= build[my_i] >= sep[j + 1]:
                        tmp_list.append(build[my_i])
            my_build.append(tmp_list)

        actual_val = [len(my_vali[i]) / len(vali) for i in range(k)]
        actual_build = [len(my_build[i]) / len(build) for i in range(k)]
        # 除数为0的情况
        psi = []
        for i in range(k):
            if actual_val[i] == 0:
                psi_cut = 1 # 惩罚项
            elif actual_build[i] == 0:
                psi_cut = 1
            else:
                psi_cut = (actual_build[i] - actual_val[i]) * np.log(actual_build[i] / actual_val[i])
            psi.append(psi_cut)
        # psi=[(0.05 - actual[i]) * np.log(0.05 / actual[i]) for i in range(k)]
        # 返回actual值，做后期处理
        return sum(psi), actual_val
    
    # 传鹏单变量分析函数
    # 传鹏单变量分析函数，已修改bad_rate计算、画图不全问题
    def bivar( self, df, var_x, tgt, plot_need_show=False, tag1='dev'):
        tag = 0
        for var in var_x:
            
#             print(">>>>",df[var].dtype, var)
            
            variable = df[[var, tgt]].sort_values(by=var)
            variable = variable.reset_index(drop=True)
            tag = tag+1
            var_target = variable[tgt]
            var_number = variable[var]

            var_num = len(variable[var].value_counts())  # 观察有多少个不同的值

            if (var_num >= 5)&(variable[var].dtype!='O'):
                var_tag=pd.cut(variable[var],10)
#                 print('var_tag:\n',var_tag)
                var_sum0=Counter(var_tag.tolist())
                var_sum=pd.DataFrame(list(var_sum0.most_common()))
                var_sum.rename(columns={0:'level',1:'value'},inplace=True)
#                 print('var_sum:\n',var_sum)
                var_tag=pd.DataFrame({'level':var_tag.tolist(),'flag':variable[tgt]})
                var_tag=pd.pivot_table(var_tag,index=['level'],values=['flag'],aggfunc=lambda x:sum(x)/len(x))
                var_tag=var_tag.reset_index()
#                 print('var_tag:\n',var_tag)
                var_sum=pd.merge(var_sum,var_tag,on='level')
                var_sum.sort_values(axis=0, inplace=True, by='level')
#                 print('var_sum:\n',var_sum)

                #x_list=[var_len]*10
                plt.tick_params(axis='x')
                plt.bar(range(1,len(var_sum['value'])+1), var_sum['value'], color='lightcoral', tick_label=var_sum['level'], label="bivar_" + str(var) + "_num")  # plt.legend(loc = 'best')
                plt.xticks(rotation=30)
                plt.twinx()
                plt.plot(range(1,len(var_sum['value'])+1), var_sum['flag'], linestyle='-', marker='o', linewidth=2.0, label= str(var) + "_bad_rate")
                plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
                plt.legend(loc='best')
                plt.title('Analysis of ' + str(var), fontweight='bold')
                plt.savefig('bivar' + str(tag) + '_' + tag1 + '.jpg')
                if plot_need_show:
                    plt.show()
            else:
                val_dict = df[var].value_counts().reset_index()
                val_dict.rename(columns={'index':'level','value':'value'},inplace=True)
                val_dict.sort_values(by='level',inplace=True)
                val_dict = val_dict.reset_index(drop=True)
#                 print("val_dict:\n",val_dict)
                val_sum=pd.pivot_table(variable,index=[var],values=[tgt],aggfunc=lambda x:sum(x)/len(x))
                val_sum=val_sum.reset_index()
                val_sum.rename(columns={var:'level',tgt:'rate'},inplace=True)
#                 print("val_sum before merge:\n",val_sum)
                val_sum=pd.merge(val_sum,val_dict,on='level')
                val_sum.sort_values(axis=0,inplace=True,by='level')
#                 print("val_sum:\n",val_sum)

#                 plt.figure(figsize=(7, 5))
                plt.bar(val_sum['level'], val_sum['rate'], color='lightcoral', tick_label=val_sum['level'], label="bivar_" + var + "_num")  
                plt.xticks(rotation=30)
                plt.twinx()
                plt.plot(val_sum['level'], val_sum['rate'], linestyle='-', marker='o', linewidth=2.0, label="bivar_" + var + "_rate")
                plt.legend(loc='best')
                plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9)
                plt.title('Analysis of ' + str(var), fontweight='bold')
                plt.savefig('bivar' + str(tag) + '_' + tag1 + '.jpg')
                if plot_need_show:
                    plt.show()

class model_train(object):
    


    def __init__(self, numlst="numerical_varlist.csv", charlst="char_varlist.csv", tgt='target'):
        # 字符型变量列表
        self.charlist = func.readCSV(path=charlst).buffer
        # 数值型变量列表
        self.numlist = func.readCSV(path=numlst).buffer
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
        self.statistics = sample.describe().T
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
        dev_woe = func.string2numeric_hm(data_train=tmp,
                              Y=self.tgt,
                              data_apply=dev,
                              stringColumeName=self.charlist)
        val_woe = func.string2numeric_hm(data_train=tmp,
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
#         df_ks, df_auc, df_gain = func.ks_report(df_y.values, df_score.T[1], bins=bins)
        df_ks, df_auc, df_gain = func.ks_report_bal(df_y.values, df_score.T[1], df[bal].values, bins=bins)       
        #print 1% capture Rate
        if pr==1:
            print("前1%抓取: ", df_gain.iloc[0])
            print("ks: ", df_ks, "auc: ", df_auc)
        return df_gain

    def normal_reports(self, df, classifier, bins=20, type='GBDT', pr=1):
        df_x, df_y = self.x_y_split(df)
        df_score = classifier.predict_proba(df_x)
        df_ks, df_auc, df_gain = func.ks_report(df_y.values, df_score.T[1], bins=bins)       
        return df_ks

    def some_reports(self, dev, val, oft, classifier, plot_vars_num='null', bins=20, type='GBDT', psi_flag=0, bivar_flag=1, filename='./output.xlsx'):
        dev_x, dev_y = self.x_y_split(dev)
        val_x, val_y = self.x_y_split(val)
        oft_x, oft_y = self.x_y_split(oft)
        dev_score = classifier.predict_proba(dev_x)
        val_score = classifier.predict_proba(val_x)
        oft_score = classifier.predict_proba(oft_x)
        
        # psi 分布图
        plt.figure(figsize=(5, 3),dpi=150)
        plt.hist(dev_score[:,1], density=True, histtype='step', bins=20, label='DEV')
        plt.hist(val_score[:,1], density=True, histtype='step', bins=20, label='VAL')
        plt.hist(oft_score[:,1], density=True, histtype='step', bins=20, label='OFT')
        plt.title('psi curve')
        plt.legend(loc='best')
        plt.xlabel('score')
        plt.savefig('test1.jpg', bbox_inches='tight')
        plt.show()
        

        
        workbook = xlsxwriter.Workbook(filename)
        worksheet = workbook.add_worksheet('demo')
        bold = workbook.add_format({'bold':True})
        
        worksheet.insert_image('A1','test1.jpg')
        
        if psi_flag==1:
                # 计算psi
            psi_dev_val, actual_dev_val = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(val_score[:,1]), k=20)
            psi_dev_oft, actual_dev_oft = self.calc_psi(pd.Series(dev_score[:,1]),pd.Series(oft_score[:,1]), k=20)
            print(psi_dev_val)
            print(psi_dev_oft)
            worksheet.write('A16', 'psi_val:{}'.format(psi_dev_val))
            worksheet.write('A17', 'psi_oft:{}'.format(psi_dev_oft))
        
        #　KS数据输出到 xlsx
        dev_ks, dev_auc, dev_gain = func.ks_report(dev_y.values, dev_score.T[1], bins=bins)
        val_ks, val_auc, val_gain = func.ks_report(val_y.values, val_score.T[1], bins=bins)
        oft_ks, oft_auc, oft_gain = func.ks_report(oft_y.values, oft_score.T[1], bins=bins)
        print("DEV Sample - KS={:.2f}%, AUC={:.2f}%".format(dev_ks*100, dev_auc*100))
        
        worksheet.write('A52','DEV Sample - KS={:.2f}%'.format(dev_ks*100))
        worksheet.write('B52','AUC={:.2f}%'.format(dev_auc*100))
        worksheet.write_row('A53', list(dev_gain.columns), bold)
        worksheet.write_column('A54', dev_gain['cum_bad_num'])
        worksheet.write_column('B54', dev_gain['cum_good_num'])
        worksheet.write_column('C54', dev_gain['cum_bad%'])
        worksheet.write_column('D54', dev_gain['cum_good%'])
        worksheet.write_column('E54', dev_gain['avg_score'])
        worksheet.write_column('F54', dev_gain['KS_value'])
        
        worksheet.write('A76','VAL Sample - KS={:.2f}%'.format(val_ks*100))
        worksheet.write('B76','AUC={:.2f}%'.format(val_auc*100))
        worksheet.write_row('A77', list(val_gain.columns), bold)
        worksheet.write_column('A78', val_gain['cum_bad_num'])
        worksheet.write_column('B78', val_gain['cum_good_num'])
        worksheet.write_column('C78', val_gain['cum_bad%'])
        worksheet.write_column('D78', val_gain['cum_good%'])
        worksheet.write_column('E78', val_gain['avg_score'])
        worksheet.write_column('F78', val_gain['KS_value'])
        
        worksheet.write('A100','OFT Sample - KS={:.2f}%'.format(oft_ks*100))
        worksheet.write('B100','AUC={:.2f}%'.format(oft_auc*100))
        worksheet.write_row('A101', list(oft_gain.columns), bold)
        worksheet.write_column('A102', oft_gain['cum_bad_num'])
        worksheet.write_column('B102', oft_gain['cum_good_num'])
        worksheet.write_column('C102', oft_gain['cum_bad%'])
        worksheet.write_column('D102', oft_gain['cum_good%'])
        worksheet.write_column('E102', oft_gain['avg_score'])
        worksheet.write_column('F102', oft_gain['KS_value'])
        
        print(dev_gain)
        print("VAL Sample - KS={:.2f}%, AUC={:.2f}%".format(val_ks*100, val_auc*100))
        print(val_gain)
        print("OFT Sample - KS={:.2f}%, AUC={:.2f}%".format(oft_ks*100, oft_auc*100))
        print(oft_gain)

        # KS 曲线
        # 加入为 0 的行
        dev_gain.index = range(1,len(dev_gain) + 1) 
        dev_gain.loc[0] = 0
        dev_gain = dev_gain.sort_index(ascending=True)
        val_gain.index = range(1,len(val_gain) + 1) 
        val_gain.loc[0] = 0
        val_gain = val_gain.sort_index(ascending=True)
        oft_gain.index = range(1,len(oft_gain) + 1) 
        oft_gain.loc[0] = 0
        oft_gain = oft_gain.sort_index(ascending=True)
        plt.figure(figsize=(12, 3),dpi=100)
        p1 = plt.subplot(131)
        p2 = plt.subplot(132)
        p3 = plt.subplot(133)
        # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=0.2, wspace=0.3)
        p1.plot(dev_gain['cum_bad%'])
        p1.plot(dev_gain['cum_good%'])
        p1.legend(loc='best')
        p1.set_title("DEV KS = {:.2f}%".format(dev_ks * 100),fontsize=10)
        
        p2.plot(val_gain['cum_bad%'])
        p2.plot(val_gain['cum_good%'])
        p2.legend(loc='best')
        p2.set_title("VAL KS = {:.2f}%".format(val_ks * 100),fontsize=10)
        
        p3.plot(oft_gain['cum_bad%'])
        p3.plot(oft_gain['cum_good%'])
        p3.legend(loc='best')
        p3.set_title("OFT KS = {:.2f}%".format(oft_ks * 100),fontsize=10)
        
        plt.savefig('1.jpg', bbox_inches='tight')
        plt.show()
        
        # Gains 曲线
        plt.figure(figsize=(4, 3),dpi=100)
        plt.plot(dev_gain['cum_bad%'], label='DEV')
        plt.plot(val_gain['cum_bad%'], label='VAL')
        plt.plot(oft_gain['cum_bad%'], label='OFT')
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
        worksheet1.write_row('A1', list(self.feature_importance.columns), bold)
        worksheet1.write_column('A2', self.feature_importance['变量名'])
        worksheet1.write_column('B2', self.feature_importance['类型'])
        worksheet1.write_column('C2', self.feature_importance['重要度'])

        if bivar_flag == 1: 
            # 输出连续型变量的单变量分析, 输出10个变化图
            if plot_vars_num=='null':
                plot_vars_num=len(list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名']))
            self.bivar(dev, 
                       list(self.feature_importance.loc[self.feature_importance['类型'].isin(['number','string']), '变量名'])[:plot_vars_num], 
                       'tgt',
                       plot_need_show=True)
            worksheet2 = workbook.add_worksheet('bivar')
            for plot_num in range(plot_vars_num):
                worksheet2.insert_image('A'+str(plot_num*26+1),'bivar'+str(plot_num+1) + '_' + tag1+'.jpg')
                if plot_num%5==0:
                    print("{}/{} is completed".format(plot_num+1,plot_vars_num))

        # 输出离散型变量的单变量分析
        # self.lielian(sample = sample_trt, lielian_list=list(self.feature_importance.loc[self.feature_importance['类型']=='string', '变量名'])[:3])
        workbook.close()

    
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
            
            print(">>>>",df[var].dtype, var)
            
            variable = df[[var, tgt]].sort_values(by=var)
            variable = variable.reset_index(drop=True)
            tag = tag+1
            var_target = variable[tgt]
            var_number = variable[var]

            var_num = len(variable[var].value_counts())  # 观察有多少个不同的值

            if (var_num >= 10)&(variable[var].dtype!='O'):
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
                print("val_dict:\n",val_dict)
                val_sum=pd.pivot_table(variable,index=[var],values=[tgt],aggfunc=lambda x:sum(x)/len(x))
                val_sum=val_sum.reset_index()
                val_sum.rename(columns={var:'level',tgt:'rate'},inplace=True)
                print("val_sum before merge:\n",val_sum)
                val_sum=pd.merge(val_sum,val_dict,on='level')
                val_sum.sort_values(axis=0,inplace=True,by='level')
                print("val_sum:\n",val_sum)

                plt.figure(figsize=(7, 5))
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

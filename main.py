# -*- coding=utf-8 -*-
import os
import time
import jieba
import jieba.analyse
from jieba import analyse
import operator
import gensim.models.word2vec as w2v

'''
关键词：卷积核，感受野，池化层 max-pooling method
'''

def cut_txt(old_file):
    import jieba
    global cut_file     # 分词之后保存的文件名
    cut_file = old_file + '_cut.txt'

    try:
        fi = open(old_file, 'r', encoding='utf-8')
    except BaseException as e:  # 因BaseException是所有错误的基类，用它可以获得所有错误类型
        print(Exception, ":", e)    # 追踪错误详细信息

    text = fi.read()  # 获取文本内容
    new_text = jieba.cut(text, cut_all=False)  # 精确模式
    str_out = ' '.join(new_text).replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
        .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
        .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
        .replace('’', '')     # 去掉标点符号
    fo = open(cut_file, 'w', encoding='utf-8')
    fo.write(str_out)
	
def model_train(train_file_name):  # model_file_name为训练语料的路径,save_model为保存模型名
    from gensim.models import word2vec
    import gensim
    import logging
    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料
    model = gensim.models.Word2Vec(sentences, size=5, window=5, workers=1, min_count=1)  # 训练skip-gram模型; 默认window=5
#    model.save(save_model_file)
#    model.wv.save_word2vec_format(save_model_name + ".bin", binary=True)   # 以二进制类型保存模型以便重用
    return model

#获取数据模型构建词向量
def get_model(model_file):
    cut_txt(model_file)
    model = model_train(cut_file)
    return model
	
	
	
#CNN获取输入层数据
def get_input_data(input_sen,compare_sen,wordvec_model):
    score=[]
    wordvec_table=[]
    keyword=analyse.extract_tags(input_sen)
    testword=analyse.extract_tags(compare_sen)
    for i in keyword:
        for j in testword:
            try:
                sim=wordvec_model.similarity(i,j)
                score.append(sim)
            except Exception as err:
                pass
        if len(score)==0:
            pass
        else:
            wordvec_table.append(score)
        score=[]
    return wordvec_table

#CNN卷积层,卷积核的感受野为2*2
def cnn_folding(dict_wordvec):
    rows=len(dict_wordvec[0])
    columns=len(dict_wordvec)
    result=[]
    result=[[0 for col in range(rows-1)] for row in range(columns-1)]
    for i in range(columns-1):
        for j in range(rows-1):
            re=(dict_wordvec[i][j]+dict_wordvec[i][j+1]+dict_wordvec[i+1][j]+dict_wordvec[i+1][j+1])/4     
            result[i][j]=re
    dict_wordvec=result
    return result

#cnn池化层，采用max-pooling方式实现池化
def cnn_pooling(dict_pooling):
    rows=len(dict_pooling[0])
    columns=len(dict_pooling)
    result=[]
    result=[[0 for col in range(rows-1)] for row in range(columns-1)]
    for i in range(columns-1):
        for j in range(rows-1):
            re=max(dict_pooling[i][j],dict_pooling[i][j+1],dict_pooling[i+1][j],dict_pooling[i+1][j+1])
            result[i][j]=re
    return result

#实现卷积层和池化层的连接层
def cnn_folding_pooling(data_list):
    res=[]
    while 1:
        r=len(data_list[0])
        c=len(data_list)
        if r==1 or c==1:
            for i in range(len(data_list)):
                for j in data_list[i]:
                    res.append(j)
            break
        pool=cnn_pooling(data_list)
        if len(pool)==1 or len(pool[1])==1:
            data_list=pool
            for i in range(len(data_list)):
                for j in data_list[i]:
                    res.append(j)
            break
        else:
            fold=cnn_folding(pool)
            data_list=fold
            pool=[[0 for col in range(r-1)] for row in range(c-1)]
            fold=[[0 for col in range(r-1)] for row in range(c-1)]
    return res
        
    
#对文本数据进行相似度比较
def get_sim(file_path,file_path1,input_sen):
    a=0
    score=[]
    score_sort=[]
    wordvec_model=get_model(file_path)
    print('模型构建成功')
    f=open(file_path1,'r')
    while 1:
        re=[]
        line=f.readline()
        if not line:
            break
        try:
            data_table=get_input_data(input_sen,line,wordvec_model)
            re=cnn_folding_pooling(data_table)
            a=a+1
            sc=0
            s=0
            for k in re:
                sc=sc+k
            s=sc/len(re)
            score.append((line,s))
        except Exception as err:
            pass
    score.sort(key=operator.itemgetter(1),reverse=True)
    score_sort.append(score[0])
    score_sort.append(score[1])
    score_sort.append(score[2])
    score_sort.append(score[3])
    score_sort.append(score[4])
    score_sort.append(score[5])
    score_sort.append(score[6])
    score_sort.append(score[7])
    score_sort.append(score[8])
    return score_sort

#主函数
if __name__=='__main__':
    time1=time.time()
    model=get_model("text.txt")
    print('模型构建成功')
    title = []
    content = []
    title_file = "C://Users//user01//Desktop//title.txt"
    f = open(title_file,'r')
    line = f.readlines()
    num = 0
    for i in line:
        title.append(i)
        num += 1

    content_file = "C://Users//user01//Desktop//content.txt"
    content_f = open(content_file,'r')
    content_line = content_f.readlines()
    for j in content_line:
        content.append(j)

    result_file = open('C://Users//user01//Desktop//result.txt','a')
    count = 0
    while count < num:
        a = title[count]
        b = content[count]
        count += 1
        print('cnn输入层数据：')
        data=get_input_data(a,b,model)
        print(data)
        print("输入层数据的维度大小：")
        print(str(len(data))+'*'+str(len(data[0])))
        print('卷积层卷积一次输出：')
        print(cnn_folding(data))
        print("卷积层数据的维度大小：")
        print(str(len(cnn_folding(data)))+'*'+str(len(cnn_folding(data)[0])))
        print("池化层池化一次输出：")
        print(cnn_pooling(cnn_folding(data)))
        print("池化层数据的维度大小：")
        print(str(len(cnn_pooling(cnn_folding(data))))+'*'+str(len(cnn_pooling(cnn_folding(data))[0])))
        c=cnn_folding_pooling(data)
        print('cnn 输出层：')
        print(c)
        score=0
        for k in c:
            score=score+k
        print("模拟结果：")
        print(score/len(c))
        time2=time.time()
        print('运行时长：')
        print(time2-time1)
        result_file.write(str(score/len(c)))
        result_file.write('\n')
    result_file.close()

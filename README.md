# dynamic-BeamSearch-seq2seq
### 基于中文语料和dynamic_rnn+BeamSearch+seq2seq chatbot 



### Requirements
- tensorflow-1.3+
- python3
- jieba

---


对话语料分别在**data**目录下 Q.txt A.txt中，可以替换成你自己的对话语料。    

---

### 用法:
    
    # 新增小黄鸡语料
    # 添加
    python prepare_dialog.py 5000


    #数据预处理
    python progress.py

    # 训练
    python seq2seq.py train

    # 预测
    python seq2seq.py infer

   

### 效果:
    

###  参考：
https://ask.hellobi.com/blog/wenwen/11367
https://github.com/yanwii/dynamic-seq2seq
https://blog.csdn.net/thriving_fcl/article/details/74165062

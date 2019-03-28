# dynamic-BeamSearch-seq2seq
### 基于中文语料和dynamic_rnn+BeamSearch+seq2seq chatbot 



### Requirements
- tensorflow-1.9.0
- python3
- jieba

gtx-1080ti
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
用的全量数据45w问答对。问答对质量不怎么好。
仅仅是为了学习使用。seq2seq的问答效果实在是太差了。
用了beamsearch  beam_size=2，所以有2个答案。
训练了很久，loss在2-3左右。

me> 你是傻逼
AI> 你才是傻逼__EOS____EOS__
AI> 你才是傻逼！__EOS__
me> 你是天才
AI> = =__EOS____EOS__
AI> 我是小通__EOS__
me> 我爱你
AI> 我爱你__EOS____EOS__
AI> 我也爱你__EOS__
me> 我喜欢你
AI> 我也喜欢你__EOS__
AI> 我喜欢你__EOS____EOS__
me> 呵呵呵
AI> =。=__EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS____EOS__
AI> 呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵呵
me> 你是谁
AI> 我是小通__EOS____EOS____EOS__
AI> 我是你的小通__EOS__
me>


###  参考：
https://ask.hellobi.com/blog/wenwen/11367
https://github.com/yanwii/dynamic-seq2seq
https://blog.csdn.net/thriving_fcl/article/details/74165062
